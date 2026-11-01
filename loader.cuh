// Host side of the model load (included by qwen38_mk.cu inside its anonymous namespace): the phase clock, parallel
// read-ahead of the GGUF and reusable host buffers. Nothing here changes the bytes the loader produces.
#pragma once

// (<condition_variable>, <memory>, <mutex>: included by qwen38_mk.cu at file scope; this header sits inside its anonymous namespace)

using lclk = std::chrono::steady_clock;
inline double lsec(lclk::time_point t) { return std::chrono::duration<double>(lclk::now() - t).count(); }

// Seconds per loader phase, in first-use order (MK_LOADLOG=1 prints the table at the end of the load).
struct LoadClock {
    std::vector<std::pair<std::string, double>> ph;
    std::mutex mu;
    void add(const char * k, double s) {
        std::lock_guard<std::mutex> l(mu);
        for (auto & p : ph)
            if (p.first == k) {
                p.second += s;
                return;
            }
        ph.emplace_back(k, s);
    }
    double get(const char * k) {
        std::lock_guard<std::mutex> l(mu);
        for (auto & p : ph)
            if (p.first == k) return p.second;
        return 0.0;
    }
    struct Scope {
        LoadClock * c;
        const char * k;
        lclk::time_point t;
        ~Scope() { c->add(k, lsec(t)); }
    };
    Scope scope(const char * k) { return Scope{this, k, lclk::now()}; }
};

// Parallel read-ahead of the GGUF's tensor data (MK_LOAD_IO threads, default 8; 0: off). Under WSL2 one thread reading a
// cold file through the mmap gets ~0.2 GB/s (4 KiB faults, small readahead), 8 threads touching 4 MiB pieces ~3 GB/s.
// The pieces go in the loader's order (blk.<i> by layer, then the rest; by offset within a
// group), at most `window` bytes past the furthest piece the loader has asked for (need), so the read-ahead cannot evict
// pages the loader has not reached yet. The touching threads also map the pages into the process: the loader's own
// accesses then take no faults.
class Prefetch {
  public:
    static constexpr size_t PIECE = 4u << 20;
    // keep (optional): the tensors to read ahead (the MTP block of a --mtp file; the main file without the blocks it
    // does not supply)
    Prefetch(const gguf::File & f, int nth, size_t window, const std::function<bool(const std::string &)> & keep = nullptr)
        : win_(std::max<size_t>(1, window / PIECE)) {
        std::vector<const gguf::Tensor *> v;
        for (const auto & kv : f.tensors)
            if (!keep || keep(kv.first)) v.push_back(&kv.second);
        auto layer = [](const std::string & n) { return n.compare(0, 4, "blk.") == 0 ? atoi(n.c_str() + 4) : 1 << 20; };
        std::sort(v.begin(), v.end(), [&](const gguf::Tensor * a, const gguf::Tensor * b) {
            const int la = layer(a->name), lb = layer(b->name);
            return la != lb ? la < lb : a->data < b->data;
        });
        for (const gguf::Tensor * t : v) {
            const size_t a = pc_.size();
            for (size_t o = 0; o < t->nbytes; o += PIECE) pc_.push_back({t->data + o, std::min(PIECE, (size_t) t->nbytes - o)});
            ix_[t->data] = {a, pc_.size()};
        }
        done_.reset(new std::atomic<uint8_t>[pc_.size() + 1]);
        for (size_t i = 0; i <= pc_.size(); i++) done_[i] = 0;
        for (int i = 0; i < nth; i++) th_.emplace_back([this] { run(); });
    }
    ~Prefetch() {
        {
            std::lock_guard<std::mutex> l(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto & t : th_) t.join();
    }
    // returns once every byte of t has been read (touched) at least once
    void need(const gguf::Tensor & t) {
        const auto it = ix_.find(t.data);
        if (it == ix_.end()) return;
        const size_t a = it->second.first, b = it->second.second;
        bool all = true;
        for (size_t i = a; i < b && all; i++) all = done_[i] != 0;
        if (all) return;
        const auto t0 = lclk::now();
        std::unique_lock<std::mutex> l(mu_);
        if (b > want_) {
            want_ = b;
            cv_.notify_all();
        }
        cvd_.wait(l, [&] {
            for (size_t i = a; i < b; i++)
                if (!done_[i]) return false;
            return true;
        });
        wait_s += lsec(t0);
    }
    double wait_s = 0; // time need() waited (the loader's I/O stall)
    size_t bytes() const {
        size_t n = 0;
        for (const auto & p : pc_) n += p.n;
        return n;
    }

  private:
    struct Piece {
        const uint8_t * p;
        size_t n;
    };
    std::vector<Piece> pc_;
    std::map<const uint8_t *, std::pair<size_t, size_t>> ix_; // tensor data -> pieces [a, b)
    std::unique_ptr<std::atomic<uint8_t>[]> done_;
    size_t next_ = 0, want_ = 0;
    const size_t win_;
    bool stop_ = false;
    std::mutex mu_;
    std::condition_variable cv_, cvd_;
    std::vector<std::thread> th_;
    void run() {
        for (;;) {
            size_t i;
            {
                std::unique_lock<std::mutex> l(mu_);
                cv_.wait(l, [&] { return stop_ || (next_ < pc_.size() && next_ < want_ + win_); });
                if (stop_ || next_ >= pc_.size()) return;
                i = next_++;
            }
            const Piece & p = pc_[i];
            unsigned s = 0;
            for (size_t o = 0; o < p.n; o += 4096) s += p.p[o];
            s += p.p[p.n - 1];
            sink_.fetch_add(s, std::memory_order_relaxed); // (keeps the reads)
            {
                std::lock_guard<std::mutex> l(mu_);
                done_[i] = 1;
            }
            cvd_.notify_all();
        }
    }
    std::atomic<unsigned> sink_{0};
};

// A host buffer reused across tensors (grows only, never zero-filled: the repacks write every byte they upload).
// Fresh std::vector<uint8_t>(n) buffers cost a single-threaded zero fill + page faults per tensor (~16 GB per load).
struct HBuf {
    std::unique_ptr<uint8_t[]> p;
    size_t cap = 0, n = 0;
    uint8_t * get(size_t want) {
        if (want > cap) {
            p.reset(new uint8_t[want]);
            cap = want;
        }
        n = want;
        return p.get();
    }
    uint8_t * data() { return p.get(); }
    size_t size() const { return n; }
};

// Weight uploads on a worker thread (MK_LOAD_ASYNC=0: synchronous cudaMemcpy in the caller). The worker
// copies each piece on its own non-blocking stream and synchronises it, so a source buffer is free again once its
// sequence number is done(); meanwhile the loader prepares the next tensors in its other host buffer set. A failed copy
// is raised in the caller's thread at the next put / wait / finish.
struct Uploader {
    bool async = false;
    cudaStream_t st = nullptr;
    std::thread th;
    std::mutex mu;
    std::condition_variable cv, cvd;
    struct Job {
        void * dst;
        const void * src;
        size_t n;
        void * dst2 = nullptr; // (put2: a second copy, then post() on the stream)
        const void * src2 = nullptr;
        size_t n2 = 0;
        std::function<cudaError_t(cudaStream_t)> post;
    };
    std::vector<Job> q; // FIFO (head index qh)
    size_t qh = 0;
    uint64_t queued = 0, done = 0;
    bool stop = false;
    cudaError_t err = cudaSuccess;
    double busy_s = 0; // time spent in the copies (worker or caller)
    size_t bytes = 0;
    double wd_s = 0; // watchdog (MK_WATCHDOG): a wait that sees no copy finish for wd_s seconds ends the process (0: off)
    void start(bool want_async) {
        if (!want_async) return;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        async = true;
        th = std::thread([this] { run(); });
    }
    void raise() {
        if (err != cudaSuccess) mk_cuda_fail(err, "weight upload (cudaMemcpyAsync)", __FILE__, __LINE__);
    }
    uint64_t put(void * dst, const void * src, size_t n) {
        if (!n) return 0;
        if (!async) {
            const auto t0 = lclk::now();
            CK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
            busy_s += lsec(t0);
            bytes += n;
            return 0;
        }
        std::lock_guard<std::mutex> l(mu);
        raise();
        q.push_back({dst, src, n});
        cv.notify_one();
        return ++queued;
    }
    // two copies and a kernel launch after them on the worker's stream (the device wv re-layout); synchronous mode: the same
    // on stream 0, then a device sync
    uint64_t put2(void * dst, const void * src, size_t n, void * dst2, const void * src2, size_t n2,
                  std::function<cudaError_t(cudaStream_t)> post) {
        if (!async) {
            const auto t0 = lclk::now();
            CK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
            if (n2) CK(cudaMemcpy(dst2, src2, n2, cudaMemcpyHostToDevice));
            CK(post(0));
            CK(cudaStreamSynchronize(0));
            busy_s += lsec(t0);
            bytes += n + n2;
            return 0;
        }
        std::lock_guard<std::mutex> l(mu);
        raise();
        Job j{dst, src, n};
        j.dst2 = dst2;
        j.src2 = src2;
        j.n2 = n2;
        j.post = std::move(post);
        q.push_back(std::move(j));
        cv.notify_one();
        return ++queued;
    }
    // cvd.wait(l, pred), under the watchdog: a GPU or driver that stops completing copies (the runtime's own waits never
    // time out) becomes a message and exit code 3, as Impl::wait_event does for the run-time waits
    template <class P> void wait_done(std::unique_lock<std::mutex> & l, P pred) {
        if (!(wd_s > 0)) {
            cvd.wait(l, pred);
            return;
        }
        uint64_t seen = done;
        auto t0 = lclk::now();
        while (!pred()) {
            cvd.wait_for(l, std::chrono::milliseconds(250));
            if (done != seen) {
                seen = done;
                t0 = lclk::now();
            } else if (!pred() && lsec(t0) > wd_s) {
                fprintf(stderr,
                        "[mk] watchdog: the weight upload finished no copy within %g s (%llu of %llu done). The device or its "
                        "driver is stuck: exiting with code 3.\n",
                        wd_s, (unsigned long long) done, (unsigned long long) queued);
                fflush(stderr);
                std::_Exit(3);
            }
        }
    }
    void wait(uint64_t seq) { // the copies up to seq are done (their sources are free)
        if (!async || !seq) return;
        std::unique_lock<std::mutex> l(mu);
        wait_done(l, [&] { return done >= seq || err != cudaSuccess; });
        raise();
    }
    void finish() { // every copy done; the worker stops
        if (!async) return;
        {
            std::unique_lock<std::mutex> l(mu);
            wait_done(l, [&] { return done >= queued || err != cudaSuccess; });
            stop = true;
        }
        cv.notify_all();
        th.join();
        async = false;
        cudaStreamDestroy(st);
        st = nullptr;
        raise();
    }
    // the worker stops after the copy it is running (the queued ones are dropped): a failed load calls it before the
    // buffers those copies read are freed
    void halt() {
        if (!async) return;
        {
            std::lock_guard<std::mutex> l(mu);
            stop = true;
        }
        cv.notify_all();
        th.join();
        async = false;
        if (st) cudaStreamDestroy(st);
        st = nullptr;
    }
    ~Uploader() { halt(); }
    void run() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> l(mu);
                cv.wait(l, [&] { return stop || qh < q.size(); });
                if (stop) return; // (finish: nothing is left; the destructor of a failed load: the rest is dropped)
                j = std::move(q[qh++]);
            }
            const auto t0 = lclk::now();
            cudaError_t e = err == cudaSuccess ? cudaMemcpyAsync(j.dst, j.src, j.n, cudaMemcpyHostToDevice, st) : err;
            if (e == cudaSuccess && j.n2) e = cudaMemcpyAsync(j.dst2, j.src2, j.n2, cudaMemcpyHostToDevice, st);
            if (e == cudaSuccess && j.post) e = j.post(st);
            if (e == cudaSuccess) e = cudaStreamSynchronize(st);
            {
                std::lock_guard<std::mutex> l(mu);
                busy_s += lsec(t0);
                bytes += j.n + j.n2;
                if (e != cudaSuccess && err == cudaSuccess) err = e;
                done++;
            }
            cvd.notify_all();
        }
    }
};

// The weight digest (MK_WDIGEST=1, load-time gates): one 64-bit hash over every device weight buffer in allocation
// order (size + bytes) and the mapped token_embd table. Device bytes are read back in 64 MiB pieces, hashed on
// several threads (piece hashes combined in order).
inline unsigned long long wd_hash(const uint8_t * b, size_t nb, unsigned long long h = 0xcbf29ce484222325ULL) {
    size_t i = 0;
    for (; i + 8 <= nb; i += 8) {
        unsigned long long w;
        memcpy(&w, b + i, 8);
        h = (h ^ w) * 0x100000001b3ULL;
        h ^= h >> 31;
    }
    for (; i < nb; i++) h = (h ^ b[i]) * 0x100000001b3ULL;
    return h;
}

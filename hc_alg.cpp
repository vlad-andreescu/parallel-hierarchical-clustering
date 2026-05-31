#include "hc.h"
#include "safe_queue.h"
#include <limits>
#include <stdexcept>
#include <thread>
#include <algorithm>
#include <mutex>
#include <condition_variable>

// ─────────────────────────────────────────────────────────────────────────────
//  Algorithm 1 — Naive Sequential HAC
//
//  Complexity:
//    Time:  O(N^3) — N-1 iterations, each scans O(N^2) pairs
//    Space: O(N^2) — the distance matrix
//
//  Indexing convention:
//    Original points have ids 0..N-1.
//    The k-th merge (0-indexed) creates a new cluster with id N+k.
//    The distance matrix is sized (2N-1)x(2N-1) upfront so every
//    cluster id, including merged ones, indexes into it directly.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Merge> naive_hac(const std::vector<Point>& points, Linkage linkage) {
    if (points.empty()) {
        return {};
    }
    const int N = static_cast<int>(points.size());
    const int TOTAL = 2 * N - 1; // max number of distinct cluster ids because we needs ids for when we merge : 
    /*initial clusters: 0, 1, 2, 3
        merge 1 creates cluster 4
        merge 2 creates cluster 5
        merge 3 creates cluster 6
        The k-th merge (0-indexed) creates a new cluster with id N+k!!!!*/

    // 1. Build initial distance matrix
    std::vector<std::vector<double>> D(TOTAL, std::vector<double>(TOTAL, 0.0)); //here we store the distance between cluster i and cluster j. We chose the distance matrix to be  (2N-1)x(2N-1) sized  so every cluster id, including merged ones, is directly in it, and we represent the set of points we're currently dealing w "active" with a vector of booleans, so if 1 merged w 2 and creates cluster N+k then active[N+k] becomes true and the active[1] and active[2] become false 
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            D[i][j] = euclidean(points[i], points[j]);

    // 2. Initialise active set
    std::vector<bool> active(TOTAL, false); //we start w all of them false because we start only w the N points active 
    for (size_t i = 0; i < N; ++i)
        active[i] = true; //tells us if the cluster i exists, initially all the starting points are active and when two points merge, they become inactive and their cluster becomes active 

    std::vector<Merge> dendrogram;
    dendrogram.reserve(N - 1);

    int next_id = N;   // ids for new merged clusters 

    // ── 3. Main loop: exactly N-1 merges ─────────────────────────────────────
    for (size_t step = 0; step < N - 1; ++step) {

        // Step A: find the closest active pair
        int    best_i = -1, best_j = -1;
        double best_d = std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < TOTAL; ++i) {
            if (!active[i]) continue;
            for (size_t j = i + 1; j < TOTAL; ++j) { //j = i+1 avoids checking D[i][j] and D[j][i] since matrix is symmetric 
                if (!active[j]) continue;
                if (D[i][j] < best_d) {
                    best_d = D[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Step B: record the merge
        int new_id = next_id++;
        dendrogram.push_back({best_i, best_j, best_d, new_id});

        // Step C: update active set
        active[best_i] = false;
        active[best_j] = false;
        active[new_id] = true;

        // Step D: update distances to the new cluster via Lance-Williams
        // now we need to compute the distance between the new merged cluster and every other active cluster.
        for (size_t k = 0; k < TOTAL; ++k) {
            if (!active[k] || k == new_id) continue;
            double d = lance_williams(D[best_i][k], D[best_j][k], linkage);
            D[new_id][k] = d;
            D[k][new_id] = d;   // keep the matrix symmetric
        }
    }

    return dendrogram;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Algorithm 2 — pTRAD
// ─────────────────────────────────────────────────────────────────────────────
void BuildDistBlock(size_t beg, size_t end, std::vector<std::vector<double>>& D, size_t N, const std::vector<Point>& points){
    for (size_t i = beg; i < end; i++){
        for (size_t j = 0; j < N; j++){
            D[i][j] = euclidean(points[i], points[j]);
        }
    }
}

void Findlocalmin(size_t beg, size_t end, size_t N, std::vector<std::vector<double>>& D, std::vector<double>& result,  std::vector<bool>& active){
    double best_local_d =  std::numeric_limits<double>::infinity();
    int best_local_i= -1;
    int best_local_j= -1;

    for (size_t i = beg; i < end; i++) {
            if (!active[i]) continue;
            for (size_t j = i + 1; j < 2*N - 1; ++j) { 
                if (!active[j]) continue;
                if (D[i][j] < best_local_d) {
                    best_local_d = D[i][j];
                    best_local_i = i;
                    best_local_j = j;
                }
            }
        }
        result[0] = best_local_d;
        result[1] = best_local_i;
        result[2] = best_local_j;
}

void Updatedist(size_t beg, size_t end, size_t N, std::vector<std::vector<double>>& D, size_t new_id, size_t best_i, size_t best_j, std::vector<bool>& active,  Linkage linkage){
    for (size_t k = beg; k < end; ++k) {
        if (!active[k] || k == new_id) continue;

        double d = lance_williams(D[best_i][k], D[best_j][k], linkage);
        D[new_id][k] = d;
        D[k][new_id] = d;
    }
}

// barrieir:main thread waits for all the other ones to finish
struct Barrier {
    std::mutex mtx;
    std::condition_variable cv;
    int pending = 0;
    void set(int n) { pending = n; }
    void done() {
        std::unique_lock<std::mutex> lock(mtx);
        if (--pending == 0) cv.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return pending == 0; });
    }
};

struct Task {
    enum Type { BUILD, FINDMIN, UPDATE, EXIT } type;
    size_t beg, end;
    int thread_id;
    size_t new_id = 0, best_i = 0, best_j = 0;   // only used for UPDATE
    std::vector<bool>* active = nullptr; //pointer so threads see updates
};

std::vector<Merge> ptrad_hac(const std::vector<Point>& points, Linkage linkage, int num_threads) {
    size_t length = points.size();
    if (length==0){
        return {};
    }
    size_t TOTAL = 2*length - 1;
    size_t block_size = TOTAL/num_threads;
    size_t block_size_build = length/num_threads;
    std::vector<std::vector<double>> results(num_threads, std::vector<double>{std::numeric_limits<double>::infinity(), -1, -1});
    std::vector<std::vector<double>> D(TOTAL, std::vector<double>(TOTAL, 0.0));

    // thread pool
    //spawning new std::thread objects every iteration cost O(N)
    SafeUnboundedQueueCV<Task> queue;
    Barrier barrier;

    std::vector<std::thread> workers(num_threads - 1);
    for (int t = 0; t < num_threads - 1; ++t) {
        workers[t] = std::thread([&]() {
            while (true) {
                Task task = queue.pop();
                if (task.type == Task::EXIT) return;
                if (task.type == Task::BUILD)
                    BuildDistBlock(task.beg, task.end, D, length, points);
                else if (task.type == Task::FINDMIN)
                    Findlocalmin(task.beg, task.end, length, D, results[task.thread_id], *task.active);
                else if (task.type == Task::UPDATE)
                    Updatedist(task.beg, task.end, length, D, task.new_id, task.best_i, task.best_j, *task.active, linkage);
                barrier.done();
            }
        });
    }

    // 1. Build initial distance matrix
    // for (size_t i = 0; i < num_threads-1; ++i){                      
    //     workers[i] = std::thread(BuildDistBlock, ...);                  
    // }

    barrier.set(num_threads - 1);
    for (int t = 0; t < num_threads - 1; ++t)
        queue.push({Task::BUILD, t*block_size_build, (t+1)*block_size_build, t});
    BuildDistBlock((num_threads-1)*block_size_build, length, D, length, points); 
    barrier.wait();

    // for (size_t i = 0; i < num_threads-1; ++i){ workers[i].join(); } 

    for (size_t i = 0; i < length; ++i){
        for (size_t j = 0; j < i; ++j){
            D[j][i] = D[i][j];
        }
    }

    int next_id = length;
    std::vector<bool> active(TOTAL, false);
    for (size_t i = 0; i < length; ++i)
        active[i] = true;

    std::vector<Merge> dendrogram;
    dendrogram.reserve(length- 1);

    for (size_t step = 0; step < length - 1; ++step){ //N-1 merges
        for (int i = 0; i < num_threads; ++i){
            results[i] = {std::numeric_limits<double>::infinity(), -1, -1};
        }

        // for (size_t i = 0; i < num_threads-1; ++i){                     
        //     workers[i] = std::thread(Findlocalmin, ...);               
        // }
        barrier.set(num_threads - 1);
        for (int t = 0; t < num_threads - 1; ++t)
            queue.push({Task::FINDMIN, t*block_size, (t+1)*block_size, t, 0, 0, 0, &active});
        Findlocalmin((num_threads-1)*block_size, TOTAL, length, D, results[num_threads-1], active);   // main thread
        barrier.wait();

        // for (size_t i = 0; i < num_threads-1; ++i){ workers[i].join(); }

        auto best = *std::min_element(results.begin(),results.end(),[](const std::vector<double>& a, const std::vector<double>& b) {return a[0] < b[0];});
        double best_d = best[0];
        int best_i  = static_cast<int>(best[1]);
        int best_j = static_cast<int>(best[2]);

        //record merge in dendro 
        int new_id = next_id++;
        dendrogram.push_back({best_i, best_j, best_d, new_id});
        active[best_i] = false;
        active[best_j] = false;
        active[new_id] = true;

        // for (size_t i = 0; i < num_threads-1; ++i){                     
        //     workers[i] = std::thread(Updatedist, ...);                
        // }
        barrier.set(num_threads - 1);
        for (int t = 0; t < num_threads - 1; ++t)
            queue.push({Task::UPDATE, t*block_size, (t+1)*block_size, t, (size_t)new_id, (size_t)best_i, (size_t)best_j, &active});
        Updatedist((num_threads-1)*block_size, TOTAL, length, D, new_id, best_i, best_j, active, linkage);   //main thread
        barrier.wait();

        // for (size_t i = 0; i < num_threads-1; ++i){ workers[i].join(); 
    }

    // Shutdown: send EXIT sentinel to each worker
    for (int t = 0; t < num_threads - 1; ++t)
        queue.push({Task::EXIT, 0, 0, 0});
    for (int t = 0; t < num_threads - 1; ++t)
        workers[t].join();

    return dendrogram;
    
}

/*Notes : Build initial distances: use length
Find local min: use TOTAL
Update distances: use TOTAL */

// ─────────────────────────────────────────────────────────────────────────────
//  Algorithm 3 — Sequential POP   (coming next)
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
//  Algorithm 4 — pPOP   (coming next)
// ─────────────────────────────────────────────────────────────────────────────
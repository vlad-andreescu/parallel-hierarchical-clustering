#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cmath>
#include <iomanip>
#include "hc.h"
#include "datacleaning.h"
#include "normalization.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Measure wall-clock time of a function call in milliseconds
template<typename Func>
double time_ms(Func f) {
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Check that two dendrograms are identical
bool dendrograms_equal(const std::vector<Merge>& a, const std::vector<Merge>& b) {
    if (a.size() != b.size()) return false;
    for (size_t s = 0; s < a.size(); ++s) {
        if (a[s].i != b[s].i || a[s].j != b[s].j)    return false;
        if (std::abs(a[s].dist - b[s].dist) > 1e-9)   return false;
    }
    return true;
}

// Generate N random-ish points deterministically (no randomness needed for benchmarks)
std::vector<Point> make_points(int N) {
    std::vector<Point> pts;
    pts.reserve(N);
    for (int i = 0; i < N; ++i) {
        // simple deterministic spread
        double x = std::sin(i * 0.1) * 100.0;
        double y = std::cos(i * 0.17) * 100.0;
        pts.push_back({{x, y}});
    }
    return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
//  1. Correctness check
// ─────────────────────────────────────────────────────────────────────────────

void check_correctness() {
    std::cout << "=== 1. Correctness: ptrad_hac vs naive_hac ===\n";

    for (int N : {4, 20, 50}) {
        auto pts = make_points(N);
        for (Linkage l : {Linkage::SINGLE, Linkage::COMPLETE}) {
            for (int P : {1, 2, 4}) {
                auto ref = naive_hac(pts, l);
                auto got = ptrad_hac(pts, l, P);
                bool ok  = dendrograms_equal(ref, got);
                std::cout << "  N=" << std::setw(3) << N
                          << "  P=" << P
                          << "  linkage=" << (l == Linkage::SINGLE ? "single  " : "complete")
                          << "  " << (ok ? "✓" : "✗ MISMATCH") << "\n";
                assert(ok);
            }
        }
    }
    std::cout << "  All correctness checks passed.\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  2. Benchmark
// ─────────────────────────────────────────────────────────────────────────────

void run_benchmark() {
    std::cout << "=== 2. Benchmark ===\n";
    std::cout << std::fixed << std::setprecision(1);

    // Header
    std::cout << std::setw(6)  << "N"
              << std::setw(12) << "naive(ms)"
              << std::setw(8)  << "P=1"
              << std::setw(8)  << "P=2"
              << std::setw(8)  << "P=4"
              << std::setw(8)  << "P=8"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (int N : {500, 1000, 2000}) {
        auto pts = make_points(N);

        double t_naive = time_ms([&]{ naive_hac(pts, Linkage::SINGLE); });

        std::cout << std::setw(6) << N << std::setw(12) << t_naive;

        for (int P : {1, 2, 4, 8}) {
            double t_par = time_ms([&]{ ptrad_hac(pts, Linkage::SINGLE, P); });
            double speedup = t_naive / t_par;
            std::cout << std::setw(7) << speedup << "x";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  3. Real data — Spotify
// ─────────────────────────────────────────────────────────────────────────────

void run_spotify(int N_limit, int P) {
    std::cout << "=== 3. Real data — Spotify (first " << N_limit << " songs, P=" << P << ") ===\n";

    // Load and normalize
    std::vector<Song> songs;
    std::vector<std::vector<double>> math_matrix;
    if (!loadAndCleanData("spotify_tracks.csv", songs, math_matrix)) {
        std::cout << "  Could not load spotify_tracks.csv, skipping.\n\n";
        return;
    }
    normalizeMatrix(math_matrix);

    // Limit to N_limit songs for a reasonable runtime
    if ((int)songs.size() > N_limit) {
        songs.resize(N_limit);
        math_matrix.resize(N_limit);
    }

    // Convert to Points
    std::vector<Point> points;
    for (const auto& row : math_matrix)
        points.push_back({row});

    // Run both and time them
    std::vector<Merge> dg_naive, dg_par;
    double t_naive = time_ms([&]{ dg_naive = naive_hac(points, Linkage::SINGLE); });
    double t_par   = time_ms([&]{ dg_par   = ptrad_hac(points, Linkage::SINGLE, P); });

    std::cout << "  naive:         " << t_naive << " ms\n";
    std::cout << "  ptrad (P=" << P << "):  " << t_par   << " ms\n";
    std::cout << "  speedup:       " << t_naive / t_par << "x\n";

    // Correctness check on real data
    bool ok = dendrograms_equal(dg_naive, dg_par);
    std::cout << "  dendrograms match: " << (ok ? "✓" : "✗ MISMATCH") << "\n";
    assert(ok);

    // Print last 5 merges (biggest groupings at the top of the tree)
    std::cout << "  Last 5 merges:\n";
    int n = static_cast<int>(dg_naive.size());
    for (int s = n - 5; s < n; ++s) {
        std::cout << "    merge " << s
                  << ": clusters " << dg_naive[s].i
                  << " + "         << dg_naive[s].j
                  << "  dist="     << std::setprecision(4) << dg_naive[s].dist
                  << "\n";
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    check_correctness();
    run_benchmark();
    run_spotify(1000, 4);
    return 0;
}

//run g++ -std=c++17 -O2 -pthread -I include hc_alg.cpp main2.cpp datacleaning.cpp normalization.cpp -o main
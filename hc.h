#pragma once
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <limits>

// ─────────────────────────────────────────────
//  Basic types
// ─────────────────────────────────────────────

struct Point {
    double x, y;
};

// One row of the dendrogram: clusters i and j merged at distance d,
// the result is given id new_id.
struct Merge {
    int    i, j;
    double dist;
    int    new_id;
};

enum class Linkage { SINGLE, COMPLETE };

// ─────────────────────────────────────────────
//  Euclidean distance
// ─────────────────────────────────────────────

inline double euclidean(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx*dx + dy*dy);
}

// ─────────────────────────────────────────────
//  Lance-Williams update
//
//  Single-link:   D(new, k) = min(D(i,k), D(j,k))
//  Complete-link: D(new, k) = max(D(i,k), D(j,k))
// ─────────────────────────────────────────────

inline double lance_williams(double d_ik, double d_jk, Linkage linkage) {
    switch (linkage) {
        case Linkage::SINGLE:   return std::min(d_ik, d_jk);
        case Linkage::COMPLETE: return std::max(d_ik, d_jk);
    }
    throw std::invalid_argument("Unknown linkage");
}

// ─────────────────────────────────────────────
//  Algorithm declarations
// ─────────────────────────────────────────────

std::vector<Merge> naive_hac(const std::vector<Point>& points, Linkage linkage);

// std::vector<Merge> ptrad_hac(const std::vector<Point>& points, Linkage linkage, int num_threads);
// std::vector<Merge> seq_pop_hac(const std::vector<Point>& points, Linkage linkage, int c, double delta);
// std::vector<Merge> ppop_hac(const std::vector<Point>& points, Linkage linkage, int c, double delta, int num_threads);
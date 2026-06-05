#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>

struct Point {
    std::vector<double> features;
};

struct Merge {
    int    i, j;
    double dist;
    int    new_id;
};

enum class Linkage { SINGLE, COMPLETE };

inline double euclidean(const Point& a, const Point& b) {
    double sum = 0.0;
    for (size_t k = 0; k < a.features.size(); ++k) {
        double diff = a.features[k] - b.features[k];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline double lance_williams(double d_ik, double d_jk, Linkage linkage) {
    switch (linkage) {
        case Linkage::SINGLE:   return std::min(d_ik, d_jk);
        case Linkage::COMPLETE: return std::max(d_ik, d_jk);
    }
    throw std::invalid_argument("Unknown linkage");
}

std::vector<Merge> naive_hac(const std::vector<Point>& points, Linkage linkage);
std::vector<Merge> ptrad_hac(const std::vector<Point>& points, Linkage linkage, int num_threads);
std::vector<Merge> seq_pop_hac(const std::vector<Point>& points, Linkage linkage, int c, double delta);
std::vector<Merge> ppop_hac(const std::vector<Point>& points, Linkage linkage, int c, double delta, int P);
std::vector<Merge> nested_pop_hac(const std::vector<Point>& points, Linkage linkage, int c_max);
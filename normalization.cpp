#include "normalization.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Compute the mean of one column across all rows.
static double columnMean(const std::vector<std::vector<double>>& matrix, size_t col) {
    double sum = 0.0;
    for (const auto& row : matrix) {
        sum += row[col];
    }
    return sum / static_cast<double>(matrix.size());
}

// Compute the population standard deviation of one column.
static double columnStdDev(const std::vector<std::vector<double>>& matrix,
                            size_t col, double mean) {
    double sq_sum = 0.0;
    for (const auto& row : matrix) {
        double diff = row[col] - mean;
        sq_sum += diff * diff;
    }
    return std::sqrt(sq_sum / static_cast<double>(matrix.size()));
}

// Apply log1p (= log(1 + x)) in-place to every value in a column.
// Safe for x >= 0; handles the common case of near-zero skewed features.
static void applyLog1p(std::vector<std::vector<double>>& matrix, size_t col) {
    for (auto& row : matrix) {
        row[col] = std::log1p(row[col]);
    }
}

// Z-score standardise one column in-place: x → (x - μ) / σ
// Skips the column silently if σ ≈ 0 (constant feature) to avoid ÷0.
static void zScoreColumn(std::vector<std::vector<double>>& matrix, size_t col) {
    double mean = columnMean(matrix, col);
    double std  = columnStdDev(matrix, col, mean);

    if (std < 1e-9) {
        std::cerr << "[normalization] Warning: column " << col
                  << " has near-zero std (" << std
                  << "). Skipping z-score (all values are identical).\n";
        return;
    }

    for (auto& row : matrix) {
        row[col] = (row[col] - mean) / std;
    }
}

// Replace the raw 'key' column (index 13) with two new columns appended at
// the end of each row: sin(2π·key/12) and cos(2π·key/12).
//
// Why cyclical encoding? Key is a circular variable — key 0 (C) and key 11 (B)
// are musically one semitone apart, but Euclidean distance would treat them as
// the farthest pair. Sin/cos maps the 12 chromatic positions onto a unit circle
// so that distance reflects true musical proximity.
//
// The raw column is then erased so it does not appear twice.
// After this call the matrix has 17 columns instead of 16.
static void expandKeyCyclical(std::vector<std::vector<double>>& matrix) {
    const double TWO_PI = 2.0 * M_PI;
    for (auto& row : matrix) {
        double key = row[13]; // raw key value (0–11)
        row.push_back(std::sin(TWO_PI * key / 12.0)); // key_sin
        row.push_back(std::cos(TWO_PI * key / 12.0)); // key_cos
        row.erase(row.begin() + 13);                   // remove raw key
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<std::string> getFeatureNames() {
    // Reflects the layout AFTER normalization (17 features).
    return {
        "release_year",      // 0  – z-scored
        "popularity",        // 1  – z-scored
        "duration_ms",       // 2  – z-scored
        "explicit",          // 3  – binary, untouched
        "danceability",      // 4  – z-scored
        "energy",            // 5  – z-scored
        "loudness",          // 6  – z-scored
        "speechiness",       // 7  – log1p + z-scored
        "acousticness",      // 8  – log1p + z-scored
        "instrumentalness",  // 9  – log1p + z-scored
        "liveness",          // 10 – log1p + z-scored
        "valence",           // 11 – z-scored
        "tempo",             // 12 – z-scored
        "mode",              // 13 – binary, untouched (raw key erased from 13 first, mode shifts here)
        "time_signature",    // 14 – z-scored (shifts here after key erase)
        "key_sin",           // 15 – cyclical encoding of key (appended)
        "key_cos"            // 16 – cyclical encoding of key (appended)
    };
}

void normalizeMatrix(std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) {
        std::cerr << "[normalization] Warning: matrix is empty, nothing to do.\n";
        return;
    }

    const size_t expected_cols = 16;
    if (matrix[0].size() != expected_cols) {
        throw std::runtime_error(
            "[normalization] Expected " + std::to_string(expected_cols) +
            " columns, got " + std::to_string(matrix[0].size()) +
            ". Check loadAndCleanData.");
    }

    // ------------------------------------------------------------------
    // Step 1 — log1p transform on right-skewed audio features.
    //
    // speechiness (7), acousticness (8), instrumentalness (9), liveness (10)
    // all have most values clustered near 0 with a long right tail.
    // log1p compresses the tail so z-scoring gets a more symmetric input.
    // ------------------------------------------------------------------
    for (size_t col : {7u, 8u, 9u, 10u}) {
        applyLog1p(matrix, col);
    }

    // ------------------------------------------------------------------
    // Step 2 — z-score standardise all continuous / ordinal columns.
    //
    // Excluded intentionally:
    //   col 3  (explicit)        – binary flag; already on {0,1}
    //   col 13 (key)             – handled by cyclical expansion below
    //   col 14 (mode)            – binary flag; already on {0,1}
    // ------------------------------------------------------------------
    for (size_t col : {0u, 1u, 2u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 15u}) {
        zScoreColumn(matrix, col);
    }

    // ------------------------------------------------------------------
    // Step 3 — cyclical encoding for key (col 13).
    //
    // Removes the raw key column and appends key_sin + key_cos.
    // The matrix grows from 16 → 17 columns after this step.
    // ------------------------------------------------------------------
    expandKeyCyclical(matrix);

    std::cout << "[normalization] Done. Matrix is now "
              << matrix.size() << " rows × "
              << matrix[0].size() << " features.\n";
}
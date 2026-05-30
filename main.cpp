#include <iostream>
#include <vector>
#include "datacleaning.h"
#include "normalization.h"

int main() {
    // -------------------------------------------------------------------
    // Step 1 & 2 — Load and clean the data
    // -------------------------------------------------------------------
    std::vector<Song> original_dataset;
    std::vector<std::vector<double>> math_matrix;

    std::string filename = "spotify_tracks.csv";

    if (!loadAndCleanData(filename, original_dataset, math_matrix)) {
        return 1;
    }

    std::cout << "\nFirst song loaded: " << original_dataset[0].track_name
              << " by " << original_dataset[0].artist_name << "\n";
    std::cout << "Features per song (raw): " << math_matrix[0].size() << "\n";

    // -------------------------------------------------------------------
    // Step 3 — Normalize math_matrix
    // -------------------------------------------------------------------
    try {
        normalizeMatrix(math_matrix);
    } catch (const std::exception& e) {
        std::cerr << "Normalization failed: " << e.what() << "\n";
        return 1;
    }

    // Sanity-check: print feature names alongside the first song's values
    std::vector<std::string> feature_names = getFeatureNames();

    std::cout << "\n--- Normalised feature vector for: "
              << original_dataset[0].track_name << " ---\n";

    for (size_t i = 0; i < feature_names.size(); ++i) {
        std::cout << "  " << feature_names[i] << ": " << math_matrix[0][i] << "\n";
    }

    // -------------------------------------------------------------------
    // Step 4 — Clustering (coming next)
    // -------------------------------------------------------------------

    return 0;
}
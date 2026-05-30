#ifndef NORMALIZATION_H
#define NORMALIZATION_H
 
#include <vector>
#include <string>
 
// ---------------------------------------------------------------------------
// Math-matrix column layout BEFORE normalization (16 features):
//
//  0  release_year      – continuous, z-score
//  1  popularity        – continuous, z-score
//  2  duration_ms       – continuous, z-score
//  3  explicit          – binary 0/1, left as-is
//  4  danceability      – [0,1], z-score
//  5  energy            – [0,1], z-score
//  6  loudness          – [-28,0], z-score
//  7  speechiness       – right-skewed → log1p THEN z-score
//  8  acousticness      – right-skewed → log1p THEN z-score
//  9  instrumentalness  – right-skewed → log1p THEN z-score
// 10  liveness          – right-skewed → log1p THEN z-score
// 11  valence           – [0,1], z-score
// 12  tempo             – continuous, z-score
// 13  key               – circular 0-11 → replaced with key_sin + key_cos
// 14  mode              – binary 0/1, left as-is
// 15  time_signature    – ordinal, z-score
//
// Layout AFTER normalization (17 features — key expands to 2 columns):
//  0-12  same as above (transformed)
//  13    mode           (shifted from 14; raw key at 13 was erased first)
//  14    time_signature (shifted from 15; z-scored)
//  15    key_sin        (appended after erase)
//  16    key_cos        (appended after erase)
// ---------------------------------------------------------------------------
 
// Returns the feature name at each index after normalization.
// Useful for debugging and for labelling cluster output.
std::vector<std::string> getFeatureNames();
 
// Normalises math_matrix in-place:
//   1. log1p on skewed columns
//   2. z-score on all continuous columns
//   3. cyclical sin/cos expansion of the 'key' column
void normalizeMatrix(std::vector<std::vector<double>>& matrix);
 
#endif // NORMALIZATION_H
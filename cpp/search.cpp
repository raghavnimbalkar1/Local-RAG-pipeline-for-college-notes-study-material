#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>       // for read_index
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>      // for reading id_map.jsonl

using json = nlohmann::json;

int main() {
    // --- Load FAISS index ---
    faiss::Index* index = faiss::read_index("../index/notes.faiss");
    if (!index) {
        std::cerr << "Failed to load FAISS index!\n";
        return 1;
    }

    // --- Load ID map ---
    std::vector<std::string> id_map;
    std::ifstream id_file("../index/id_map.jsonl");
    if (!id_file.is_open()) {
        std::cerr << "Failed to open id_map.jsonl\n";
        return 1;
    }

    std::string line;
    while (std::getline(id_file, line)) {
        json j = json::parse(line);
        std::string info = j["source"].get<std::string>() + " (page " +
                           std::to_string(j["page"].get<int>()) + ")";
        id_map.push_back(info);
    }

    // --- Dummy query vector ---
    int dim = index->d;                  // automatically get embedding dimension
    std::vector<float> query(dim, 0.01f); // example query, replace later

    // --- Search top-k ---
    int k = 5;
    std::vector<faiss::idx_t> indices(k);
    std::vector<float> distances(k);

    index->search(1, query.data(), k, distances.data(), indices.data());

    // --- Print results ---
    for (int i = 0; i < k; i++) {
        std::cout << "[" << distances[i] << "] "
                  << (indices[i] < id_map.size() ? id_map[indices[i]] : "unknown")
                  << "\n";
    }

    // --- Clean up ---
    delete index;

    return 0;
}

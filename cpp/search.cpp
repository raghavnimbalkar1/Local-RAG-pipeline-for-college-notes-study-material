#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <stdexcept>

// --- Tiny .npy loader (float32 only) ---
std::vector<float> load_npy(const std::string& filename, int expected_dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + filename);
    }

    // Read file into buffer
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());

    // Parse header
    std::string header(buffer.begin() + 6, buffer.begin() + 256); // skip magic/version
    size_t start = header.find('{');
    size_t end   = header.find('}');
    if (start == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("Invalid .npy header in " + filename);
    }

    std::string dict = header.substr(start, end - start + 1);

    // Verify dtype
    if (dict.find("'descr': '<f4'") == std::string::npos &&
        dict.find("\"descr\": \"<f4\"") == std::string::npos) {
        throw std::runtime_error("Expected dtype float32 in " + filename);
    }

    // Data offset
    uint16_t header_len = *reinterpret_cast<uint16_t*>(&buffer[8]);
    size_t data_offset = 10 + header_len;

    // Extract floats
    size_t num_floats = (buffer.size() - data_offset) / sizeof(float);
    if (num_floats != (size_t)expected_dim) {
        throw std::runtime_error("Dimension mismatch: got " +
                                 std::to_string(num_floats) + ", expected " +
                                 std::to_string(expected_dim));
    }

    const float* data_ptr = reinterpret_cast<const float*>(&buffer[data_offset]);
    return std::vector<float>(data_ptr, data_ptr + num_floats);
}

using json = nlohmann::json;

int main() {
    try {
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
            delete index;
            return 1;
        }

        std::string line;
        while (std::getline(id_file, line)) {
            if (line.empty()) continue;
            json j = json::parse(line);
            std::string info = j["source"].get<std::string>() + " (page " +
                               std::to_string(j["page"].get<int>()) + ")";
            id_map.push_back(info);
        }

        // --- Load query vector ---
        int dim = index->d;  // embedding dimension
        std::vector<float> query = load_npy("../index/query.npy", dim);

        // --- Search top-k ---
        int k = 5;
        std::vector<faiss::idx_t> indices(k);
        std::vector<float> distances(k);

        index->search(1, query.data(), k, distances.data(), indices.data());

        // --- Print results ---
        std::cout << "\nTop " << k << " results:\n";
        for (int i = 0; i < k; i++) {
            if (indices[i] < 0) continue;
            std::cout << "[" << distances[i] << "] "
                      << (indices[i] < id_map.size() ? id_map[indices[i]] : "unknown")
                      << "\n";
        }

        delete index;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <numa.h>
#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <algorithm>

// NUMA-aware memory allocation
float* numa_alloc_float(size_t size, int node) {
    return (float*)numa_alloc_onnode(size * sizeof(float), node);
}

void numa_free_float(float* ptr, size_t size) {
    numa_free(ptr, size * sizeof(float));
}

// Simple k-means clustering
std::vector<int> kmeans_clustering(const float* data, unsigned num_points, unsigned dim, unsigned k, unsigned max_iter = 100) {
    std::vector<int> clusters(num_points);
    std::vector<float> centroids(k * dim);
    
    // Initialize centroids randomly
    for (unsigned i = 0; i < k; ++i) {
        unsigned random_point = rand() % num_points;
        std::copy(data + random_point * dim, data + (random_point + 1) * dim, centroids.begin() + i * dim);
    }

    for (unsigned iter = 0; iter < max_iter; ++iter) {
        // Assign points to nearest centroid
        #pragma omp parallel for
        for (unsigned i = 0; i < num_points; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int nearest_centroid = 0;
            for (unsigned j = 0; j < k; ++j) {
                float dist = l2_distance_avx(data + i * dim, centroids.data() + j * dim, dim);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_centroid = j;
                }
            }
            clusters[i] = nearest_centroid;
        }

        // Update centroids
        std::vector<float> new_centroids(k * dim, 0);
        std::vector<int> cluster_sizes(k, 0);
        for (unsigned i = 0; i < num_points; ++i) {
            int cluster = clusters[i];
            for (unsigned d = 0; d < dim; ++d) {
                new_centroids[cluster * dim + d] += data[i * dim + d];
            }
            cluster_sizes[cluster]++;
        }

        for (unsigned i = 0; i < k; ++i) {
            if (cluster_sizes[i] > 0) {
                for (unsigned d = 0; d < dim; ++d) {
                    centroids[i * dim + d] = new_centroids[i * dim + d] / cluster_sizes[i];
                }
            }
        }
    }

    return clusters;
}

void load_data_numa(const char* filename, float**& data, unsigned& num, unsigned& dim, int numa_nodes) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "open file error" << std::endl;
        exit(-1);
    }
    in.read(reinterpret_cast<char*>(&dim), 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = static_cast<size_t>(ss);
    num = static_cast<unsigned>(fsize / (dim + 1) / 4);

    // Allocate temporary buffer
    float* temp_data = new float[num * dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read(reinterpret_cast<char*>(temp_data + i * dim), dim * 4);
    }
    in.close();

    // Perform clustering
    std::vector<int> clusters = kmeans_clustering(temp_data, num, dim, numa_nodes);

    // Allocate NUMA-aware memory and distribute data
    data = new float*[numa_nodes];
    std::vector<unsigned> cluster_sizes(numa_nodes, 0);
    for (int i = 0; i < num; ++i) {
        cluster_sizes[clusters[i]]++;
    }

    for (int node = 0; node < numa_nodes; ++node) {
        data[node] = numa_alloc_float(cluster_sizes[node] * dim, node);
    }

    std::vector<unsigned> cluster_offsets(numa_nodes, 0);
    for (int i = 0; i < num; ++i) {
        int cluster = clusters[i];
        unsigned offset = cluster_offsets[cluster]++;
        std::copy(temp_data + i * dim, temp_data + (i + 1) * dim, data[cluster] + offset * dim);
    }

    delete[] temp_data;
}

void free_data_numa(float** data, unsigned num, unsigned dim, int numa_nodes) {
    for (int node = 0; node < numa_nodes; ++node) {
        numa_free_float(data[node], num * dim);
    }
    delete[] data;
}

// ... (keep the l2_distance_avx function from the previous answer)

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cout << argv[0] << " data_file query_file nsg_path search_L search_K result_path numa_nodes" << std::endl;
        exit(-1);
    }

    float **data_load = nullptr, **query_load = nullptr;
    unsigned points_num, dim, query_num, query_dim;
    int numa_nodes = atoi(argv[7]);

    load_data_numa(argv[1], data_load, points_num, dim, numa_nodes);
    load_data_numa(argv[2], query_load, query_num, query_dim, numa_nodes);
    assert(dim == query_dim);

    unsigned L = static_cast<unsigned>(atoi(argv[4]));
    unsigned K = static_cast<unsigned>(atoi(argv[5]));

    if (L < K) {
        std::cerr << "search_L cannot be smaller than search_K!" << std::endl;
        exit(-1);
    }

    efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
    index.Load(argv[3]);

    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);

    auto s = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<unsigned>> res(query_num);

    #pragma omp parallel for
    for (unsigned i = 0; i < query_num; i++) {
        std::vector<unsigned> tmp(K);
        // Perform search on all NUMA nodes and merge results
        std::vector<std::vector<unsigned>> node_results(numa_nodes);
        for (int node = 0; node < numa_nodes; ++node) {
            node_results[node].resize(K);
            index.Search(query_load[node] + (i % (query_num / numa_nodes)) * dim, data_load[node], K, paras, node_results[node].data());
        }
        // Merge results (you might want to implement a more sophisticated merging strategy)
        std::merge(node_results[0].begin(), node_results[0].end(),
                   node_results[1].begin(), node_results[1].end(),
                   tmp.begin());
        if (numa_nodes > 2) {
            for (int node = 2; node < numa_nodes; ++node) {
                std::vector<unsigned> merged_tmp(K * node);
                std::merge(tmp.begin(), tmp.end(),
                           node_results[node].begin(), node_results[node].end(),
                           merged_tmp.begin());
                tmp = std::vector<unsigned>(merged_tmp.begin(), merged_tmp.begin() + K);
            }
        }
        res[i] = std::move(tmp);
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "search time: " << diff.count() << "\n";

    save_result(argv[6], res);

    // Free NUMA-allocated memory
    free_data_numa(data_load, points_num, dim, numa_nodes);
    free_data_numa(query_load, query_num, query_dim, numa_nodes);

    return 0;
}
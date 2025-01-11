#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <numa.h>
#include <immintrin.h>
#include <omp.h>

// NUMA-aware memory allocation
float* numa_alloc_float(size_t size, int node) {
    return (float*)numa_alloc_onnode(size * sizeof(float), node);
}

void numa_free_float(float* ptr, size_t size) {
    numa_free(ptr, size * sizeof(float));
}

void load_data(const char* filename, float*& data, unsigned& num, unsigned& dim) {
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

    // Use NUMA-aware allocation
    int numa_node = 0; // Assume we're using NUMA node 0
    data = numa_alloc_float(num * dim, numa_node);

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read(reinterpret_cast<char*>(data + i * dim), dim * 4);
    }
    in.close();
}

void save_result(const char* filename, const std::vector<std::vector<unsigned>>& results) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    for (const auto& result : results) {
        unsigned GK = static_cast<unsigned>(result.size());
        out.write(reinterpret_cast<const char*>(&GK), sizeof(unsigned));
        out.write(reinterpret_cast<const char*>(result.data()), GK * sizeof(unsigned));
    }
    out.close();
}

// AVX-optimized L2 distance computation
float l2_distance_avx(const float* a, const float* b, unsigned dim) {
    __m256 sum = _mm256_setzero_ps();
    for (unsigned i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    __m128 sum_low = _mm256_extractf128_ps(sum, 0);
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    sum_low = _mm_add_ps(sum_low, sum_high);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    return _mm_cvtss_f32(sum_low);
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cout << argv[0] << " data_file query_file nsg_path search_L search_K result_path" << std::endl;
        exit(-1);
    }

    float* data_load = nullptr, * query_load = nullptr;
    unsigned points_num, dim, query_num, query_dim;

    load_data(argv[1], data_load, points_num, dim);
    load_data(argv[2], query_load, query_num, query_dim);
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
        index.Search(query_load + i * dim, data_load, K, paras, tmp.data());
        res[i] = std::move(tmp);
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "search time: " << diff.count() << "\n";

    save_result(argv[6], res);

    // Free NUMA-allocated memory
    numa_free_float(data_load, points_num * dim);
    numa_free_float(query_load, query_num * query_dim);

    return 0;
}
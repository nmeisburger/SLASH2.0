#pragma once

#include <algorithm>
#include <experimental/random>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <stdint.h>
#include <unordered_map>
#include <vector>

struct Item {
    uint32_t item;
    uint32_t cnt;
};

class LSH {
  private:
    uint32_t L;
    uint32_t reservoir_size;
    uint32_t range_pow;
    uint32_t range;

    uint32_t max_rand;
    uint32_t *gen_rand;

    uint32_t *all_reservoir_data;
    uint32_t **reservoirs;
    omp_lock_t *reservoir_locks;

  public:
    static constexpr uint32_t Empty = -1;

    LSH(uint32_t num_tables, uint32_t reservoir_size, uint32_t range_pow, uint32_t max_rand);

    void insert(uint32_t num_items, uint32_t **hashes, uint32_t *items);

    void insert(uint32_t *hashes, uint32_t item);

    void retrieve(uint32_t num_query, uint32_t **hashes, uint32_t *results_buffer);

    Item *topK(uint32_t num_query, uint32_t top_k, uint32_t **hashes);

    void reset();

    void view();

    void add_random_items(uint32_t num_items, bool verbose);

    ~LSH();
};
#include "LSH.h"

using namespace std;

LSH::LSH(uint32_t num_tables, uint32_t reservoir_size, uint32_t range_pow, uint32_t max_rand) {
    this->L = num_tables;
    this->reservoir_size = reservoir_size;
    this->range_pow = range_pow;
    this->range = 1 << range_pow;

    this->all_reservoir_data = new uint32_t[(reservoir_size + 1) * L * range];
    this->reservoirs = new uint32_t *[L * range];
    this->reservoir_locks = new omp_lock_t[L * range];
    for (size_t i = 0; i < L * range; i++) {
        reservoirs[i] = all_reservoir_data + i * (reservoir_size + 1);
        reservoirs[i][0] = 0;
        for (size_t j = 1; j <= reservoir_size; j++) {
            reservoirs[i][j] = LSH::Empty;
        }
        omp_init_lock(reservoir_locks + i);
    }

    this->max_rand = max_rand;
    this->gen_rand = new uint32_t[max_rand];
    for (unsigned i = 1; i < max_rand; i++) {
        // gen_rand[i] = ((uint32_t)rand()) % i;
        gen_rand[i] = std::experimental::randint((uint32_t)0, i);
    }
}

void LSH::insert(uint32_t num_items, uint32_t **hashes, uint32_t *items) {
    uint32_t *reservoir;
    uint32_t loc;
    uint32_t hash_index;
#pragma omp parallel for default(none)                                                             \
    shared(num_items, hashes, items) private(reservoir, loc, hash_index)
    for (size_t n = 0; n < num_items; n++) {
        for (size_t table = 0; table < L; table++) {
            hash_index = hashes[n][table];
            reservoir = reservoirs[table * range + hash_index];

            omp_set_lock(reservoir_locks + table * range + hash_index);
            loc = reservoir[0];
            reservoir[0]++;
            omp_unset_lock(reservoir_locks + table * range + hash_index);

            if (loc < reservoir_size) {
                reservoir[loc + 1] = items[n];
            } else {
                uint32_t new_loc =
                    gen_rand[std::min((max_rand - 1), loc)]; // Anshu: pre-generate for speed (have
                                                             // a pregenerated random factory)
                if (loc < reservoir_size) {
                    reservoir[new_loc + 1] = items[n];
                }
            }
        }
    }
}

void LSH::insert(uint32_t *hashes, uint32_t item) {
    uint32_t *reservoir;
    uint32_t loc;
    uint32_t hash_index;
    for (size_t table = 0; table < L; table++) {
        hash_index = hashes[table];
        reservoir = reservoirs[table * range + hash_index];

        omp_set_lock(reservoir_locks + table * range + hash_index);
        loc = reservoir[0];
        reservoir[0]++;
        omp_unset_lock(reservoir_locks + table * range + hash_index);

        if (loc < reservoir_size) {
            reservoir[loc + 1] = item;
        } else {
            loc = gen_rand[std::min((max_rand - 1), loc)];
            if (loc < reservoir_size) {
                reservoir[loc + 1] = item;
            }
        }
    }
}

void LSH::retrieve(uint32_t num_query, uint32_t **hashes, uint32_t *results_buffer) {

    size_t loc, index;
#pragma omp parallel for default(none) shared(num_query, hashes, results_buffer) private(loc, index)
    for (size_t query = 0; query < num_query; query++) {
        for (size_t table = 0; table < L; table++) {
            size_t loc = query * L + table;
            size_t index = table * range + hashes[query][table];
            std::copy(reservoirs[index] + 1, reservoirs[index] + 1 + reservoir_size,
                      results_buffer + loc * reservoir_size);
        }
    }
}

Item *LSH::topK(uint32_t num_query, uint32_t top_k, uint32_t **hashes) {

    Item *result = new Item[num_query * top_k];

#pragma omp parallel for default(none) shared(num_query, top_k, hashes, result)
    for (size_t q = 0; q < num_query; q++) {

        Item *thisResult = result + q * top_k;

        unordered_map<uint32_t, uint32_t> cnts;

        for (size_t table = 0; table < L; table++) {
            size_t loc = table * range + hashes[q][table];
            for (size_t r = 1; r < reservoirs[loc][0] + 1; r++) {
                uint32_t item = reservoirs[loc][r];
                if (cnts.find(item) == cnts.end()) {
                    cnts[item] = 1;
                } else {
                    cnts[item]++;
                }
            }
        }

        vector<Item> sortedCnts;

        for (const auto &[key, val] : cnts) {
            sortedCnts.push_back({key, val});
        }

        sort(sortedCnts.begin(), sortedCnts.end(),
             [](const auto &a, const auto &b) { return a.cnt >= b.cnt; });

        size_t k = 0;
        for (; k < sortedCnts.size() && k < top_k; k++) {
            thisResult[k] = sortedCnts[k];
        }

        for (; k < top_k; k++) {
            thisResult[k] = {LSH::Empty, 0};
        }
    }
    return result;
}

void LSH::reset() {
    for (size_t t = 0; t < L; t++) {
        for (size_t r = 0; r < range; r++) {
            uint32_t *reservoir = reservoirs[t * range + r];
            reservoir[0] = 0;
            for (size_t i = 0; i < reservoir_size; i++) {
                reservoir[i + 1] = LSH::Empty;
            }
        }
    }
}

void LSH::view() {
    for (size_t t = 0; t < L; t++) {
        printf("LSH Table %lu\n", t);
        for (size_t r = 0; r < range; r++) {
            uint32_t *reservoir = reservoirs[t * range + r];
            printf("Reservoir [%d/%d]", reservoir[0], reservoir_size);
            for (size_t i = 0; i < reservoir[0]; i++) {
                printf(" %u", reservoir[i + 1]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

LSH::~LSH() {
    delete[] all_reservoir_data;
    delete[] reservoirs;

    for (size_t i = 0; i < L * range; i++) {
        omp_destroy_lock(reservoir_locks + i);
    }

    delete[] reservoir_locks;
}

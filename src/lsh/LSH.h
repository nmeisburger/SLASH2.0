#pragma once

#include <assert.h>
#include <omp.h>
#include <stdint.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

class LSHTestFixture;

struct Item;

class LSH {
  friend class LSHTestFixture;

 private:
  uint64_t numTables_;
  uint64_t reservoirSize_;
  uint64_t rangePow_;
  uint64_t range_;

  uint64_t maxRand_;
  uint32_t* genRand_;

  uint32_t* allReservoirData_;
  uint32_t** reservoirs_;
  omp_lock_t* reservoirLocks_;

 public:
  static constexpr uint32_t Empty = -1;

  LSH(uint32_t numTables, uint32_t reservoirSize, uint32_t rangePow, uint32_t maxRand);

  void insertBatch(uint64_t numItems, uint32_t* ids, uint32_t* hashes);

  void insertRangedBatch(uint64_t numItems, uint32_t start, uint32_t* hashes);

  uint32_t** queryReservoirs(uint64_t numItems, uint32_t* hashes);

  Item* queryTopK(uint64_t numItems, uint32_t* hashes, uint64_t k);

  void reset();

  void view();

  void checkRanges(uint32_t start, uint32_t end) {
    for (size_t i = 0; i < range_ * numTables_; i++) {
      for (size_t j = 1; j <= std::min(reservoirs_[i][0], (uint32_t)reservoirSize_); j++) {
        if (!(reservoirs_[i][j] >= start && reservoirs_[i][j] < end)) {
          printf("[%lu, %lu]: %u\n", i, j, reservoirs_[i][j]);
          exit(1);
        }
      }
      for (size_t j = reservoirs_[i][0] + 1; j <= reservoirSize_; j++) {
        if (reservoirs_[i][j] != LSH::Empty) {
          printf("[%lu, %lu]: %u\n", i, j, reservoirs_[i][j]);
          exit(1);
        }
      }
    }
  }

  ~LSH();
};

struct Item {
  uint32_t item;
  uint32_t cnt;

  Item() {
    item = LSH::Empty;
    cnt = 0;
  }

  Item(uint32_t i, uint32_t c) {
    item = i;
    cnt = c;
  }
};
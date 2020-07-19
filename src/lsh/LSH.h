#pragma once

#include <omp.h>
#include <stdint.h>

#include <algorithm>
#include <experimental/random>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

class LSHTestFixture;

struct Item {
  uint32_t item;
  uint32_t cnt;
};

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

  ~LSH();
};
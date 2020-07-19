#pragma once

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <functional>

#include "../lsh/LSH.h"

using namespace std;

class Slash {
 private:
  int rank_, worldSize_;

  LSH *lsh_;
  uint32_t numTables_, reservoirSize_, rangePow_;

  function<uint32_t *(string &)> imageProcessor_;

 public:
  Slash(uint32_t numTables, uint32_t reservoirSize, uint32_t rangePow, int rank, int worldSize)
      : numTables_(numTables),
        reservoirSize_(reservoirSize),
        rangePow_(rangePow),
        rank_(rank),
        worldSize_(worldSize) {
    lsh_ = new LSH(numTables_, reservoirSize_, rangePow_, 100000);
  }

  void add(string filename, uint64_t numItems, uint64_t batchSize, uint64_t offset = 0);

  void add(vector<string> filenames, uint64_t numItemsPerFile, uint64_t batchSize);

  uint32_t *distributedTopK(string filename, uint64_t numQueries, uint64_t k, uint64_t offset = 0);

  ~Slash() { delete lsh_; }
};
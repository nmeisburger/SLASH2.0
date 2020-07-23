#pragma once

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <functional>

#include "../doph/DOPH.h"
#include "../lsh/LSH.h"
#include "../util/Reader.h"

using namespace std;

class Slash {
 private:
  int rank_, worldSize_;

  std::unique_ptr<LSH> lsh_;
  std::unique_ptr<DOPH> doph_;
  uint64_t numTables_, K_, reservoirSize_, rangePow_;

  inline pair<uint32_t *, uint32_t *> partition(uint64_t n, uint64_t offset = 0) {
    uint32_t *lens = new uint32_t[worldSize_];
    uint32_t *offsets = new uint32_t[worldSize_];

    uint32_t baselen = n / worldSize_;
    uint32_t r = baselen % worldSize_;
    for (int i = 0; i < worldSize_; i++) {
      lens[i] = baselen;
      if (i < r) {
        lens[i]++;
      }
    }

    offsets[0] = offset;
    for (int i = 1; i < worldSize_; i++) {
      offsets[i] = offsets[i - 1] + lens[i - 1];
    }
    return {lens, offsets};
  }

 public:
  Slash(uint64_t numTables, uint64_t k, uint64_t reservoirSize, uint64_t rangePow)
      : numTables_(numTables), reservoirSize_(reservoirSize), rangePow_(rangePow), K_(k) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
    lsh_ = std::make_unique<LSH>(numTables_, reservoirSize_, rangePow_, 100000);
    doph_ = std::make_unique<DOPH>(K_, numTables_, rangePow_, rank_, worldSize_);
  }

  // TODO: integrate automatic batching after initial testing.

  void store(string filename, uint64_t numItems, uint64_t batchSize, uint32_t avgDim,
             uint64_t offset = 0);

  void store(vector<string> &&filenames, uint64_t numItemsPerFile, uint32_t avgDim,
             uint64_t batchSize);

  uint32_t *distributedTopK(string filename, uint64_t numQueries, uint64_t k, uint64_t offset = 0);

  ~Slash() {}
};
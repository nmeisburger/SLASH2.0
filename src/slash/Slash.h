#pragma once

#include <assert.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <random>
#include "unordered_map"

#include "../doph/DOPH.h"
#include "../lsh/LSH.h"
#include "../util/Reader.h"
#include "util/io.cpp"
#include "../hash/srpHash.h"
#include "../config.h"

using namespace std;

struct comparePair
{
        bool operator()(pair<int, int> p1, pair<int, int> p2)
        {
                // if frequencies of two elements are same
                // then the larger number should come first
                if (p1.second == p2.second)
                        return p1.first > p2.first;

                // insert elements in the priority queue on the basis of
                // decreasing order of frequencies
                return p1.second > p2.second;
        }
};

class Slash {
 private:
  int rank_, worldSize_;

  std::unique_ptr<LSH> lsh_;
  std::unique_ptr<DOPH> doph_;
  uint64_t numTables_, K_, reservoirSize_, rangePow_;
  vector<float> _meanvec;
  vector<srpHash*> _storesrp;

  inline pair<uint32_t *, uint32_t *> partition(uint64_t n, uint64_t offset = 0) {
    uint32_t *lens = new uint32_t[worldSize_];
    uint32_t *offsets = new uint32_t[worldSize_];

    uint32_t baselen = n / worldSize_;
    uint32_t r = n % worldSize_;
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

  inline pair<uint32_t *, uint32_t *> partition_query(uint64_t n, uint64_t num_feature, uint64_t offset = 0) {
    uint32_t *lens = new uint32_t[worldSize_];
    uint32_t *offsets = new uint32_t[worldSize_];

    uint32_t num = n / num_feature;
    uint32_t baselen = num / worldSize_;
    uint32_t r = num % worldSize_;
    for (int i = 0; i < worldSize_; i++) {
      lens[i] = baselen * num_feature;
      if (i < r) {
        lens[i] += num_feature;
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
    doph_ = std::make_unique<DOPH>(K_, numTables_, rangePow_, worldSize_, rank_);

    srand(time(0));
    int *seeds = new int(numTables_);
    //TODO: Broadcast the seeds from root Node;
//    if (rank_ == 0) {
//            for (int m = 0; m < numTables_; m++) {
//                    seeds[m] = rand();
//            }
//    }
//    MPI_Bcast(seeds, numTables_, MPI_INT, 0, MPI_COMM_WORLD);
//
//    cout << "Node: " << rank_ << " have seeds: ";
//    for (int j = 0; j < numTables_; j++) {
//            cout << seeds[j] << " ";
//    }
//    cout << endl;

    for (int n = 0; n < numTables_; n++) {
            srpHash *srp = new srpHash(128, k, 1, rand());
            _storesrp.push_back(srp);
    }
    cout << "Node: " << rank_ << " have srp #: " << _storesrp.size() << endl;
    lsh_->checkRanges(0, 1000);
  }

  // TODO: integrate automatic batching after initial testing.

  void store(const string filename, uint64_t numItems, uint64_t batchSize, uint32_t avgDim,
             uint64_t offset = 0);
  
  void storevec(const string filename, uint64_t numItems, size_t sample = 1);

  vector<uint32_t> query(string filename, uint64_t numItems);

  void multiStore(vector<string> &&filenames, uint64_t numItemsPerFile, uint32_t avgDim,
                  uint64_t batchSize);

  uint32_t *topK(const string filename, uint32_t avgDim, uint64_t numQueries, uint64_t k,
                 uint64_t offset = 0);

  uint32_t *distributedTopK(string filename, uint64_t numQueries, uint64_t k, uint64_t offset = 0);

  ~Slash() {}
};
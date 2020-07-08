#pragma once

#include <algorithm>
#include <functional>
#include <mpi.h>
#include <omp.h>

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
        : numTables_(numTables), reservoirSize_(reservoirSize), rangePow_(rangePow), rank_(rank),
          worldSize_(worldSize) {
        lsh_ = new LSH(numTables_, reservoirSize_, rangePow_, 100000);
    }

    void store(vector<pair<string, uint32_t>> &imagesAndIds);

    Item *query(vector<string> &queries, uint32_t topK);

    ~Slash() { delete lsh_; }
};
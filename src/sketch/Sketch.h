#pragma once

#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <vector>

constexpr uint64_t SKETCH_NULL = -1;

struct TopKResult {
    uint64_t *items;
    uint64_t count;

    TopKResult() : count(0) {}
};

struct Item {
    uint64_t heavyHitter;
    int32_t count;

    Item() : heavyHitter(SKETCH_NULL), count(0) {}
};

class Sketch {
  private:
    uint64_t _numHashes, _rowSize, _numSketches, _sketchSize, _totalSize;

    uint32_t *_seeds;

    int _rank, _worldSize;

    Item *_sketch;

    void merge(Item *other);

    inline void hash(uint64_t item, uint32_t *hashes);

    inline uint64_t index(uint64_t sketch, uint64_t row, uint64_t item) {
        return sketch * _numHashes * _rowSize + row * _rowSize + item;
    }

  public:
    Sketch(uint64_t numHashes, uint64_t rowSize, uint64_t numSketches, int rank, int worldSize);

    void mergeAll();

    void insert(uint64_t *items, uint64_t n);

    TopKResult *topK(uint32_t k, uint32_t threshold = 0);
};
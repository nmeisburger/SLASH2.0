#pragma once

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "lsh/LSH.h"

class MockSketch;
class SketchTestFixture;

struct TopKResult {
  uint32_t *items;
  uint32_t count;

  TopKResult() : count(0) {}
};

struct SketchItem {
  uint32_t heavyHitter;
  uint32_t count;

  SketchItem() : heavyHitter(LSH::Empty), count(0) {}
  SketchItem(uint32_t h, uint32_t c) : heavyHitter(h), count(c) {}
};

class Sketch {
  friend class MockSketch;
  friend class SketchTestFixture;

 private:
  uint64_t _numHashes, _rowSize, _numSketches, _sketchSize, _totalSize;

  uint32_t *_seeds;

  int _rank, _worldSize;

  SketchItem *_sketch;

  void merge(SketchItem *other);

  inline virtual void hash(uint32_t item, uint32_t *hashes);

  inline uint64_t index(uint64_t sketch, uint64_t row, uint64_t item) {
    return sketch * _sketchSize + row * _rowSize + item;
  }

 public:
  Sketch(uint64_t numHashes, uint64_t rowSize, uint64_t numSketches, int rank, int worldSize);

  void mergeAll();

  void insert(uint32_t *items, uint64_t n);

  TopKResult *topK(uint64_t k, uint32_t threshold = 0);

  void view();
};
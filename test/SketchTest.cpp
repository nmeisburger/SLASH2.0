#include <mpi.h>

#include "gtest/gtest.h"
#include "sketch/Sketch.h"

class MockSketch : public Sketch {
 public:
  MockSketch(uint64_t numHashes, uint64_t rowSize, uint64_t numSketches, int rank, int worldSize)
      : Sketch(numHashes, rowSize, numSketches, rank, worldSize) {}

  inline void hash(uint32_t item, uint32_t* hashes) override {
    for (size_t i = 0; i < _numHashes; i++) {
      hashes[i] = (item + i) % _rowSize;
    }
  }
};

class SketchTestFixture : public ::testing::Test {
 public:
  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
    sketch_ = std::make_shared<MockSketch>(4, 4, 3, rank_, worldSize_);
    smallSketch_ = std::make_shared<MockSketch>(2, 4, 3, rank_, worldSize_);
  }

  void callMerge(SketchItem* items) { smallSketch_->merge(items); }

  void setSketch(SketchItem* items) { smallSketch_->_sketch = items; }

  SketchItem* getSketchContents() { return sketch_->_sketch; }

  SketchItem* getSmallSketchContents() { return smallSketch_->_sketch; }

  std::shared_ptr<Sketch> sketch_;

  std::shared_ptr<Sketch> smallSketch_;
  int rank_, worldSize_;
};

TEST_F(SketchTestFixture, InsertionTest) {
  uint32_t items[6][8] = {{0, 0, 3, 4}, {7, 7, 5, 2}, {0, 1, 2, 3},
                          {0, 1, 2, 3}, {0, 0, 3, 4}, {3, 7, 7, 7}};

  SketchItem correct[] = {
      {0, 1}, {5, 1},          {2, 1},          {7, 2},          {7, 2},          {0, 1},
      {5, 1}, {2, 1},          {2, 1},          {7, 2},          {0, 1},          {5, 1},
      {5, 1}, {2, 1},          {7, 2},          {0, 1},          {0, 2},          {1, 2},
      {2, 2}, {3, 2},          {3, 2},          {0, 2},          {1, 2},          {2, 2},
      {2, 2}, {3, 2},          {0, 2},          {1, 2},          {1, 2},          {2, 2},
      {3, 2}, {0, 2},          {0, 1},          {LSH::Empty, 0}, {LSH::Empty, 0}, {7, 2},
      {7, 2}, {0, 1},          {LSH::Empty, 0}, {LSH::Empty, 0}, {LSH::Empty, 0}, {7, 2},
      {0, 1}, {LSH::Empty, 0}, {LSH::Empty, 0}, {LSH::Empty, 0}, {7, 2},          {0, 1}};

  uint32_t** ptr = new uint32_t*[8];
  for (int i = 0; i < 8; i++) {
    ptr[i] = items[i];
  }

  sketch_->insert(ptr, 2, 4);

  for (int i = 0; i < 48; i++) {
    EXPECT_EQ(getSketchContents()[i].heavyHitter, correct[i].heavyHitter);
    EXPECT_EQ(getSketchContents()[i].count, correct[i].count);
  }
}

TEST_F(SketchTestFixture, BasicMerge) {
  SketchItem items1[] = {
      {4, 2},          {1, 5},          {7, 1},          {LSH::Empty, 0},
      {4, 5},          {8, 2},          {LSH::Empty, 0}, {1, 2},

      {LSH::Empty, 0}, {3, 4},          {4, 2},          {6, 3},
      {LSH::Empty, 0}, {LSH::Empty, 0}, {7, 1},          {2, 1},

      {4, 2},          {LSH::Empty, 0}, {11, 1},         {7, 5},
      {4, 5},          {8, 2},          {9, 3},          {1, 2},
  };

  SketchItem items2[] = {{4, 3},          {1, 5},          {8, 1},          {2, 2},
                         {LSH::Empty, 0}, {9, 3},          {LSH::Empty, 0}, {1, 1},

                         {1, 3},          {2, 3},          {4, 1},          {LSH::Empty, 0},
                         {1, 1},          {LSH::Empty, 0}, {6, 4},          {3, 2},

                         {5, 2},          {LSH::Empty, 0}, {7, 3},          {2, 2},
                         {LSH::Empty, 0}, {9, 1},          {8, 1},          {3, 3}};

  SketchItem correct[] = {
      {4, 5}, {1, 10},         {8, 1}, {2, 2}, {4, 5}, {9, 1},          {LSH::Empty, 0}, {1, 3},

      {1, 3}, {3, 1},          {4, 3}, {6, 3}, {1, 1}, {LSH::Empty, 0}, {6, 3},          {3, 1},

      {5, 1}, {LSH::Empty, 0}, {7, 2}, {7, 3}, {4, 5}, {8, 1},          {9, 2},          {3, 1}};

  setSketch(items1);
  callMerge(items2);

  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(getSmallSketchContents()[i].heavyHitter, correct[i].heavyHitter);
    EXPECT_EQ(getSmallSketchContents()[i].count, correct[i].count);
  }
}
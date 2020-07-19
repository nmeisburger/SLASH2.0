#include "gtest/gtest.h"
#include "lsh/LSH.h"

using namespace ::testing;

class LSHTestFixture : public ::testing::Test {
 public:
  void SetUp() override {
    lsh_ = std::make_shared<LSH>(numTables_, reservoirSize_, rangePow_, 10);
    lsh_->genRand_[4] = 2;
  }

  uint32_t* getAllRows() { return lsh_->allReservoirData_; }

  std::shared_ptr<LSH> lsh_;

  const uint64_t numTables_ = 4;
  const uint64_t reservoirSize_ = 4;
  const uint64_t rangePow_ = 2;
};

TEST_F(LSHTestFixture, BatchedInsertion) {
  uint32_t ids[] = {0, 3, 8, 4};

  uint32_t hashes[] = {0, 1, 0, 3, 0, 2, 3, 3, 2, 1, 1, 3, 2, 0, 1, 3};

  lsh_->insertBatch(4, ids, hashes);

  uint32_t expectedRows[] = {
      2,          0,          3,          LSH::Empty, LSH::Empty, 0,          LSH::Empty,
      LSH::Empty, LSH::Empty, LSH::Empty, 2,          8,          4,          LSH::Empty,
      LSH::Empty, 0,          LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty, 1,
      4,          LSH::Empty, LSH::Empty, LSH::Empty, 2,          0,          8,
      LSH::Empty, LSH::Empty, 1,          3,          LSH::Empty, LSH::Empty, LSH::Empty,
      0,          LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty, 1,          0,
      LSH::Empty, LSH::Empty, LSH::Empty, 2,          8,          4,          LSH::Empty,
      LSH::Empty, 0,          LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty, 1,
      3,          LSH::Empty, LSH::Empty, LSH::Empty, 0,          LSH::Empty, LSH::Empty,
      LSH::Empty, LSH::Empty, 0,          LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty,
      0,          LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty, 4,          0,
      3,          8,          4};

  for (int i = 0; i < 80; i++) {
    EXPECT_EQ(getAllRows()[i], expectedRows[i]);
  }
}

TEST_F(LSHTestFixture, HandlesReservoirOverflow) {
  uint32_t ids[] = {1, 2, 3, 4, 5};

  uint32_t hashes[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

  lsh_->insertBatch(5, ids, hashes);

  uint32_t expectedRows[] = {
      5,          1,          2,          5,          4,          0,          LSH::Empty,
      LSH::Empty, LSH::Empty, LSH::Empty, 0,          LSH::Empty, LSH::Empty, LSH::Empty,
      LSH::Empty, 0,          LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty, 0,
      LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty, 5,          1,          2,
      5,          4,          0,          LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty,
      0,          LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty,

      0,          LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty, 0,          LSH::Empty,
      LSH::Empty, LSH::Empty, LSH::Empty, 5,          1,          2,          5,
      4,          0,          LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty,

      0,          LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty, 0,          LSH::Empty,
      LSH::Empty, LSH::Empty, LSH::Empty, 0,          LSH::Empty, LSH::Empty, LSH::Empty,
      LSH::Empty, 5,          1,          2,          5,          4};

  for (int i = 0; i < 80; i++) {
    EXPECT_EQ(getAllRows()[i], expectedRows[i]);
  }
}

TEST_F(LSHTestFixture, QueryReservoirs) {
  uint32_t ids[] = {0, 3, 8, 4};

  uint32_t hashes[] = {0, 1, 0, 3, 0, 2, 3, 3, 2, 1, 1, 3, 2, 0, 1, 3};

  lsh_->insertBatch(4, ids, hashes);

  uint32_t queryHashes[] = {0, 1, 2, 2, 2, 0, 1, 3, 1, 2, 3, 3};

  uint32_t rows[12][4] = {{0, 3, LSH::Empty, LSH::Empty},
                          {0, 8, LSH::Empty, LSH::Empty},
                          {LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty},
                          {LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty},
                          {8, 4, LSH::Empty, LSH::Empty},
                          {4, LSH::Empty, LSH::Empty, LSH::Empty},
                          {8, 4, LSH::Empty, LSH::Empty},
                          {0, 3, 8, 4},
                          {LSH::Empty, LSH::Empty, LSH::Empty, LSH::Empty},
                          {3, LSH::Empty, LSH::Empty, LSH::Empty},
                          {3, LSH::Empty, LSH::Empty, LSH::Empty},
                          {0, 3, 8, 4}};

  uint32_t** results = lsh_->queryReservoirs(3, queryHashes);

  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(results[i][j], rows[i][j]);
    }
  }
}

TEST_F(LSHTestFixture, QueryTopk) {
  uint32_t ids[] = {0, 3, 8, 4};

  uint32_t hashes[] = {0, 1, 0, 3, 0, 2, 3, 3, 2, 1, 1, 3, 2, 0, 1, 3};

  lsh_->insertBatch(4, ids, hashes);

  uint32_t queryHashes[] = {0, 1, 2, 2, 2, 0, 1, 3, 1, 2, 3, 3};

  Item topk[] = {{0, 2}, {3, 1}, {8, 1}, {LSH::Empty, 0}, {4, 4}, {8, 3},
                 {0, 1}, {3, 1}, {3, 3}, {0, 1},          {8, 1}, {4, 1}};

  Item* results = lsh_->queryTopK(3, queryHashes, 4);

  for (int i = 0; i < 12; i++) {
    EXPECT_EQ(results[i].item, topk[i].item);
    EXPECT_EQ(results[i].cnt, topk[i].cnt);
  }
}
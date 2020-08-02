#include "LSH.h"

using namespace std;

LSH::LSH(uint32_t numTables, uint32_t reservoirSize, uint32_t rangePow, uint32_t maxRand)
    : numTables_(numTables), reservoirSize_(reservoirSize), rangePow_(rangePow), maxRand_(maxRand) {
  this->range_ = 1 << rangePow_;

  this->allReservoirData_ = new uint32_t[(reservoirSize_ + 1) * numTables_ * range_];
  this->reservoirs_ = new uint32_t *[numTables_ * range_];
  this->reservoirLocks_ = new omp_lock_t[numTables_ * range_];
  for (size_t i = 0; i < numTables_ * range_; i++) {
    reservoirs_[i] = allReservoirData_ + i * (reservoirSize_ + 1);
    reservoirs_[i][0] = 0;
    for (size_t j = 1; j < reservoirSize_ + 1; j++) {
      reservoirs_[i][j] = LSH::Empty;
    }
    omp_init_lock(reservoirLocks_ + i);
  }

  srand(189283);

  this->genRand_ = new uint32_t[maxRand_];
  for (size_t i = 1; i < maxRand_; i++) {
    genRand_[i] = ((uint32_t)rand()) % i;
  }
}

void LSH::insertBatch(uint64_t numItems, uint32_t *ids, uint32_t *hashes) {
  uint32_t *reservoir;
  uint64_t loc;
  uint32_t hashIndex;
#pragma omp parallel for default(none) \
    shared(numItems, ids, hashes) private(reservoir, loc, hashIndex)
  for (size_t n = 0; n < numItems; n++) {
    for (size_t table = 0; table < numTables_; table++) {
      hashIndex = hashes[n * numTables_ + table];
      reservoir = reservoirs_[table * range_ + hashIndex];

      omp_set_lock(reservoirLocks_ + table * range_ + hashIndex);
      loc = reservoir[0];
      reservoir[0]++;
      omp_unset_lock(reservoirLocks_ + table * range_ + hashIndex);

      if (loc < reservoirSize_) {
        reservoir[loc + 1] = ids[n];
      } else {
        // TODO: genRand_[loc % maxRand_] maybe?
        uint32_t newLoc = genRand_[std::min((maxRand_ - 1), loc)];
        if (newLoc < reservoirSize_) {
          reservoir[newLoc + 1] = ids[n];
        }
      }
    }
  }
}

void LSH::insertRangedBatch(uint64_t numItems, uint32_t start, uint32_t *hashes) {
  uint32_t *reservoir;
  uint64_t loc;
  uint32_t hashIndex;
  // #pragma omp parallel for default(none) \
//     shared(numItems, start, hashes) private(reservoir, loc, hashIndex)
  for (size_t n = 0; n < numItems; n++) {
    for (size_t table = 0; table < numTables_; table++) {
      hashIndex = hashes[n * numTables_ + table];
      reservoir = reservoirs_[table * range_ + hashIndex];

      omp_set_lock(reservoirLocks_ + table * range_ + hashIndex);
      loc = reservoir[0];
      reservoir[0]++;
      omp_unset_lock(reservoirLocks_ + table * range_ + hashIndex);

      if (loc < reservoirSize_) {
        reservoir[loc + 1] = start + n;
      } else {
        // TODO: genRand_[loc % maxRand_] maybe?
        uint32_t newLoc = genRand_[std::min((maxRand_ - 1), loc)];
        if (newLoc < reservoirSize_) {
          reservoir[newLoc + 1] = start + n;
        }
      }
    }
  }
  checkRanges(0, 1000);
}

uint32_t **LSH::queryReservoirs(uint64_t numItems, uint32_t *hashes) {
  uint32_t **rows = new uint32_t *[numItems * numTables_];
  size_t loc, index;
#pragma omp parallel for default(none) shared(numItems, hashes, rows) private(loc, index)
  for (size_t query = 0; query < numItems; query++) {
    for (size_t table = 0; table < numTables_; table++) {
      index = query * numTables_ + table;
      loc = table * range_ + hashes[index];
      rows[index] = reservoirs_[loc] + 1;
    }
  }
  return rows;
}

Item *LSH::queryTopK(uint64_t numItems, uint32_t *hashes, uint64_t k) {
  Item *result = new Item[numItems * k];

  // #pragma omp parallel for default(none) shared(numItems, k, hashes, result)
  for (size_t q = 0; q < numItems; q++) {
    Item *thisResult = result + q * k;

    unordered_map<uint32_t, uint32_t> cnts;

    for (size_t table = 0; table < numTables_; table++) {
      size_t loc = table * range_ + hashes[q * numTables_ + table];
      for (size_t r = 1; r < std::min(reservoirs_[loc][0] + 1, (uint32_t)reservoirSize_ + 1); r++) {
        uint32_t item = reservoirs_[loc][r];
        assert(item >= 0 && item < 1000 && "Found bad item");
        if (cnts.find(item) == cnts.end()) {
          cnts[item] = 1;
        } else {
          cnts[item]++;
        }
      }
    }

    vector<Item> sortedCnts;

    assert(sortedCnts.size() == 0);
    for (const auto &entry : cnts) {
      assert(entry.first >= 0 && entry.first < 1000);
      sortedCnts.push_back(Item{entry.first, entry.second});
    }

    for (auto &i : sortedCnts) {
      assert(i.item >= 0 && i.item < 1000);
    }

    sort(sortedCnts.begin(), sortedCnts.end(), [](Item a, Item b) {
      assert(a.item >= 0 && a.item < 1000);
      assert(b.item >= 0 && b.item < 1000);

      return a.cnt >= b.cnt;
    });

    size_t i = 0;
    for (; i < sortedCnts.size(); i++) {
      assert(sortedCnts.at(i).item >= 0 && sortedCnts.at(i).item < 1000);
      thisResult[i] = {sortedCnts.at(i).item, sortedCnts.at(i).cnt};
      if (i == k) {
        break;
      }
    }

    for (; i < k; i++) {
      thisResult[i] = {LSH::Empty, 0};
    }
  }

  printf("this is done\n");
  return result;
}

void LSH::reset() {
  for (size_t t = 0; t < numTables_; t++) {
    for (size_t r = 0; r < range_; r++) {
      uint32_t *reservoir = reservoirs_[t * range_ + r];
      reservoir[0] = 0;
      for (size_t i = 0; i < reservoirSize_; i++) {
        reservoir[i + 1] = LSH::Empty;
      }
    }
  }
}

void LSH::view() {
  for (size_t t = 0; t < numTables_; t++) {
    printf("LSH Table %lu\n", t);
    for (size_t r = 0; r < range_; r++) {
      uint32_t *reservoir = reservoirs_[t * range_ + r];
      printf("Reservoir [%d/%lu]", reservoir[0], reservoirSize_);
      for (size_t i = 0; i < std::min(reservoir[0], (uint32_t)reservoirSize_); i++) {
        printf(" %u", reservoir[i + 1]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

LSH::~LSH() {
  delete[] allReservoirData_;
  delete[] reservoirs_;

  for (size_t i = 0; i < numTables_ * range_; i++) {
    omp_destroy_lock(reservoirLocks_ + i);
  }

  delete[] reservoirLocks_;
  delete[] genRand_;
}

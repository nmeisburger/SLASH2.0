#include "Slash.h"

void Slash::store(const string filename, uint64_t numItems, uint64_t batchSize, uint32_t avgDim,
                  uint64_t offset) {
  auto start = chrono::system_clock::now();

  auto p = partition(numItems);

  uint64_t myLen = p.first[rank_];
  uint64_t myOffset = p.second[rank_];

  std::unique_ptr<Reader> reader = std::make_unique<Reader>(filename, avgDim, myOffset);

  auto data = reader->readSparse(myLen);

  uint32_t *hashIndices = new uint32_t[numItems * numTables_];

  doph_->getHashes(hashIndices, data.indices, data.markers, myLen);

  lsh_->insertRangedBatch(myLen, myOffset, hashIndices);

  delete[] hashIndices;

  data.clear();

  auto end = chrono::system_clock::now();
  chrono::duration<double> elapsed = end - start;

  cout << "Slash::store Complete: " << elapsed.count() << " seconds, " << numItems << " items"
       << endl;
}

void Slash::multiStore(vector<string> &&filenames, uint64_t numItemsPerFile, uint32_t avgDim,
                       uint64_t batchSize) {
  auto start = chrono::system_clock::now();

  auto filePartition = partition(filenames.size());

  uint64_t numFiles = filePartition.first[rank_];
  uint64_t fileOffset = filePartition.second[rank_];

  for (size_t fileIdx = 0; fileIdx < numFiles; fileIdx++) {
    std::unique_ptr<Reader> reader =
        std::make_unique<Reader>(filenames.at(fileOffset + fileIdx), avgDim);

    auto data = reader->readSparse(numItemsPerFile);

    uint32_t *hashIndices = new uint32_t[numItemsPerFile * numTables_];

    doph_->getHashes(hashIndices, data.indices, data.markers, numItemsPerFile);

    lsh_->insertRangedBatch(numItemsPerFile, fileOffset * numItemsPerFile, hashIndices);

    data.clear();

    delete[] hashIndices;
  }

  auto end = chrono::system_clock::now();
  chrono::duration<double> elapsed = end - start;

  cout << "Slash::multiStore Complete: " << elapsed.count() << " seconds" << endl;
}

uint32_t *Slash::topK(const string filename, uint32_t avgDim, uint64_t numQueries, uint64_t k,
                      uint64_t offset) {
  auto start = chrono::system_clock::now();

  // lsh_->checkRanges(0, 1000);
  std::unique_ptr<Reader> reader = std::make_unique<Reader>(filename, avgDim, offset);

  auto data = reader->readSparse(numQueries);

  uint32_t *hashIndices = new uint32_t[numQueries * numTables_];

  doph_->getHashes(hashIndices, data.indices, data.markers, data.n);

  Item *result = lsh_->queryTopK(data.n, hashIndices, k);

  data.clear();

  delete[] hashIndices;

  uint32_t *topKResult = new uint32_t[numQueries * k];

  for (size_t i = 0; i < numQueries * k; i++) {
    if (result[i].item == LSH::Empty || (result[i].item >= 0 && result[i].item < 1000)) {
    } else {
      cout << result[i].item << " " << result[i].cnt << endl;
      exit(1);
    }
    topKResult[i] = result[i].item;
  }

  auto end = chrono::system_clock::now();
  chrono::duration<double> elapsed = end - start;

  cout << "Slash::topK Complete: " << elapsed.count() << " seconds, " << numQueries << " queries"
       << endl;
  return topKResult;
}

TopKResult *Slash::distributedTopK(string filename, uint32_t avgDim, uint64_t numQueries,
                                   uint64_t k, uint64_t sketchHashes, uint64_t sketchRowSize,
                                   uint64_t offset) {
  Sketch *s = new Sketch(sketchHashes, sketchRowSize, numQueries, rank_, worldSize_);

  auto p = partition(numQueries, 0);

  std::unique_ptr<Reader> reader =
      std::make_unique<Reader>(filename, avgDim, p.second[rank_] + offset);

  auto data = reader->readSparse(p.first[rank_]);
  assert(data.n == p.first[rank_]);

  uint32_t *hashes = new uint32_t[numQueries * numTables_];

  doph_->getHashes(hashes + p.second[rank_] * numTables_, data.indices, data.markers, data.n);

  int lens[worldSize_];
  int offsets[worldSize_];
  for (int i = 0; i < worldSize_; i++) {
    lens[i] = p.first[i] * numTables_;
    offsets[i] = p.second[i] * numTables_;
  }

  MPI_Allgatherv(MPI_IN_PLACE, p.first[rank_], MPI_UNSIGNED, hashes, lens, offsets, MPI_UNSIGNED,
                 MPI_COMM_WORLD);

  auto reservoirs = lsh_->queryReservoirs(numQueries, hashes);

  s->insert(reservoirs, numTables_, numQueries);

  s->mergeAll();

  auto result = s->topK(k);

  delete s;

  return result;
}

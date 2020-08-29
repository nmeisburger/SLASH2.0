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

void Slash::storevec(string filename, size_t sample) {
  // TODO: Spare some of the vectors for query. 
  // Read vectors
  vector<vector<float>> mat = readvec(filename);
  uint64_t size = mat.size();
  uint32_t dim = mat.at(0).size();
  cout << "size: " << size << "dimension: " << dim << endl;
  // Sample to get mean. Coco_vector: 37M vectors
  float *sumvec = new float[128];
  
  for (int n = 0; n < (dim - 1); n++) { // The vectors have image IDs attached at the end
    sumvec[n] = 0.0;
  }
  srand(time(0));
  for (int i = 0; i < size/sample; i++) {
    int ind = rand() % size;
    vector<float> temp = mat.at(ind);
    for (int n = 0; n < (dim - 1); n++) {
      sumvec[n] += temp.at(n);
    }
  }
  for (int j = 0; j < (dim - 1); j++) {
    _meanvec.push_back(sumvec[j]/(size/sample));
  }
  cout << "mean calculated" << endl;

  // SRP hash & store
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

  lsh_->checkRanges(0, 1000);
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

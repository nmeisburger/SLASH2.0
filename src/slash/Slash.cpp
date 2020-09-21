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
  // Read vectors
  vector<vector<float>> mat = readvec(filename, 129);
  uint64_t size = mat.size();
  uint32_t dim = mat.at(0).size();
  cout << "size: " << size << "  dimension " << dim << endl;
  cout << "test vector: ";
  for (auto i : mat.at(0)) {
    cout << i << " ";
  }
  // Sample to get mean. Coco_vector: 27M vectors
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
  cout << endl << "mean calculated. Begin hashing" << endl;

  // SRP hash & store
  uint32_t *ids = new uint32_t[size];
  uint32_t *hashlst = new uint32_t[numTables_];
  uint32_t *hashes = new uint32_t[numTables_ * size];

  dim = dim - 1;
  
  for (int x = 0; x < mat.size(); x++) {

      vector<float> single = mat.at(x);
      unsigned int imgID = single.back();
      single.pop_back();
      
      single = vecminus(single, _meanvec, dim);
    
      if (imgID % 100 == 0 && x % 350 == 0 ) {cout << "at image " << imgID << " vector: " << x << endl;}

      unsigned int hash = 0;
      for(int m = 0; m < numTables_; m++) {

          srpHash *_srp = _storesrp.at(m);
          unsigned int *hashcode = _srp->getHash(single, 450);

          hash = 0;

          // Convert to integer
          for (int n = 0; n < K_; n++) {
          hash += hashcode[n] * pow(2, (_srp->_numhashes - n - 1));
          }

          // TODO: This one line can be deleted.
          hashlst[m] = hash;

          hashes[x * numTables_ + m] = hash;
          delete [] hashcode;
        }

      ids[x] = imgID;
   }
   cout << "Hash done, inserting next" << endl;

   lsh_-> insertBatch(size, ids, hashes);
   cout << "insert done" << endl;

}

uint32_t *Slash::query(string filename){
    vector<vector<float>> mat = readvec(filename, 129);
    uint64_t size = mat.size();
    uint32_t dim = mat.at(0).size() - 1;
    cout << "size: " << size << "  dimension " << dim << endl;
    cout << "test vector: ";
    for (auto i : mat.at(0)) {
    cout << i << " ";
    }
    // TODO: Check that the first vector is read. 

    unordered_map<unsigned int, int> score;
    // Minus mean and query. 
    // uint32_t mark = 0;
    uint32_t *queries = new uint32_t[numTables_ * 350];
    uint32_t *result = new uint32_t[size / NUM_FEATURE];
    int count = 0;

    for (int x = 0; x < mat.size(); x++) {
       vector<float> queryvec = mat.at(x);
       unsigned int queryID = queryvec.back();
       queryvec.pop_back();

       queryvec = vecminus(queryvec, _meanvec, dim);

       unsigned int hash = 0;
       for (int m = 0; m < numTables_; m++) {
          srpHash *_srp = _storesrp.at(m);
          unsigned int *hashcode = _srp->getHash(queryvec, 450);

          // Convert to integer
          for (int n = 0; n < K_; n++) {
          hash += hashcode[n] * pow(2, (_srp->_numhashes - n - 1));
          }

          queries[(x % 350) * numTables_ + m] = hash;
          delete[] hashcode;
       }
      
      // When the vectors belonging to one image is processed. 
      if (x > 0 && x % 350 == 0){
          cout << "Querying id" << queryID << endl;
          unordered_map<unsigned int, int> score;
          uint32_t **retrieved = lsh_-> queryReservoirs(350, queries);
          for (int i = 0; i < numTables_ * NUM_FEATURE; j++) {
              for (int j = 0; j < RESERVOIR_SIZE; j++) {

                  if (score.count(retrieved[i][j]) == 0) {
                      score[retrieved[i][j]] = 0;
                  }
                  else {
                      score[retrieved[i][j]] ++;
                  }

              }
          }
          vector<pair<unsigned int, unsigned int> > freq_arr(score.begin(), score.end());
          sort(freq_arr.begin(), freq_arr.end(), comparePair());

          score.clear();
          if (freq_arr[0].first == -1) {
                cout << "Most match score is: " << freq_arr[1].second << endl;
                result[count] =  freq_arr[1].first;
          }
          cout << endl << "Most match score is: " << freq_arr[0].second << endl;
          result[count] = freq_arr[0].first;
          count ++;
      }
    }
    return result;
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

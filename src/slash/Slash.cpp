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

void Slash::storevec(string filename, uint64_t numItems,  size_t sample) {
  // Divide up the work
  auto p = partition(numItems);
  uint64_t myLen = p.first[rank_];
  uint64_t myOffset = p.second[rank_];

  // Read vectors
  cout << "Node: " << rank_ << " Reading vectors from number " << myOffset << " to " << myOffset + myLen << endl;
  vector<vector<float>> mat = readvec(filename, 129, myOffset, myLen);
  uint64_t size = mat.size();
  uint32_t dim = mat.at(0).size() - 1;
  cout << "Node: " << rank_ << " size: " << size << "  dimension " << dim << endl;
  // Sample to get mean. Coco_vector: 27M vectors
  float *sumvec = new float[128];
   cout << "Last test vector: ";
   for (auto i : mat.at(3499)) {
     cout << i << " ";
   }
  
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
  for (int j = 0; j < dim; j++) {
    _meanvec.push_back(sumvec[j]/(size/sample));
  }
  cout << endl << "Node: " << rank_  << " mean calculated. Begin hashing." << endl;

  // cout << "test mean vector: ";
  // for (auto i : _meanvec) {
  //   cout << i << " ";
  // }
  // cout << endl;

  // SRP hash & store
  uint32_t *ids = new uint32_t[size];
  uint32_t *hashlst = new uint32_t[numTables_];
  uint32_t *hashes = new uint32_t[numTables_ * size];

  
  for (int x = 0; x < mat.size(); x++) {

      vector<float> single = mat.at(x);

      // cout << "test vector 2: ";
      // for (auto i : single) {
      //   cout << i << " ";
      // }
      // cout << endl;

      unsigned int imgID = single.back();
      single.pop_back();
      
      single = vecminus(single, _meanvec, dim);
    
      if (imgID % 100 == 0 && x % 350 == 0 ) {cout << "Node: " << rank_ << " at image " << imgID << " vector: " << x << endl;}

      // cout << "test vector 3: ";
      // for (auto i : single) {
      //   cout << i << " ";
      // }
      unsigned int hash = 0;
      for(int m = 0; m < numTables_; m++) {

          srpHash *_srp = _storesrp.at(m);
          unsigned int *hashcode = _srp->getHash(single, 450);
          // cout << endl << "Hash code:  ";
          // for (int l = 0; l < K_; l++) {cout << hashcode[l] << " ";}
          // cout << endl;

          hash = 0;

          // Convert to integer
          for (int n = 0; n < K_; n++) {
          hash += hashcode[n] * pow(2, (_srp->_numhashes - n - 1));
          }

          // TODO: This one line can be deleted.
          hashlst[m] = hash;
          // cout << "id: " << imgID << " hash value: " << hash << endl;

          hashes[x * numTables_ + m] = hash;
          delete [] hashcode;
        }

      ids[x] = imgID;
   }
   cout << "Node: " << rank_ << " Hash done, inserting next" << endl;

   lsh_-> insertBatch(size, ids, hashes);
   cout << "Node: " << rank_ << " insert done" << endl;

}

vector<uint32_t> Slash::query(string filename, uint64_t numItems){

//     lsh_-> view();
    // auto p = partition_query(numItems, NUM_FEATURE);
    // uint64_t myLen = p.first[rank_];
    // uint64_t myOffset = p.second[rank_];

    vector<vector<float>> mat = readvec(filename, 129);
    uint64_t size = mat.size();
    uint32_t dim = mat.at(5).size() - 1;
    cout << "[Query] Node: " << rank_ << " size: " << size << "  dimension " << dim << endl;
    // cout << "test vector: ";
    // for (auto i : mat.at(0)) {
    // cout << i << " ";
    // }
    // TODO: Check that the first vector is read. 

    unordered_map<unsigned int, int> score;
    // Minus mean and query. 
    // uint32_t mark = 0;
    uint32_t *queries = new uint32_t[numTables_ * 350];
    vector<uint32_t> result;
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

                    hash = 0;
                    // Convert to integer
                    for (int n = 0; n < K_; n++) {
                            hash += hashcode[n] * pow(2, (_srp->_numhashes - n - 1));
                    }

                    // See range

                    queries[m * 350 + (x % 350)] = hash;
                    delete[] hashcode;
            }

            // When the vectors belonging to one image is processed.
            if (x > 0 && (x + 1) % NUM_FEATURE == 0) {
                    cout << "Node: " << rank_ << " Querying id: " << queryID << endl;
                    unordered_map<unsigned int, int> score;
                    // cout << "Initializing" << endl;
                    uint32_t **retrieved = lsh_->queryReservoirs(350, queries);
                    // cout << "Before updating score" << endl;
                    for (int i = 0; i < numTables_ * NUM_FEATURE; i++) {
                            for (int j = 0; j < RESERVOIR_SIZE; j++) {
                                    if (retrieved[i][j] == LSH::Empty) { continue; }

                                    if (score.count(retrieved[i][j]) == 0) {
                                            score[retrieved[i][j]] = 0;
                                    } else {
                                            score[retrieved[i][j]]++;
                                    }

                            }
                    }
                    cout << "Node: " << rank_ << " Score computed separately" << endl;
                    vector <pair<unsigned int, unsigned int>> freq_arr(score.begin(), score.end());
                    //Merge the maps of all the Nodes.
                    // First convert the map to normal array.
                    int arr_size = freq_arr.size() * 2;
                    cout << "Node: " << rank_ << " score map size: " << arr_size << endl;
//                    cout << "Node: " << " Inializing array"<< endl;
                    unsigned int *send_buf = new unsigned int[arr_size];

                    for (int i = 0; i < freq_arr.size(); i++) {
                            int idx = 2 * i;
                            send_buf[idx] = freq_arr.at(i).first;
                            send_buf[idx + 1] = freq_arr.at(i).second;
                    }
                    cout << "Node: " << rank_ << " Score converted to array" << endl;
                    // Send the sizes of each map first
                    int *rec_size_buf = new int[worldSize_];
                    int *send_size_buf = new int[1];
                    send_size_buf[0] = freq_arr.size() * 2;
                    MPI_Gather(send_size_buf, 1, MPI_INT, rec_size_buf, 1, MPI_INT, 0,
                               MPI_COMM_WORLD);


                    if (rank_ == 0) {
                            cout << "!!!! Root Node received sizes:";
                            for (int j = 0; j < worldSize_; j++) {
                                    cout << rec_size_buf[j] << " ";
                            }
                            cout << endl;
                    }


                    unsigned int *rec_buf;
                    unsigned int total = 0;
                    // Define the array of offsets
                    int *displs = new int[worldSize_];
                    unsigned int add = 0;
                    displs[0] = 0;
                    if (rank_ == 0) {
                            total = 0;
                            for (int i = 0; i < worldSize_; i++) {
                                    total += rec_size_buf[i];
                            }
                            rec_buf = new unsigned int[total];
                            for (int x = 1; x < worldSize_; x++) {
                                    add += rec_size_buf[x - 1];
                                    displs[x] = add;
                            }
                            cout << "Node 0 rec buffer initialized" << endl;
                    }

                    // Gather all the scores to Node 0
                    MPI_Gatherv(send_buf, arr_size, MPI_UNSIGNED, rec_buf, rec_size_buf, displs, MPI_UNSIGNED, 0,
                                MPI_COMM_WORLD);

                    if (rank_ == 0) {
                            cout << "!!!! Root Node received scores:";
                            for (int m = 0; m < 20; m++) {
                                    cout << rec_buf[m] << " ";
                            }
                            cout << endl;

                            unordered_map<unsigned int, unsigned int> new_score;
                            for (int j = 0; j < total; j += 2) {
                                    if (new_score.count(rec_buf[j]) == 0) {
                                            new_score[rec_buf[j]] = rec_buf[j + 1];
                                    } else {
                                            new_score[rec_buf[j]] += rec_buf[j + 1];
                                    }
                            }
                            vector <pair<unsigned int, unsigned int>> final_arr(new_score.begin(), new_score.end());
                            cout << "Root Node final score vector computed" << endl;

                            sort(final_arr.begin(), final_arr.end(), comparePair());

                            score.clear();
                            new_score.clear();
                            if (final_arr[0].first == -1) {
                                    cout << "Hit -1 :( Most match score is: " << final_arr[1].second << endl;
                                    result.push_back(final_arr[1].first);
                            }
                            cout << "Node: " << rank_ << " Querying id: " << queryID << " Match ID is: "
                                 << final_arr[0].first << " Most match score is: " << final_arr[0].second << endl
                                 << endl;
                            result.push_back(final_arr[0].first);
                            count++;
                            delete[] retrieved;
                    }
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

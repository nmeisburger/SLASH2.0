#include "DOPH.h"

void DOPH::getHashes(unsigned int *hashIndices, unsigned int *dataIdx, unsigned int *dataMarker,
                     size_t numInputEntries) {
#pragma omp parallel for
  for (size_t inputIdx = 0; inputIdx < numInputEntries; inputIdx++) {
    unsigned int *hashes = new unsigned int[_numhashes];
    unsigned int sizenonzeros = dataMarker[inputIdx + 1] - dataMarker[inputIdx];

    optimalMinHash(hashes, (unsigned int *)(dataIdx + dataMarker[inputIdx]), sizenonzeros);

    for (size_t tb = 0; tb < _L; tb++) {
      unsigned int index = 0;
      for (size_t k = 0; k < _K; k++) {
        unsigned int h = hashes[_K * tb + k];
        h *= _rand1[_K * tb + k];
        h ^= h >> 13;
        h ^= _rand1[_K * tb + k];
        index += h * hashes[_K * tb + k];
      }
      index = (index << 2) >> (32 - _rangePow);

      hashIndices[hashIndicesOutputIdx(_L, inputIdx, tb)] = index;
    }
    delete[] hashes;
  }
}

unsigned int DOPH::getRandDoubleHash(int binid, int count) {
  unsigned int tohash = ((binid + 1) << 10) + count;
  return ((unsigned int)_randHash[0] * tohash << 3) >>
         (32 - _lognumhash);  // _lognumhash needs to be ceiled.
}

void DOPH::optimalMinHash(unsigned int *hashArray, unsigned int *nonZeros,
                          unsigned int sizenonzeros) {
  /* This function computes the minhash and perform densification. */
  unsigned int *hashes = new unsigned int[_numhashes];

  unsigned int range = 1 << _rangePow;
  // binsize is the number of times the range is larger than the total number of hashes we need.
  unsigned int binsize = ceil(range / _numhashes);

  for (size_t i = 0; i < _numhashes; i++) {
    hashes[i] = NULL_HASH;
  }

  for (size_t i = 0; i < sizenonzeros; i++) {
    unsigned int h = nonZeros[i];
    h *= _randa;
    h ^= h >> 13;
    h *= 0x85ebca6b;
    unsigned int curhash =
        ((unsigned int)(((unsigned int)h * nonZeros[i]) << 5) >> (32 - _rangePow));
    unsigned int binid =
        std::min((unsigned int)floor(curhash / binsize), (unsigned int)(_numhashes - 1));
    if (hashes[binid] > curhash) hashes[binid] = curhash;
  }
  /* Densification of the hash. */
  for (unsigned int i = 0; i < _numhashes; i++) {
    unsigned int next = hashes[i];
    if (next != NULL_HASH) {
      hashArray[i] = hashes[i];
      continue;
    }
    unsigned int count = 0;
    while (next == NULL_HASH) {
      count++;
      unsigned int index = std::min((unsigned)getRandDoubleHash(i, count), _numhashes);
      next = hashes[index];
      if (count > 100)  // Densification failure.
        break;
    }
    hashArray[i] = next;
  }
  delete[] hashes;
}

void DOPH::showDOPHConfig() {
  printf("Random Seed 1: %d\n", _randHash[0]);
  printf("Random Seed 2: %d\n", _randHash[1]);
  printf("Hash Seed: %d\n", _randa);

  for (int i = 0; i < _K * _L; i++) {
    printf("Hash Param %d: %d\n", i, _rand1[i]);
  }
}

DOPH::DOPH(unsigned int _K_in, unsigned int _L_in, unsigned int _rangePow_in, int worldSize,
           int worldRank) {
  // Constant Parameters accross all nodes
  _K = _K_in;
  _L = _L_in;
  _numTables = _L_in;
  _rangePow = _rangePow_in;
  _numhashes = _K * _L;
  _lognumhash = log2(_numhashes);

  _rand1 = new int[_K * _L];
  _randHash = new int[2];

  // Random hash functions
  // srand(time(NULL));

  // Fixed random seed for hash functions
  srand(145297);

  // MPI

  _worldSize = worldSize;
  _worldRank = worldRank;

  if (_worldRank == 0) {
    for (int i = 0; i < _numhashes; i++) {
      _rand1[i] = rand();
      if (_rand1[i] % 2 == 0) _rand1[i]++;
    }

    // _randa and _randHash* are random odd numbers.
    _randa = rand();
    if (_randa % 2 == 0) _randa++;
    _randHash[0] = rand();
    if (_randHash[0] % 2 == 0) _randHash[0]++;
    _randHash[1] = rand();
    if (_randHash[1] % 2 == 0) _randHash[1]++;
  }

  MPI_Bcast(&_randa, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(_randHash, 2, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(_rand1, _numhashes, MPI_INT, 0, MPI_COMM_WORLD);

  std::cout << "LSH Initialized in Node " << _worldRank << std::endl;
}

DOPH::~DOPH() {
  delete[] _randHash;
  delete[] _rand1;
}
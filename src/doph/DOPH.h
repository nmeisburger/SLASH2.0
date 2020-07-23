#pragma once
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <random>

#define UNIVERSAL_HASH(x, M, a, b) ((unsigned)(a * x + b) >> (32 - M))
#define BINARY_HASH(x, a, b) ((unsigned)(a * x + b) >> 31)

#define hashIndicesOutputIdx(numTables, dataIndx, tb) \
  (unsigned long long)(numTables * dataIndx + tb)

#define NULL_HASH -1

class DOPH {
 private:
  unsigned int _rangePow;
  unsigned int _numTables;
  unsigned int _numhashes, _lognumhash, _K, _L;
  int *_randHash, _randa, *_rand1;

  int _worldSize, _worldRank;

  unsigned int getRandDoubleHash(int binid, int count);

  void optimalMinHash(unsigned int *hashArray, unsigned int *nonZeros, unsigned int sizenonzeros);

 public:
  void getHashes(unsigned int *hashIndices, unsigned int *dataIdx, unsigned int *dataMarker,
                 size_t numInputEntries);

  void showDOPHConfig();

  DOPH(unsigned int _K_in, unsigned int _L_in, unsigned int _rangePow_in, int worldSize,
       int worldRank);

  ~DOPH();
};
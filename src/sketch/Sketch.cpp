#include "Sketch.h"

Sketch::Sketch(uint64_t numHashes, uint64_t rowSize, uint64_t numSketches, int rank, int worldSize)
    : _numHashes(numHashes),
      _rowSize(rowSize),
      _numSketches(numSketches),
      _rank(rank),
      _worldSize(worldSize) {
  _sketchSize = _rowSize * _numHashes;
  _totalSize = _sketchSize * _numSketches;

  _seeds = new uint32_t[_numHashes];

  for (uint32_t i = 0; i < _numHashes; i++) {
    _seeds[i] = rand();
  }

  _sketch = new SketchItem[_sketchSize * _numSketches]();
}

void Sketch::hash(uint32_t item, uint32_t *hashes) {
  for (size_t hashIndx = 0; hashIndx < _numHashes; hashIndx++) {
    unsigned int h = _seeds[hashIndx];
    unsigned int k = item;
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    uint32_t curhash = h % _rowSize;
    hashes[hashIndx] = curhash;
  }
}

void Sketch::merge(SketchItem *other) {
  SketchItem *curr;
  SketchItem *otherCurr;
  for (size_t i = 0; i < _totalSize; i++) {
    otherCurr = other + i;
    curr = _sketch + i;
    if (curr->heavyHitter == otherCurr->heavyHitter) {
      curr->count += otherCurr->count;
      continue;
    }
    if (curr->count < otherCurr->count) {
      curr->heavyHitter = otherCurr->heavyHitter;
      curr->count = otherCurr->count - curr->count;
    } else if (curr->count > otherCurr->count) {
      curr->count -= otherCurr->count;
    } else {
      curr->heavyHitter = otherCurr->heavyHitter;
      curr->count = 1;
    }
  }
}

void Sketch::mergeAll() {
  uint32_t numIterations = std::ceil(std::log(_worldSize) / std::log(2));
  SketchItem *recvBuffer = new SketchItem[_totalSize];
  uint64_t combinedSize = 3 * _totalSize;
  MPI_Status status;
  for (uint32_t iter = 0; iter < numIterations; iter++) {
    if (_rank % ((int)std::pow(2, iter + 1)) == 0 && (_rank + std::pow(2, iter)) < _worldSize) {
      int source = _rank + std::pow(2, iter);
      MPI_Recv(recvBuffer, combinedSize, MPI_UNSIGNED, source, iter, MPI_COMM_WORLD, &status);
      merge(recvBuffer);
      // printf("Iteration %d: Node %d: Recv from %d\n", iter, _myRank, source);
    } else if (_rank % ((int)std::pow(2, iter + 1)) == ((int)std::pow(2, iter))) {
      int destination = _rank - ((int)std::pow(2, iter));
      MPI_Send(_sketch, combinedSize, MPI_UNSIGNED, destination, iter, MPI_COMM_WORLD);
      // printf("Iteration %d: Node %d: Send from %d\n", iter, _myRank, destination);
    }
  }
  delete[] recvBuffer;
}

void Sketch::insert(uint32_t *items, uint64_t itemsPerSketch) {
  SketchItem *curr;
  uint32_t newItem;

  for (size_t sketch = 0; sketch < _numSketches; sketch++) {
    uint32_t *hashes = new uint32_t[_numHashes];
    for (size_t n = 0; n < itemsPerSketch; n++) {
      newItem = items[sketch * itemsPerSketch + n];
      hash(newItem, hashes);
      for (size_t h = 0; h < _numHashes; h++) {
        curr = _sketch + index(sketch, h, hashes[h]);
        if (curr->heavyHitter == newItem) {
          curr->count++;
        } else {
          if (curr->count > 1) {
            curr->count--;
          } else {
            curr->heavyHitter = newItem;
            curr->count = 1;
          }
        }
      }
    }
    delete[] hashes;
  }
}

TopKResult *Sketch::topK(uint64_t k, uint32_t threshold) {
  TopKResult *result = new TopKResult[_numSketches]();

  for (size_t s = 0; s < _numSketches; s++) {
    result[s].items = new uint32_t[k];
    SketchItem curr;
    uint32_t hashes[_numHashes];
    std::vector<SketchItem> candidates;
    for (size_t i = 0; i < _sketchSize; i++) {
      curr = _sketch[index(s, 0, i)];
      if (curr.count > threshold) {
        candidates.push_back(std::move(curr));
      } else {
        hash(curr.heavyHitter, hashes);
        for (size_t r = 0; r < _numHashes; r++) {
          curr = _sketch[index(s, r, hashes[r])];
          if (curr.count > threshold) {
            candidates.push_back(std::move(curr));
            break;
          }
        }
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](SketchItem &a, SketchItem &b) { return a.count > b.count; });

    size_t _k = k;

    result[s].count = std::min(candidates.size(), _k);
    for (size_t i = 0; i < result[s].count; i++) {
      result[s].items[i] = candidates[i].heavyHitter;
    }
  }

  return result;
}

void Sketch::view() {
  for (size_t s = 0; s < _numSketches; s++) {
    printf("Sketch %lu:\n", s);
    for (size_t h = 0; h < _numHashes; h++) {
      printf("[%lu] ", h);
      for (size_t r = 0; r < _rowSize; r++) {
        SketchItem &item = _sketch[s * _sketchSize + h * _rowSize + r];
        if (item.heavyHitter == LSH::Empty) {
          printf("(--, %u) ", item.count);
        } else {
          printf("(%u, %u) ", item.heavyHitter, item.count);
        }
      }
      printf("\n");
    }
    printf("\n");
  }
}

#ifndef SURF_SRP_SRPHASH_H
#define SURF_SRP_SRPHASH_H

#include <vector>

class srpHash {
private:
    size_t _dim;
    size_t  _samSize;
    short ** _randBits;
    int ** _indices;
    int seed;
public:
    int _numhashes;
    srpHash(size_t dimension, size_t numOfHashes, int ratio, int seed);
    unsigned int * getHash(std::vector<float> vec, int length);
    ~srpHash();
};


#endif //SURF_SRP_SRPHASH_H

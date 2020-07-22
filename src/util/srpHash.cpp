
#include "srpHash.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <ctime>

using namespace std;

srpHash::srpHash(size_t dimension, size_t numOfHashes, int ratio, int seed) {
        _dim = dimension;
        _numhashes = numOfHashes;
        _samSize = ceil(1.0*_dim / ratio);

        int *a = new int[_dim];
        for (size_t i = 0; i < _dim; i++) {
                a[i] = i;
        }

        srand(seed);
        _randBits = new short *[_numhashes];
        _indices = new int *[_numhashes];


        // auto rd = std::random_device{};
        // auto rng = default_random_engine {rd()};
        // mt19937 mt(rd());
        // uniform_int_distribution<int> dist(0, numeric_limits<int>::max());
        for (size_t i = 0; i < _numhashes; i++) {
                random_shuffle(a, a+_dim);
                // shuffle(a, a+_dim, rng);
//                cout << "a[i] is " << a[i] << endl;
                _randBits[i] = new short[_samSize];
                _indices[i] = new int[_samSize];
                for (size_t j = 0; j < _samSize; j++) {
                        _indices[i][j] = a[j];
                        // int curr = dist(mt);
                       int curr = rand();
                        if (curr % 2 == 0) {
                                _randBits[i][j] = 1;
                        } else {
                                _randBits[i][j] = -1;
                        }
                }
                std::sort(_indices[i], _indices[i]+_samSize);
        }
        delete [] a;
}

unsigned int * srpHash::getHash(vector<float> vec, int length) {
        auto *hashes = new unsigned int[_numhashes];

        // #pragma omp parallel for
        for (size_t i = 0; i < _numhashes; i++) {
                double s = 0;
                for (size_t j = 0; j < _samSize; j++) {
                        float v = vec.at(_indices[i][j]);
//                        cout << "In srpHash, float v: " << v << endl;
                        if (_randBits[i][j] >= 0) {
                                s += v;
                        } else {
                                s -= v;
                        }
                }
                hashes[i] = (s >= 0 ? 0 : 1);
        }
        return hashes;
}

srpHash::~srpHash() {
        for (size_t i = 0; i < _numhashes; i++) {
                delete[]   _randBits[i];
                delete[]   _indices[i];
        }
        delete[]   _randBits;
        delete[]   _indices;

}



#pragma once

#include <assert.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

struct ReadResult {
  uint32_t *indices;
  float *values;
  uint32_t *markers;
  uint32_t *labels;
  uint64_t n;
};

class Reader {
 private:
  std::string _filename;

  std::shared_ptr<std::ifstream> _file;

  uint32_t _avgDim;

  char *_buffer;

 public:
  static std::vector<std::string> split(std::string s, std::string delimeter);

  Reader(std::string filename, uint32_t avgDim, uint64_t offset = 0, size_t blockSize = 1000000)
      : _filename(filename), _avgDim(avgDim) {
    _file = std::make_shared<std::ifstream>(filename);

    if (!_file->is_open() || _file->bad() || !_file->good() || _file->fail()) {
      throw std::runtime_error("Invalid file provided to reader.");
    }

    _buffer = new char[blockSize];
    _file->rdbuf()->pubsetbuf(_buffer, blockSize);

    std::string line;
    if (offset > 0) {
      size_t o = 0;
      while (std::getline(*_file, line) && o < offset) {
        o++;
      }
    }
  }

  ~Reader() {}

  ReadResult readSparse(uint64_t n);
};

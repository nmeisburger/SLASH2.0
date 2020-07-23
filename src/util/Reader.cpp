#include "Reader.h"

std::vector<std::string> Reader::split(std::string s, std::string delimeter) {
  std::vector<std::string> contents;

  std::string str = s;
  size_t loc;
  while ((loc = str.find_first_of(delimeter)) != std::string::npos) {
    contents.push_back(str.substr(0, loc));
    str = str.substr(loc + 1);
  }
  if (str != "") {
    contents.push_back(str);
  }
  return contents;
}

ReadResult Reader::readSparse(uint64_t n) {
  auto start = std::chrono::system_clock::now();

  uint64_t bufferLen = _avgDim * n;

  ReadResult result;
  result.indices = new uint32_t[bufferLen];
  result.values = new float[bufferLen];
  result.markers = new uint32_t[n + 1];
  result.labels = new uint32_t[n];

  uint64_t numVecs = 0;
  uint64_t totalDim = 0;

  std::string line;

  unsigned int index, pos;
  float val;

  while (std::getline(*_file, line) && numVecs < n) {
    std::vector<std::string> contents = split(line, " ");
    result.labels[numVecs] = stoul(contents.at(0));

    result.markers[numVecs] = totalDim;
    for (size_t i = 1; i < contents.size(); i++) {
      pos = contents.at(i).find_first_of(":");
      index = stoul(contents.at(i).substr(0, pos));
      val = stof(contents.at(i).substr(pos + 1, contents.at(i).length()));
      result.indices[totalDim] = index;
      result.values[totalDim] = val;
      totalDim++;
    }

    numVecs++;
  }

  if (numVecs != n) {
    throw std::runtime_error("Unable to read specified number of vectors from file.");
  }

  result.markers[numVecs] = totalDim;
  result.n = numVecs;

  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed = end - start;

  printf("Read Complete: %fs %lu vectors, total dim: %lu\n", elapsed.count(), numVecs, totalDim);

  return result;
}
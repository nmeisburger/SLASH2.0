#include <stdint.h>

#include <cmath>
#include <iostream>

#include "../lsh/LSH.h"

float SparseVecMul(uint32_t *indicesA, float *valuesA, uint32_t sizeA, uint32_t *indicesB,
                   float *valuesB, uint32_t sizeB) {
  float result = 0;
  uint32_t ctA = 0;
  uint32_t ctB = 0;
  uint32_t iA, iB;

  /* Maximum iteration: nonzerosA + nonzerosB.*/
  while (ctA < sizeA && ctB < sizeB) {
    iA = indicesA[ctA];
    iB = indicesB[ctB];

    if (iA == iB) {
      result += valuesA[ctA] * valuesB[ctB];
      ctA++;
      ctB++;
    } else if (iA < iB) {
      ctA++;
    } else if (iA > iB) {
      ctB++;
    }
  }
  return result;
}

float cosineDist(uint32_t *indiceA, float *valA, uint32_t nonzerosA, uint32_t *indiceB, float *valB,
                 uint32_t nonzerosB) {
  float up = 0;
  float a = 0;
  float b = 0;
  uint32_t startA, endA, startB, endB;

  up = SparseVecMul(indiceA, valA, nonzerosA, indiceB, valB, nonzerosB);
  a = SparseVecMul(indiceA, valA, nonzerosA, indiceA, valA, nonzerosA);
  b = SparseVecMul(indiceB, valB, nonzerosB, indiceB, valB, nonzerosB);
  a = sqrtf(a);
  b = sqrtf(b);
  if (a == 0 || b == 0) {
    return 0;
  }
  return up / (a * b);
}

void similarityMetric(uint32_t *queries_indice, float *queries_val, uint32_t *queries_marker,
                      uint32_t *bases_indice, float *bases_val, uint32_t *bases_marker,
                      uint32_t *queryOutputs, uint32_t numQueries, uint32_t topk, uint32_t *nList,
                      uint32_t nCnt) {
  float *out_avt = new float[nCnt]();

  uint32_t *cnts = new uint32_t[nCnt]();

  std::cout << "[similarityMetric] Averaging output. " << std::endl;
  /* Output average. */
  for (size_t i = 0; i < numQueries; i++) {
    uint32_t startA, endA;
    startA = queries_marker[i];
    endA = queries_marker[i + 1];
    for (uint32_t j = 0; j < topk; j++) {
      uint32_t query = queryOutputs[i * topk + j];
      if (query == LSH::Empty) {
        continue;
      }
      if (query >= 1000) {
        printf("ERROR: %u\n", query);
        exit(1);
      }
      uint32_t startB, endB;
      startB = bases_marker[query];
      endB = bases_marker[query + 1];
      float dist = cosineDist(queries_indice + startA, queries_val + startA, endA - startA,
                              bases_indice + startB, bases_val + startB, endB - startB);
      for (uint32_t n = 0; n < nCnt; n++) {
        if (j < nList[n]) {
          out_avt[n] += dist;
          cnts[n]++;
        }
      }
    }
  }

  /* Print results. */
  printf(
      "\nS@k = s_out(s_true): In top k, average output similarity (average "
      "groundtruth similarity). \n");
  for (uint32_t n = 0; n < nCnt; n++) {
    printf("S@%d = %1.3f \n", nList[n], out_avt[n] / (cnts[n]));
  }
  // for (uint32_t n = 0; n < nCnt; n++) printf("%d ", nList[n]);
  // printf("\n");
  // for (uint32_t n = 0; n < nCnt; n++) printf("%1.3f ", out_avt[n] / (numQueries * nList[n]));
  // printf("\n");
}
#include <mpi.h>

#include <iostream>

#include "slash/Slash.h"
#include "util/Eval.h"
#include "util/Reader.h"

int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
  int my_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const std::string webspam =
      "/mnt/c/Users/nmeis/Research/SLASH/dataset/webspam/webspam_trigram.svm";

  Slash s(24, 4, 64, 12);

  s.store(webspam, 1000, 10, 4000, 100);

  auto query = s.topK(webspam, 4000, 100, 32);

  Reader r(webspam, 4000);

  auto data = r.readSparse(1100);

  unsigned int nlist[] = {1, 4, 8, 16, 32};

  similarityMetric(data.indices, data.values, data.markers, data.indices, data.values,
                   data.markers + 100, query, 100, 32, nlist, 5);

  MPI_Finalize();

  return 0;
}

#include "gtest/gtest.h"
#include <mpi.h>
#include <omp.h>

int main(int argc, char **argv) {
    MPI_Init(0, 0);

    omp_set_num_threads(1);

    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
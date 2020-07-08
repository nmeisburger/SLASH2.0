#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Finalize();

    return 0;
}

// #include "Slash.h"

// void Slash::store(vector<pair<string, uint32_t>> &imagesAndIds) {

//     size_t len = imagesAndIds.size() / worldSize_;
//     if (rank_ < (imagesAndIds.size() % worldSize_)) {
//         len++;
//     }

//     uint32_t *ids = new uint32_t[len];
//     uint32_t **hashes = new uint32_t *[len];

//     size_t offset =
//         imagesAndIds.size() / worldSize_ + min((size_t)rank_, (imagesAndIds.size() %
//         worldSize_));

// #pragma omp parallel for default(none) shared(imagesAndIds, ids, hashes, len, offset)
//     for (size_t i = 0; i < len; i++) {
//         hashes[i] = imageProcessor_(imagesAndIds[i + offset].first);
//         ids[i] = imagesAndIds[i + offset].second;
//     }

//     lsh_->insert(len, hashes, ids);

//     for (size_t i = 0; i < len; i++) {
//         delete[] hashes[i];
//     }
//     delete[] ids;
//     delete[] hashes;
// }

// Item *Slash::query(vector<string> &queries, uint32_t topK) {

//     size_t querySize = queries.size();

//     uint32_t **queryHashes = new uint32_t *[querySize];

// #pragma omp parallel for default(none) shared(queries, queryHashes, querySize)
//     for (size_t q = 0; q < querySize; q++) {
//         queryHashes[q] = imageProcessor_(queries[q]);
//     }

//     Item *recvBuffer = new Item[querySize * topK];

//     Item *scratchBuffer = new Item[querySize * topK];

//     Item *localTopK = lsh_->topK(querySize, topK, queryHashes);

//     size_t mpiCopySize = 2 * topK * querySize;
//     uint32_t numIterations = ceil(log(worldSize_) / log(2));
//     MPI_Status status;

//     for (uint32_t iter = 0; iter < numIterations; iter++) {
//         if (rank_ % ((int)pow(2, iter + 1)) == 0 && (rank_ + pow(2, iter)) < worldSize_) {
//             int source = rank_ + pow(2, iter);
//             MPI_Recv(recvBuffer + querySize * topK, mpiCopySize, MPI_UNSIGNED, source, iter,
//                      MPI_COMM_WORLD, &status);

// #pragma omp parallel for default(none) shared(topK, querySize, recvBuffer, localTopK,
// scratchBuffer)
//             for (size_t q = 0; q < querySize; q++) {
//                 copy(recvBuffer + q * querySize, recvBuffer + (q + 1) * querySize,
//                      scratchBuffer + 2 * querySize * q);

//                 copy(localTopK + q * querySize, localTopK + (q + 1) * querySize,
//                      scratchBuffer + (2 * q + 1) * querySize);

//                 sort(scratchBuffer + 2 * q * querySize, scratchBuffer + (2 * q + 2) * querySize,
//                      [](Item &a, Item &b) { return a.cnt >= b.cnt; });

//                 copy(scratchBuffer + 2 * q * querySize, scratchBuffer + (2 * q + 1) * querySize,
//                      localTopK + q * querySize);
//             }

//         } else if (rank_ % ((int)pow(2, iter + 1)) == ((int)pow(2, iter))) {
//             int destination = rank_ - ((int)pow(2, iter));
//             MPI_Send(localTopK, mpiCopySize, MPI_UNSIGNED, destination, iter, MPI_COMM_WORLD);
//         }
//     }

//     for (size_t i = 0; i < querySize; i++) {
//         delete[] queryHashes[i];
//     }

//     delete[] scratchBuffer;
//     delete[] recvBuffer;

//     return localTopK;
// }
# SLASH
High performance distributed nearest neighbor system using LSH and sketching.

## Development
1. `$ git clone https://github.com/nmeisburger0/SLASH2.0`
2. `$ cd SLASH`
3. `$ git submodule update --init`
4. (If running on NOTS) `$ source scripts/setup.sh`
5. `$ mkdir build`
6. `$ cd build`
7. `$ cmake ..`
8. `$ make all`
9. To run the program run `$ build/src/SLASH_run`.
10. For testing run `$ build/test/SLASH_test`.

## Testing
This package is linked with the gtest and gmock testing and mocking libraries. This allow for easy creation of unit tests and mock classes. For more information see the documentation: 
* [GTest Documentation](https://github.com/google/googletest/tree/master/googletest/docs)
* [GMock Documentation](https://github.com/google/googletest/blob/master/googlemock/docs)

# SLASH
High preformance distributed nearest neighbor system using LSH and sketching.

## Development
1. Clone this repository
2. `$ cd SLASH`
3. `$ mkdir lib; mkdir build`
4. `$ cd lib`
5. `$ git clone https://github.com/google/googletest/`
6. `$ cd ..`
7. `$ source scripts/setup.sh`
8. `$ cd build`
9. `$ cmake ..`
10. `$ make all`
11. To run the program run `$ src/SLASH_run` from the build directory
12. For testing run `$ test/SLASH_test` from the build directory

## Testing
This package is linked with the gtest and gmock testing and mocking libraries. This allow for easy creation of unit tests and mock classes. For more information see the documentation: 
* [GTest Documentation](https://github.com/google/googletest/tree/master/googletest/docs)
* [GMock Documentation](https://github.com/google/googletest/blob/master/googlemock/docs)

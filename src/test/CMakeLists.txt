# Cmake file for the test executable

file (GLOB_RECURSE TEST_SRC *.cpp)
add_executable(test-nn ${TEST_SRC})

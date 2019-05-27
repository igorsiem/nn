# Top-level Cmake file for nn project

cmake_minimum_required(VERSION 3.10)
project($ENV{QPRJ_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_subdirectory(test)

cmake_minimum_required(VERSION 3.5)
project(AI)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(AI_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)


#setting library paths:
set(ARMADILLO_LIB /usr/lib/x86_64-linux-gnu/)
set(ARMADILLO_INCLUDE /usr/include/)
set(INTEL /home/fran/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64_lin/)
LINK_DIRECTORIES(${ARMADILLO_LIB} ${INTEL})

include_directories("${AI_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${ARMADILLO_INCLUDE}")


ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

add_subdirectory(NLP)
add_subdirectory(RBM)
add_subdirectory(Optimizers)
add_subdirectory(tests)
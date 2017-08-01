cmake_minimum_required(VERSION 3.5)
project (Optimizers)

set (OPTIMIZERS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set (OPTIMIZERS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)


file(GLOB_RECURSE SRCS *.cpp)
file(GLOB_RECURSE HDRS *.h)

set(OPTIMIZERS_SRCS
        ${SRCS}
        ${HDRS}
        )

include_directories("${OPTIMIZERS_SOURCE_DIR}")
include_directories("${OPTIMIZERS_INCLUDE_DIR}")

#add_library(${PROJECT_NAME} SHARED ${OPTIMIZERS_SRCS})
add_executable(${PROJECT_NAME} ${OPTIMIZERS_SRCS})
target_link_libraries(${PROJECT_NAME} armadillo mkl_rt)
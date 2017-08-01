cmake_minimum_required(VERSION 3.5)
project (RBM)

set (RBM_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set (RBM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB_RECURSE SRCS *.cpp)
file(GLOB_RECURSE HDRS *.h)

set(RBM_SRCS
        ${SRCS}
        ${HDRS}
        )

include_directories("${RBM_SOURCE_DIR}")
include_directories("${RBM_INCLUDE_DIR}")

add_library(${PROJECT_NAME} SHARED ${RBM_SRCS})
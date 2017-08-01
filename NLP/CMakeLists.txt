cmake_minimum_required(VERSION 3.5)
project (NLP)

set (NLP_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set (NLP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB_RECURSE SRCS *.cpp)
file(GLOB_RECURSE HDRS *.h)

set(NLP_SRCS
        ${SRCS}
        ${HDRS}
        ${}
        )

include_directories("${NLP_SOURCE_DIR}")
include_directories("${NLP_INCLUDE_DIR}")

add_library(${PROJECT_NAME} SHARED ${NLP_SRCS})
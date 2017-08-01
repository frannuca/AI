cmake_minimum_required(VERSION 3.5)
project (tests)

set (TEST_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set (TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB_RECURSE SRCS *.cpp)
file(GLOB_RECURSE HDRS *.h)

set(TEST_SRCS
        ${SRCS}
        ${HDRS}
        )

include_directories("${TEST_SOURCE_DIR}")
include_directories("${TEST_INCLUDE_DIR}")

add_executable(${PROJECT_NAME} ${TEST_SRCS})

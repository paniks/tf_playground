cmake_minimum_required(VERSION 3.10)
project(tf_playground)

set(CMAKE_CXX_STANDARD 14)

# HDF5
find_package(HDF5 REQUIRED COMPONENTS C CXX HL REQUIRED)
include_directories( ${HDF5_INCLUDE_DIRS} )
link_directories( ${HDF5_LIBRARY_DIRS} )

add_executable(tf_playground main.cpp)

target_link_libraries(tf_playground
        ${HDF5_CXX_LIBRARIES}
        )

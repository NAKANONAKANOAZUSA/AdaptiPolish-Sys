cmake_minimum_required(VERSION 3.10)
project(pc_backend)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PCL REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

pybind11_add_module(pc_backend pc_backend.cpp)
target_link_libraries(pc_backend PRIVATE ${PCL_LIBRARIES})

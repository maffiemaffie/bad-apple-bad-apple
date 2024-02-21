cmake_minimum_required(VERSION 3.24)
project(himaffie)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4.9.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(himaffie main.cpp)
target_link_libraries(himaffie ${OpenCV_LIBS})

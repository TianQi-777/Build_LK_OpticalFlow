cmake_minimum_required(VERSION 2.8)

project(optical_flow)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")


find_package(OpenCV 3.0 QUIET)

include_directories( "/usr/include/eigen3" )


add_executable(optical_flow optical_flow)


target_link_libraries( optical_flow
${OpenCV_LIBS}
)

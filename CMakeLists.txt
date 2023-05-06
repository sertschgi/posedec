cmake_minimum_required(VERSION 3.10)

project(posedec VERSION 1.0)

# Set the build directory
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)

# Make sure the build directory exists
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR})

# Set the directories for external dependencies
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/external/opencv")
set(cxxopts_DIR "${CMAKE_SOURCE_DIR}/external/cxxopts")

# Find the external libraries
find_package(OpenCV REQUIRED)
add_subdirectory(external/cxxopts)

# Add the include directories for the external libraries
include_directories(include ${OpenCV_INCLUDE_DIRS} ${cxxopts_SOURCE_DIR}/include)

# Add the source files to the build
add_executable(posedec src/detector.cpp)

# Link against the external libraries
target_link_libraries(posedec ${OpenCV_LIBS} cxxopts)

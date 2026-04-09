###########################################################
#                                                         #
# Copyright (c)                                           #
#                                                         #
# The Verifiable & Control-Theoretic Robotics (VECTR) Lab #
# University of California, Los Angeles                   #
#                                                         #
# Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   #
# Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         #
#                                                         #
###########################################################

cmake_minimum_required(VERSION 3.12.4)
project(direct_lidar_inertial_odometry VERSION 1.1.1)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package( PCL REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

find_package( Eigen3 REQUIRED )
include_directories(${EIGEN3_INCLUDE_DIR})

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)

include_directories(include)

# Not all machines have <cpuid.h> available
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
try_compile(HAS_CPUID ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
file(REMOVE ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
if(HAS_CPUID)
  add_compile_definitions(HAS_CPUID)
endif()

# NanoFLANN
find_package(nanoflann CONFIG REQUIRED)

# NanoGICP
add_library(nano_gicp STATIC
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
target_link_libraries(nano_gicp PUBLIC ${PCL_LIBRARIES} ${OpenMP_LIBS} nanoflann::nanoflann)
target_include_directories(nano_gicp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Odometry Node
add_library(dlio_odom_node src/dlio/odom.cc)
target_compile_options(dlio_odom_node PRIVATE ${OpenMP_FLAGS})
target_link_libraries(dlio_odom_node PUBLIC ${PCL_LIBRARIES} ${OpenMP_LIBS} nano_gicp)
target_include_directories(dlio_odom_node PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Mapping Node
# add_executable (dlio_map_node src/dlio/map_node.cc src/dlio/map.cc)
# target_compile_options(dlio_map_node PRIVATE ${OpenMP_FLAGS})
# target_link_libraries(dlio_map_node ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads)

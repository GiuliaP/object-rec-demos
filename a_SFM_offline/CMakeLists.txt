# Copyright: (C) 2015 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required (VERSION 2.8)
set(PROJECTNAME SFM_offline)
project(${PROJECTNAME})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV REQUIRED)
find_package(OpenMP)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")

if(OPENMP_FOUND)

   message(STATUS "OPENMP FOUND")

   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

   set (folder_source_elas
               elas/src/elas_omp.cpp
               elas/src/descriptor.cpp
               elas/src/filter.cpp
               elas/src/matrix.cpp
               elas/src/triangle.cpp)

else()

message(STATUS "OPENMP NOT FOUND")

set (folder_source_elas
               elas/src/elas.cpp
               elas/src/descriptor.cpp
               elas/src/filter.cpp
               elas/src/matrix.cpp
               elas/src/triangle.cpp)

endif()

set (folder_header_elas 
                  elas/include/elas.h
                  elas/include/descriptor.h
                  elas/include/image.h
                  elas/include/filter.h
                  elas/include/timer.h
                  elas/include/matrix.h
                  elas/include/triangle.h)

include_directories(${PROJECT_SOURCE_DIR}/elas/include)

source_group("ELAS Source Files" FILES ${folder_source_elas})
source_group("ELAS Header Files" FILES ${folder_header_elas})

set(folder_source main.cpp elasWrapper.cpp elasWrapper.h)

source_group("Source Files" FILES ${folder_source})

add_executable(${PROJECTNAME} ${folder_source} ${folder_source_elas} ${folder_header_elas})

target_link_libraries(${PROJECTNAME} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
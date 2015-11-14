# Copyright: 2015 Istituto Italiano di Tecnologia, iCub Facility
# Authors: Sean Ryan Fanello, Carlo Ciliberto, Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8)

set(projectname linearClassifier_offline)
project(${projectname})

find_package(Gurls REQUIRED)

set(folder_header ${folder_header} linearClassifier.h RLSlinear.h)                                                  
set(folder_source ${folder_source} main.cpp linearClassifier.cpp RLSlinear.cpp)                                                 

include_directories(${Gurls_INCLUDE_DIRS})

add_definitions(-DGURLS_AVAILABLE=1)
add_definitions(${Gurls_DEFINITIONS})

source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})

add_executable(${projectname} ${folder_source} ${folder_header})

target_link_libraries(${projectname} ${Gurls_LIBRARIES})
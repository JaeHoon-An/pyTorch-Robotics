cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pyTorch-Robotics)

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/Library/libtorch")
ADD_COMPILE_DEFINITIONS (DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/")
ADD_COMPILE_DEFINITIONS (MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/model/")

set(REGRESSION TRUE)

if(${REGRESSION})
    add_subdirectory(regression)
endif()
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(regression)

find_package(Qt5 CONFIG REQUIRED COMPONENTS Widgets)
find_package(Torch REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(regression
        main.cpp
        RegressionUI/mainwindow.cpp
        RegressionUI/qcustomplot.cpp)
target_link_libraries(regression
        "${TORCH_LIBRARIES}"
        pthread
        Qt5::Widgets)
set_property(TARGET regression PROPERTY CXX_STANDARD 14)
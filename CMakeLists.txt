cmake_minimum_required(VERSION 3.5.1)

project(wf-cuda)

option(TEST_WITH_GUI "Test with GUI" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_CXX_FLAGS "--std=c++11")
set(CUDA_PROPAGATE_HOST_FLAGS ON)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(GTest)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
)

if(${TEST_WITH_GUI})
    add_definitions(-DTEST_WITH_GUI)
endif()


set(sources
    cuda_utils.cpp  cuda_utils.h
    cameramodel.h
    kalmanline.h
    filters.h
    filters/box.cu
)

if(${GTest_FOUND})
    cuda_add_executable(wf-test
        ${sources}
        "test/test.cpp"
        "test/utils.h"
        test/test_gpu.cpp
        test/test_cameramodel.cpp
        test/test_kalmanline.cpp
        test/test_filters.cpp
        utils.cpp       utils.h
        "filters.cu"
    )
    target_link_libraries(wf-test
        ${GTEST_LIBRARIES}
        ${OpenCV_LIBS}
        ${OpenCV_LIBRARIES}
    )
endif()

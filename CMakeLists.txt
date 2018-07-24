cmake_minimum_required(VERSION 3.5.1)

project(wf-cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(GTest)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
#    ${Eigen3_INCLUDE_DIRS}
)


if(${GTest_FOUND})
    cuda_add_executable(wf-test
        "test/test.cpp"
        test/test_gpu.cpp
        utils.cpp       utils.h
        "boxFilter.cu"
    )
    target_link_libraries(wf-test
        ${GTEST_LIBRARIES}
        ${OpenCV_LIBS}
        ${OpenCV_LIBRARIES}
    )
endif()

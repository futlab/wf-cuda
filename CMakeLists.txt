cmake_minimum_required(VERSION 3.5.1)

project(wf-cuda)

find_package(CUDA REQUIRED)
find_package(gtest)

cuda_add_executable(${PROJECT_NAME}
    "main.cpp"
    "boxFilter.cu"
)

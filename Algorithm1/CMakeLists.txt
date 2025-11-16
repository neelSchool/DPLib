cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(TestTorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/home/neel/libtorch")   # <-- CHANGE THIS

find_package(Torch REQUIRED)

add_executable(train train.cpp)
target_link_libraries(train "${TORCH_LIBRARIES}")


set_property(TARGET train PROPERTY CXX_STANDARD 17)

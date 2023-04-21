cmake_minimum_required(VERSION 3.0.0)
project(dlcpp VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3")

set(Python3_ROOT_DIR /home/liumu/anaconda3/envs/DL)
find_package(Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
link_directories(${Python3_LINRARY_DIRS})

set(CMAKE_PREFIX_PATH /home/liumu/sourceLib/libtorch/share/cmake)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIBRARY_DIRS})

find_package(Boost COMPONENTS system context REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(TorchVision REQUIRED)
include_directories(${TORCHVISION_INCLUDE_DIRS})
link_directories(${TORCHVISION_LIBRARY_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

add_subdirectory("convx")
add_subdirectory("DLComputation")
add_subdirectory("linearNet")
add_subdirectory("matplot")
add_subdirectory("MLP")
add_subdirectory("modernConvx")
add_subdirectory("tensorOperator")

include_directories(.)

include(CTest)
enable_testing()

add_executable(dlcpp main.cpp)
target_link_libraries(dlcpp ${Boost_LIBRARIES})
target_link_libraries(dlcpp convx DLComputation linearNet matplotlibcpp MYMLP modernConvx tensorOperator)
target_link_libraries(dlcpp ${Python3_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

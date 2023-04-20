set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(Cap4 MYMLP)
add_library(${Cap4} SHARED mlp.cpp mlp.h overfit.h overfit.cpp dropout.h dropout.cpp)
target_link_libraries(${Cap4}
    ${TORCH_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})
target_link_libraries(${Cap4} ${OPENCV_LIBS} TorchVision::TorchVision matplotlibcpp)
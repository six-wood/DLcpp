set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(Cap6 modernConvx)
add_library(${Cap6} SHARED modernConvx.cpp modernConvx.h)
target_link_libraries(${Cap6}
    ${TORCH_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})
target_link_libraries(${Cap6} ${OPENCV_LIBS})
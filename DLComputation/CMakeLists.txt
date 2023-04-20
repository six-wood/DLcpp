set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(Cap4 DLComputation)
add_library(${Cap4} SHARED Computation.h Computation.cpp useGPU.h useGPU.cpp)
target_link_libraries(${Cap4}
    ${TORCH_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})
target_link_libraries(${Cap4} ${OPENCV_LIBS})
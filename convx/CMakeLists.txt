set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(Cap5 convx)
add_library(${Cap5} SHARED convx.cpp convx.h)
target_link_libraries(${Cap5}
    ${TORCH_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})
target_link_libraries(${Cap5} ${OPENCV_LIBS})
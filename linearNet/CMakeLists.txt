set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(Cap3 linearNet)
add_library(${Cap3} SHARED lineaRegression.h lineaRegression.cpp softMax.cpp softMax.h)
target_link_libraries(${Cap3}
    ${TORCH_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})
target_link_libraries(${Cap3} ${OPENCV_LIBS})
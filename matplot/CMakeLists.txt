set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(plot matplotlibcpp)
add_library(${plot} SHARED matplotlibcpp.h matplotlibcpp.cpp)
target_link_libraries(${plot} ${PYTHON_LIBRARIES} ${OPENCV_LIBRARIES})
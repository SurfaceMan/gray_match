cmake_minimum_required(VERSION 3.12)

project(match)
find_package(OpenCV REQUIRED)
add_executable(match 
main.cpp 
grayMatch.h 
grayMatch.cpp
)

target_include_directories(match PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(match ${OpenCV_LIBRARIES})
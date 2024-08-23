cmake_minimum_required(VERSION 3.12)

project(match)
find_package(OpenCV REQUIRED)
add_executable(${PROJECT_NAME}
    main.cpp
    grayMatch.h
    grayMatch.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /external:W0>
        $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-fPIC -fvisibility=hidden -Wall -Wextra -Wpedantic -Wmisleading-indentation -Wunused -Wuninitialized -Wshadow -Wconversion -Werror>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>>:/W4 /WX /external:W0>
)

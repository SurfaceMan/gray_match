cmake_minimum_required(VERSION 3.12)

project(match)
find_package(OpenCV REQUIRED)

#==============================================================
#library
#==============================================================
add_library(algo SHARED
    grayMatch.h
    grayMatch.cpp
    serialize.cpp
    privateType.h
    apiExport.h
)

target_include_directories(algo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(algo ${OpenCV_LIBRARIES})
target_compile_options(algo PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /external:W0>
        $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>: -mlsx -fPIC -fvisibility=hidden -Wall -Wextra -Wpedantic -Wmisleading-indentation -Wunused -Wuninitialized -Wshadow -Wconversion -Werror>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>>:/W4 /WX /external:W0>
)
target_compile_definitions(algo PUBLIC API_EXPORTS)

#==============================================================
#exe
#==============================================================
add_executable(${PROJECT_NAME}
    main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} algo)
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /external:W0>
        $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>: -fPIC -fvisibility=hidden -Wall -Wextra -Wpedantic -Wmisleading-indentation -Wunused -Wuninitialized -Wshadow -Wconversion -Werror>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>>:/W4 /WX /external:W0>
)

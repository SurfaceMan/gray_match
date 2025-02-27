cmake_minimum_required(VERSION 3.12)
project(match)

find_package(OpenCV REQUIRED)

option(ENABLE_OPENMP "enable openmp" OFF)
if(ENABLE_OPENMP)
        # find OpenMP
        find_package(OpenMP REQUIRED)

        if(OPENMP_FOUND)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        endif(OPENMP_FOUND)
endif(ENABLE_OPENMP)

add_executable(${PROJECT_NAME}
    main.cpp
    grayMatch.h
    grayMatch.cpp
    sum.h
    sum.cpp
    privateType.h
)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} $<$<BOOL:${OPENMP_FOUND}>:OpenMP::OpenMP_CXX>)
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /external:W0>
        $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-fPIC -fvisibility=hidden -Wall -Wextra -Wpedantic -Wmisleading-indentation -Wunused -Wuninitialized -Wshadow -Wconversion -Werror>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>>:/W4 /WX /external:W0>
        $<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},loongarch64>:-mlsx>
)
target_compile_definitions(${PROJECT_NAME} PRIVATE
        $<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},loongarch64>:CV_LSX>
)
target_compile_definitions(${PROJECT_NAME} PRIVATE IMG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/img")

message("Arch:${CMAKE_SYSTEM_PROCESSOR}")

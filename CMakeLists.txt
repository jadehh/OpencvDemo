cmake_minimum_required(VERSION 3.22.1)
set(DEMO_NAME "opencv_demo")
project(${DEMO_NAME})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


set(CMAKE_CXX_STANDARD 17)
SET(OPENCV_DIR "/usr/local/opencv" CACHE PATH "Location of libraries")

if (NOT DEFINED OPENCV_DIR OR OPENCV_DIR STREQUAL "")
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/usr/local/opencv")
endif ()
if (NOT WIN32)
    set(OPENCV_INCLUDE_DIR "${OPENCV_DIR}/include/opencv4")
    set(OPENCV_LIB_DIR "${OPENCV_DIR}/lib")
    set(OPENCV_LIBS
            opencv_calib3d
            opencv_core
            opencv_dnn
            opencv_features2d
            opencv_flann
            opencv_highgui
            opencv_imgcodecs
            opencv_imgproc
            opencv_ml
            opencv_objdetect
            opencv_photo
            opencv_stitching
            opencv_video
            opencv_videoio)
else ()
    set(OPENCV_INCLUDE_DIR "${OPENCV_DIR}/build/include/")
    set(OPENCV_LIB_DIR "${OPENCV_DIR}/build/x64/vc16/lib")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(OPENCV_LIBS opencv_world4110d)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(OPENCV_LIBS opencv_world4110)
    else()
        message(STATUS "当前构建类型: ${CMAKE_BUILD_TYPE}")
    endif()
endif ()
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${OPENCV_INCLUDE_DIR}
)
link_directories(
        ${OPENCV_LIB_DIR}
)
# 消除警告
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-deprecated-declarations)
endif ()
AUX_SOURCE_DIRECTORY(./src SRCS)
add_executable(${DEMO_NAME} ${SRCS} main.cpp)
if (NOT WIN32)
    target_link_libraries(
            ${DEMO_NAME}
            pthread
            ${OPENCV_LIBS}

    )
else ()
    target_link_libraries(
            ${DEMO_NAME}
            ${OPENCV_LIBS}
    )
endif ()
if (WIN32)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DIR}/build/x64/vc16/bin/opencv_world4110d.dll ./opencv_world4110d.dll
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DIR}/build/x64/vc16/bin/opencv_world4110.dll ./opencv_world4110.dll
        )
    endif()
endif ()

cmake_minimum_required (VERSION 3.12)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost REQUIRED COMPONENTS system)

find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)

if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost NOT FOUND")
endif()

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS}) 

add_executable (detector_server "detector_server.cpp" "utilities.cpp" "utilities.h" "box.cpp" "box.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET detector_server PROPERTY CXX_STANDARD 20)
endif()

target_link_libraries(detector_server ${Boost_LIBRARIES})
target_link_libraries(detector_server ${OpenCV_LIBS})
target_link_libraries(detector_server "${TORCH_LIBRARIES}") 

if (MSVC) 
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll") 
    add_custom_command(TARGET detector_server 
            POST_BUILD 
            COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            ${TORCH_DLLS} 
            $<TARGET_FILE_DIR:detector_server>) 
endif (MSVC) 


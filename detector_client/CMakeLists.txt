cmake_minimum_required (VERSION 3.12)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost 1.82.0 REQUIRED)

find_package(OpenCV REQUIRED) 

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable (detector_client "detector_client.cpp" "detector_client.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET detector_client PROPERTY CXX_STANDARD 20)
endif()

target_link_libraries(detector_client ${Boost_LIBRARIES})
target_link_libraries(detector_client ${OpenCV_LIBS}) 


cmake_minimum_required(VERSION 3.10)
project(collision)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找libigl
find_package(libigl CONFIG REQUIRED)

# 编译标志
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ".")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# 添加源文件
file(GLOB SRCFILES src/*.cpp)
add_executable(${PROJECT_NAME}_bin ${SRCFILES})

# 按照vcpkg的建议链接库
target_link_libraries(${PROJECT_NAME}_bin PRIVATE 
    igl::igl_core 
    igl_copyleft::igl_copyleft_cgal 
    igl_copyleft::igl_copyleft_core
)
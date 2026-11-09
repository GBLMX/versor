cmake_minimum_required(VERSION 3.30)

project(vsr_runtime_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(vsr_runtime vsr_runtime.cpp)
target_include_directories(vsr_runtime PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)
target_compile_features(vsr_runtime PRIVATE cxx_std_26)

# Keep the module next to the source-test script for a simple local workflow.
set_target_properties(vsr_runtime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

include(CTest)
if(BUILD_TESTING)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    add_test(NAME vsr_python_binding
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_binding.py)
    set_tests_properties(vsr_python_binding PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
endif()

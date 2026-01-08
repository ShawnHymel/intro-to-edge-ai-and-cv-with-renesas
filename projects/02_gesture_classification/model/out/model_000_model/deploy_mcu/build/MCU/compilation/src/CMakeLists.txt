cmake_minimum_required(VERSION 3.24)

project(ec-nn-lib VERSION 0.1)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "-Wall -Wextra -O3")

set(DEFAULT_BUILD_TYPE "Release")

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

if(DEFINED ENV{CMSIS_REPO_GIT_REPOSITORY_ENV})
    set(CMSIS_REPO_GIT_REPOSITORY_OVERRIDE "$ENV{CMSIS_REPO_GIT_REPOSITORY_ENV}" CACHE STRING "Override Git repository for CMSIS_REPO from environment variable" FORCE)
    message(STATUS "Overriding CMSIS_REPO Git repository using environment variable: $ENV{CMSIS_REPO_GIT_REPOSITORY_ENV}")
else()
    set(CMSIS_REPO_GIT_REPOSITORY_OVERRIDE https://github.com/ARM-software/CMSIS-NN.git CACHE STRING "")
endif()

if(DEFINED ENV{PYBIND11_REPO_GIT_REPOSITORY_ENV})
    set(PYBIND11_REPO_GIT_REPOSITORY_OVERRIDE "$ENV{PYBIND11_REPO_GIT_REPOSITORY_ENV}" CACHE STRING "Override Git repository for PYBIND11_REPO from environment variable" FORCE)
    message(STATUS "Overriding PYBIND11_REPO Git repository using environment variable: $ENV{PYBIND11_REPO_GIT_REPOSITORY_ENV}")
else()
     set(PYBIND11_REPO_GIT_REPOSITORY_OVERRIDE https://github.com/pybind/pybind11.git CACHE STRING "")
endif()

#==== cmsis-nn ===================================================================
FetchContent_Declare(
    cmsisnn
    GIT_REPOSITORY "${CMSIS_REPO_GIT_REPOSITORY_OVERRIDE}"
    GIT_TAG "v7.0.0"
    GIT_SHALLOW ON
    USES_TERMINAL_DOWNLOAD 1
    GIT_PROGRESS 1
    PATCH_COMMAND git am "${CMAKE_CURRENT_SOURCE_DIR}/cmsis-windows.patch"
)
FetchContent_MakeAvailable(cmsisnn)
FetchContent_GetProperties(cmsisnn SOURCE_DIR CMSISNNHeaders)
target_compile_definitions(cmsis-nn PRIVATE CMSIS_NN_USE_SINGLE_ROUNDING)
target_compile_definitions(cmsis-nn PRIVATE POSITION_INDEPENDENT_CODE=ON)

add_library(ec-nn-model
  compute_sub_0002.c
  kernel_library_utils.c
   kernel_library_int.c 
  
)
set_property(TARGET ec-nn-model PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ec-nn-model PRIVATE cmsis-nn)

add_executable(compare compare.cpp)
target_link_libraries(compare PRIVATE ec-nn-model)

if (${BUILD_PY_BINDINGS})
  FetchContent_Declare(
      pybind11
      GIT_REPOSITORY "${PYBIND11_REPO_GIT_REPOSITORY_OVERRIDE}"
      GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(py_compute python_bindings.cpp)
  target_link_libraries(py_compute PRIVATE ec-nn-model)
endif()


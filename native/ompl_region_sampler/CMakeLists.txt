cmake_minimum_required(VERSION 3.18)
project(hnuter_ompl_region_sampler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED OMPL_SOURCE_DIR)
    message(FATAL_ERROR "OMPL_SOURCE_DIR must point to the matching OMPL source tree")
endif()
if(NOT DEFINED OMPL_GENERATED_INCLUDE_DIR)
    message(FATAL_ERROR "OMPL_GENERATED_INCLUDE_DIR must contain ompl/config.h")
endif()
if(NOT DEFINED OMPL_LIBRARY)
    message(FATAL_ERROR "OMPL_LIBRARY must point to the Python wheel's libompl.so")
endif()
if(NOT DEFINED OMPL_CXX11_ABI)
    set(OMPL_CXX11_ABI 0)
endif()

# The wheel and this module must use the same libstdc++ string ABI.  This
# definition also has to reach nanobind-static, not only the extension target.
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${OMPL_CXX11_ABI})

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Eigen3 REQUIRED)
add_subdirectory("${OMPL_SOURCE_DIR}/external/nanobind" nanobind)

nanobind_add_module(_ompl_region_sampler STABLE_ABI region_sampler.cpp)
target_include_directories(_ompl_region_sampler PRIVATE
    "${OMPL_SOURCE_DIR}/src"
    "${OMPL_GENERATED_INCLUDE_DIR}"
)
target_link_libraries(_ompl_region_sampler PRIVATE "${OMPL_LIBRARY}" Eigen3::Eigen)
get_filename_component(OMPL_LIBRARY_DIR ${OMPL_LIBRARY} DIRECTORY)
set_target_properties(_ompl_region_sampler PROPERTIES
    BUILD_RPATH "${OMPL_LIBRARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

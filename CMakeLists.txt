cmake_minimum_required(VERSION 3.15)

cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0104 NEW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

project(Falcor
    DESCRIPTION "Falcor Realtime Rendering Framework"
    LANGUAGES CXX C
)

# -----------------------------------------------------------------------------
# Configuration options
# -----------------------------------------------------------------------------

# Rendering backend.
set(FALCOR_RENDER_BACKEND D3D12 CACHE STRING "Falcor rendering backend")
set_property(CACHE FALCOR_RENDER_BACKEND PROPERTY STRINGS D3D12 GFX-D3D12 GFX-VK)


# Header validation.
# If enabled, additional targets are generated to validate that headers are self sufficient.
set(FALCOR_VALIDATE_HEADERS OFF CACHE BOOL "Enable header validation")

# Precompiled headers.
# If enabled, precompiled headers are used to speed up compilation.
set(FALCOR_PRECOMPILED_HEADERS ON CACHE BOOL "Enable precompiled headers")

# Check if D3D12 is available.
if(FALCOR_RENDER_BACKEND STREQUAL "D3D12" OR FALCOR_RENDER_BACKEND STREQUAL "GFX-D3D12")
    set(FALCOR_HAS_D3D12 ON)
else()
    set(FALCOR_HAS_D3D12 OFF)
endif()

# Check if Vulkan is available.
if(FALCOR_RENDER_BACKEND STREQUAL "GFX-VK")
    set(FALCOR_HAS_VULKAN ON)
else()
    set(FALCOR_HAS_VULKAN OFF)
endif()

# -----------------------------------------------------------------------------
# Check platform
# -----------------------------------------------------------------------------

if(${CMAKE_SYSTEM_NAME} MATCHES "Window")
    set(FALCOR_PLATFORM "Windows")
    set(FALCOR_WINDOWS TRUE)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(FALCOR_PLATFORM "Linux")
    set(FALCOR_LINUX TRUE)
else()
    message(FATAL_ERROR "Unsupported platform!")
endif()
message(STATUS "Platform: ${FALCOR_PLATFORM}")

# -----------------------------------------------------------------------------
# git submodules
# -----------------------------------------------------------------------------

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/external/pybind11/CMakeLists.txt)
    message(STATUS "Updating git submodules ...")
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init
        COMMAND_ERROR_IS_FATAL ANY
    )
endif()

# -----------------------------------------------------------------------------
# Packman
# -----------------------------------------------------------------------------

# Falcor uses packman to pull binary dependencies. We need to pull the dependencies
# before CMake starts configuring the project as some of the configuration relies
# on these dependencies being available. We also add additional targets to pull
# the dependencies when the project is built such that they are updated automatically
# if the manifest files change.

if(FALCOR_WINDOWS)
    set(PACKMAN "${CMAKE_SOURCE_DIR}/tools/packman/packman.cmd")
    set(PACKMAN_PLATFORM "windows-x86_64")
elseif(FALCOR_LINUX)
    set(PACKMAN "${CMAKE_SOURCE_DIR}/tools/packman/packman")
    set(PACKMAN_PLATFORM "linux-x86_64")
endif()

# Pull dependencies at configure time.
message(STATUS "Updating packman dependencies ...")
execute_process(
    COMMAND ${PACKMAN} pull ${CMAKE_SOURCE_DIR}/dependencies.xml --platform ${PACKMAN_PLATFORM}
    COMMAND_ERROR_IS_FATAL ANY
)

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

# Setup workaround for re-running mt.exe when it fails due to a race condition with AV software.
include(mt-retry)

# On Linux, we build with RPATH set to $ORIGIN to make build relocatable.
if(FALCOR_LINUX)
    set(CMAKE_INSTALL_RPATH $ORIGIN)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()


# -----------------------------------------------------------------------------
# CUDA
# -----------------------------------------------------------------------------


# Set default CUDA architecture to 75 (Turing).
# This can be overwritten for individual targets using the CUDA_ARCHITECTURE property.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75-virtual)
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "Enabling CUDA support.")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(FALCOR_HAS_CUDA ON)
else()
    set(FALCOR_HAS_CUDA OFF)
endif()

# -----------------------------------------------------------------------------
# External dependencies
# -----------------------------------------------------------------------------

add_subdirectory(external)

message(STATUS "Feature flags:")
message(STATUS "FALCOR_HAS_D3D12: ${FALCOR_HAS_D3D12}")
message(STATUS "FALCOR_HAS_VULKAN: ${FALCOR_HAS_VULKAN}")
message(STATUS "FALCOR_HAS_NVAPI: ${FALCOR_HAS_NVAPI}")
message(STATUS "FALCOR_HAS_PIX: ${FALCOR_HAS_PIX}")
message(STATUS "FALCOR_HAS_CUDA: ${FALCOR_HAS_CUDA}")
message(STATUS "FALCOR_HAS_OPTIX: ${FALCOR_HAS_OPTIX}")
message(STATUS "FALCOR_HAS_D3D12_AGILITY_SDK: ${FALCOR_HAS_D3D12_AGILITY_SDK}")
message(STATUS "FALCOR_HAS_NRD: ${FALCOR_HAS_NRD}")
message(STATUS "FALCOR_HAS_DLSS: ${FALCOR_HAS_DLSS}")
message(STATUS "FALCOR_HAS_NV_USD: ${FALCOR_HAS_NV_USD}")
message(STATUS "FALCOR_HAS_MDL_SDK: ${FALCOR_HAS_MDL_SDK}")

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------

# Require builds to be outside of source tree.
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please use a build directory instead.")
endif()

# Enable folders (for Visual Studio).
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Setup available build configurations.
if(NOT SETUP_CONFIGURATION_TYPES)
    set(SETUP_CONFIGURATION_TYPES 1)

    if(CMAKE_CONFIGURATION_TYPES)
        # multi config generator
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
    else()
        # single config generator
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
    endif()
endif()

set(FALCOR_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(FALCOR_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_CONFIGURATION_TYPES)
    set(FALCOR_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>)
else()
    set(FALCOR_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY})
endif()

set(FALCOR_SHADER_OUTPUT_DIRECTORY ${FALCOR_OUTPUT_DIRECTORY}/Shaders)
set(FALCOR_DATA_OUTPUT_DIRECTORY ${FALCOR_OUTPUT_DIRECTORY}/Data)

# -----------------------------------------------------------------------------
# Packman dependencies
# -----------------------------------------------------------------------------

add_custom_target(packman_dependencies DEPENDS packman_dependencies_stamp)
set_target_properties(packman_dependencies PROPERTIES FOLDER "Misc")
add_custom_command(
    OUTPUT packman_dependencies_stamp
    COMMAND ${PACKMAN} pull ${CMAKE_SOURCE_DIR}/dependencies.xml --platform ${PACKMAN_PLATFORM}
    COMMAND ${CMAKE_COMMAND} -E touch packman_dependencies_stamp
    MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/dependencies.xml
    COMMENT "Updating packman dependencies"
)

# -----------------------------------------------------------------------------
# Shader file handling
# -----------------------------------------------------------------------------

# Regex for matching shader files.
set(SHADER_EXTENSION_REGEX "\.(slang|slangh|hlsli|hlsl)$")

# Setup build rules to copy all shaders of a target to the output directory.
# The specified output_dir is relative to the global shader output directory (FALCOR_SHADER_OUTPUT_DIRECTORY).
function(target_copy_shaders target output_dir)
    get_target_property(target_source_dir ${target} SOURCE_DIR)
    set(shader_output_dir ${FALCOR_SHADER_OUTPUT_DIRECTORY}/${output_dir})

    # Get list of all target sources.
    get_target_property(target_sources_ ${target} SOURCES)

    # Create custom commands for copying shader sources.
    foreach(file ${target_sources_})
        if(${file} MATCHES ${SHADER_EXTENSION_REGEX})
            if(IS_ABSOLUTE ${file})
                file(RELATIVE_PATH file ${target_source_dir} ${file})
            endif()

            set(src_file ${target_source_dir}/${file})
            set(dst_file ${shader_output_dir}/${file})

            add_custom_command(
                OUTPUT ${dst_file}
                COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${src_file} ${dst_file}
                MAIN_DEPENDENCY ${src_file}
                COMMENT "${target}: Copying shader ${file}"
            )
        endif()
    endforeach()
endfunction()

# -----------------------------------------------------------------------------
# Data file handling
# -----------------------------------------------------------------------------

# Setup a post-build rule to copy the data folder of a target to the output directory.
# TODO: Take an optional argument for specifying a sub directory.
macro(target_copy_data_folder target)
    get_target_property(target_source_dir ${target} SOURCE_DIR)
    set(target_output_dir ${FALCOR_DATA_OUTPUT_DIRECTORY})

    # Support both upper and lower case data folders.
    if(EXISTS ${target_source_dir}/Data)
        add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${target_source_dir}/Data ${target_output_dir})
    elseif(EXISTS ${target_source_dir}/data)
        add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${target_source_dir}/data ${target_output_dir})
    endif()
endmacro()

# -----------------------------------------------------------------------------
# Falcor applications
# -----------------------------------------------------------------------------

# Create a Falcor application and link the main Falcor library.
function(add_falcor_executable target)
    add_executable(${target})

    target_link_libraries(${target} PRIVATE Falcor)

    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY})

    set_target_properties(${target} PROPERTIES VS_DEBUGGER_ENVIRONMENT "FALCOR_DEVMODE=1")
endfunction()

# -----------------------------------------------------------------------------
# Renderpasses
# -----------------------------------------------------------------------------

# Global render pass target list.
set_property(GLOBAL PROPERTY FALCOR_RENDERPASS_TARGETS)

# Create a Falcor renderpass.
function(add_renderpass target)
    add_library(${target} SHARED)

    target_link_libraries(${target} PRIVATE Falcor)

    target_compile_definitions(${target} PRIVATE PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    set_target_properties(${target} PROPERTIES
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}
        LIBRARY_OUTPUT_DIRECTORY ${FALCOR_LIBRARY_OUTPUT_DIRECTORY}
    )

    # Add target to global renderpass target list.
    set_property(GLOBAL APPEND PROPERTY FALCOR_RENDERPASS_TARGETS ${target})
endfunction()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Helper function to create a source group for Visual Studio.
# This adds all the target's sources to a source group in the given folder.
function(target_source_group target folder)
    set_target_properties(${target} PROPERTIES FOLDER ${folder})
    get_target_property(target_source_dir ${target} SOURCE_DIR)
    get_target_property(target_sources_ ${target} SOURCES)
    source_group(TREE ${target_source_dir} FILES ${target_sources_} PREFIX "")
endfunction()

# -----------------------------------------------------------------------------
# Deploy dependencies
# -----------------------------------------------------------------------------

# We currently use the legacy batch files for deploying additional binaries.
# TODO: This should be replaced with a better approach allowing individual
# targets to deploy dependencies.

set(source_dir ${CMAKE_SOURCE_DIR})
set(output_dir ${FALCOR_OUTPUT_DIRECTORY})
file(TO_NATIVE_PATH ${source_dir} source_dir)
file(TO_NATIVE_PATH ${output_dir} output_dir)

if(FALCOR_WINDOWS)
    add_custom_target(deploy_dependencies ALL
        COMMAND ${CMAKE_SOURCE_DIR}/build_scripts/deploycommon.bat ${source_dir} ${output_dir} $<CONFIG>
    )
elseif(FALCOR_LINUX)
    add_custom_target(deploy_dependencies ALL
        COMMAND ${CMAKE_SOURCE_DIR}/build_scripts/deploycommon.sh ${source_dir} ${output_dir} $<CONFIG>
    )
endif()
set_target_properties(deploy_dependencies PROPERTIES FOLDER "Misc")


# -----------------------------------------------------------------------------
# Header validation
# -----------------------------------------------------------------------------

# Helper function to validate header files of a target to be self sufficient.
# This creates a new target with "ValidateHeaders" suffix that compiles one
# translation unit for each header file found in the supplied targets sources list.
function(validate_headers target)
    if(NOT FALCOR_VALIDATE_HEADERS)
        return()
    endif()

    message(STATUS "Setting up header validation for target: ${target}")

    # Create a new target for validating headers.
    set(validate_target ${target}ValidateHeaders)
    add_library(${validate_target})

    # Copy properties from original target.
    foreach(property INCLUDE_DIRECTORIES LINK_LIBRARIES COMPILE_DEFINITIONS COMPILE_FEATURES)
        get_target_property(value ${target} ${property})
        # message(STATUS "${property}: ${value}")
        if(value)
            set_property(TARGET ${validate_target} PROPERTY ${property} ${value})
        endif()
    endforeach()

    # Add target source directory as an include directory.
    get_target_property(target_source_dir ${target} SOURCE_DIR)
    target_include_directories(${validate_target} PRIVATE ${target_source_dir})

    # Get list of source files.
    get_target_property(sources ${target} SOURCES)

    # Create a list of CPP files, each including one header.
    set(validate_sources "")
    foreach(file ${sources})
        if(${file} MATCHES "^[^.][^.].*\\.h$")
            # message(STATUS ${file})
            string(MAKE_C_IDENTIFIER ${file} id)
            set(cpp_file ${validate_target}/${id}.cpp)
            set(cpp_content "#include \"${file}\"\n")
            file(CONFIGURE OUTPUT ${cpp_file} CONTENT ${cpp_content})
            list(APPEND validate_sources ${CMAKE_CURRENT_BINARY_DIR}/${cpp_file})
        endif()
    endforeach()

    target_sources(${validate_target} PRIVATE ${validate_sources})
    set_target_properties(${validate_target} PROPERTIES FOLDER "Validation")
endfunction()

# -----------------------------------------------------------------------------
# Ninja log analyzer
# -----------------------------------------------------------------------------

if(${CMAKE_GENERATOR} MATCHES "Ninja")
    add_custom_target(ninja_log_analyzer ALL
        COMMAND ${CMAKE_SOURCE_DIR}/tools/.packman/python/python ${CMAKE_SOURCE_DIR}/tools/ninja_log_analyzer.py ${CMAKE_CURRENT_BINARY_DIR}/.ninja_log -c 10
    )
    add_dependencies(ninja_log_analyzer Falcor FalcorTest Mogwai ${renderpass_targets})
    set_target_properties(ninja_log_analyzer PROPERTIES FOLDER "Misc")
endif()

# -----------------------------------------------------------------------------
# Project sources
# -----------------------------------------------------------------------------

add_subdirectory(Source/Falcor)
add_subdirectory(Source/Mogwai)
add_subdirectory(Source/RenderPasses)
add_subdirectory(Source/Samples)
add_subdirectory(Source/Tools)

# Make Falcor core library depend on deploying all dependencies.
add_dependencies(Falcor deploy_dependencies)

# Make Mogwai depend on all render passes to have them built prior to running Mogwai.
get_property(renderpass_targets GLOBAL PROPERTY FALCOR_RENDERPASS_TARGETS)
if(renderpass_targets)
    add_dependencies(Mogwai ${renderpass_targets})
endif()

# Make Mogwai the default startup project in VS.
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Mogwai)


# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------

# d3d12
add_library(d3d12 INTERFACE)
target_link_libraries(d3d12 INTERFACE dxgi.lib d3d12.lib)

# On Linux, we need GTK3.
if(FALCOR_LINUX)
    find_package(GTK3 REQUIRED)
    add_library(gtk3 INTERFACE)
    target_include_directories(gtk3 INTERFACE ${GTK3_INCLUDE_DIRS})
    target_link_libraries(gtk3 INTERFACE ${GTK3_LIBRARIES})
endif()

# -----------------------------------------------------------------------------
# Dependencies in git submodules
# -----------------------------------------------------------------------------

# args
add_library(args INTERFACE)
target_include_directories(args INTERFACE args)

# fmt
message(STATUS "Configure fmt")
add_subdirectory(fmt)
set_target_properties(fmt PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "Libraries"
)

# glfw
message(STATUS "Configure glfw")
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
add_subdirectory(glfw)

# glm
message(STATUS "Configure glm")
add_subdirectory(glm)

# imgui
add_library(imgui INTERFACE)
target_include_directories(imgui INTERFACE imgui)

# pybind11
message(STATUS "Configure pybind11")
set(PYBIND11_FINDPYTHON ON)
set(Python_ROOT_DIR ${CMAKE_SOURCE_DIR}/external/packman/python)
add_subdirectory(pybind11)


# -----------------------------------------------------------------------------
# Dependencies in local files
# -----------------------------------------------------------------------------

# mikktspace
add_library(mikktspace STATIC mikktspace/mikktspace.c)
target_include_directories(mikktspace PUBLIC mikktspace)
set_target_properties(mikktspace PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "Libraries"
)

# imgui_addons
add_library(imgui_addons INTERFACE)
target_include_directories(imgui_addons INTERFACE imgui_addons)

# external_includes
add_library(external_includes INTERFACE)
target_include_directories(external_includes INTERFACE include tiny-cuda-nn/dependencies)

# -----------------------------------------------------------------------------
# Dependencies in packman
# -----------------------------------------------------------------------------

set(PACKMAN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/packman)

set(FALCOR_DEPS_DIR ${PACKMAN_DIR}/deps)
if(FALCOR_WINDOWS)
    add_library(falcor_deps INTERFACE)
    target_include_directories(falcor_deps INTERFACE ${FALCOR_DEPS_DIR}/include)
    target_link_directories(falcor_deps INTERFACE
        $<$<CONFIG:Release>:${FALCOR_DEPS_DIR}/lib>
        $<$<CONFIG:Debug>:${FALCOR_DEPS_DIR}/debug/lib>
    )
    target_link_libraries(falcor_deps INTERFACE
        # FreeImage
        $<$<CONFIG:Release>:FreeImage.lib>
        $<$<CONFIG:Debug>:FreeImaged.lib>
        # ffmpeg
        avcodec.lib avutil.lib avformat.lib swscale.lib
        # assimp
        # Note: For better performance in Debug builds we always use the release version.
        ${FALCOR_DEPS_DIR}/lib/assimp-vc142-mt.lib
        # OpenEXR
        $<$<CONFIG:Release>:Half-2_5.lib>
        $<$<CONFIG:Debug>:Half-2_5_d.lib>
        # OpenVDB
        openvdb.lib
        # lz4
        # Note: For better performance in Debug builds we always use the release version.
        ${FALCOR_DEPS_DIR}/lib/lz4.lib
        # hdf5
        $<$<CONFIG:Release>:hdf5.lib>
        $<$<CONFIG:Debug>:hdf5_D.lib>
        # zlib
        $<$<CONFIG:Release>:zlib.lib>
        $<$<CONFIG:Debug>:zlibd.lib>
        # pugixml
        pugixml.lib
    )
elseif(FALCOR_LINUX)
    add_library(falcor_deps INTERFACE)
    target_include_directories(falcor_deps INTERFACE ${FALCOR_DEPS_DIR}/include)
    target_link_directories(falcor_deps INTERFACE
        $<$<CONFIG:Release>:${FALCOR_DEPS_DIR}/lib>
        $<$<CONFIG:Debug>:${FALCOR_DEPS_DIR}/debug/lib>
    )
    target_link_libraries(falcor_deps INTERFACE
        # FreeImage
        $<$<CONFIG:Release>:FreeImage.so>
        $<$<CONFIG:Debug>:FreeImaged.so>
        # ffmpeg
        avcodec.so avutil.so avformat.so swscale.so swresample.so
        # assimp
        # Note: For better performance in Debug builds we always use the release version.
        ${FALCOR_DEPS_DIR}/lib/libassimp.so
        # OpenEXR
        $<$<CONFIG:Release>:Imath-2_5.so>
        $<$<CONFIG:Debug>:Imath-2_5_d.so>
        $<$<CONFIG:Release>:Half-2_5.so>
        $<$<CONFIG:Debug>:Half-2_5_d.so>
        $<$<CONFIG:Release>:IlmThread-2_5.so>
        $<$<CONFIG:Debug>:IlmThread-2_5_d.so>
        $<$<CONFIG:Release>:IlmImf-2_5.so>
        $<$<CONFIG:Debug>:IlmImf-2_5_d.so>
        $<$<CONFIG:Release>:Iex-2_5.so>
        $<$<CONFIG:Debug>:Iex-2_5_d.so>
        # OpenVDB
        openvdb.so
        # lz4
        # Note: For better performance in Debug builds we always use the release version.
        ${FALCOR_DEPS_DIR}/lib/liblz4.a
        # hdf5
        $<$<CONFIG:Release>:hdf5.so>
        $<$<CONFIG:Debug>:hdf5_debug.so>
        # zlib
        $<$<CONFIG:Release>:libz.a>
        $<$<CONFIG:Debug>:libz.a>
        # pugixml
        $<$<CONFIG:Release>:pugixml.so>
        $<$<CONFIG:Debug>:pugixml.so>
        # tbb
        $<$<CONFIG:Release>:tbb.so>
        $<$<CONFIG:Debug>:${FALCOR_DEPS_DIR}/lib/libtbb.so>
    )
endif()

# python
set(PYTHON_DIR ${PACKMAN_DIR}/python)
if(FALCOR_WINDOWS)
    add_library(python SHARED IMPORTED GLOBAL)
    set_target_properties(python PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${PYTHON_DIR}/include
        IMPORTED_IMPLIB ${PYTHON_DIR}/libs/python37.lib
        IMPORTED_LOCATION ${PYTHON_DIR}/python37.dll
    )
elseif(FALCOR_LINUX)
    add_library(python SHARED IMPORTED GLOBAL)
    set_target_properties(python PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${PYTHON_DIR}/include
        IMPORTED_LOCATION ${PYTHON_DIR}/lib/libpython3.7m.so
    )
endif()

# nvapi
set(NVAPI_DIR ${PACKMAN_DIR}/nvapi)
if(EXISTS ${NVAPI_DIR}/nvapi.h)
    add_library(nvapi STATIC IMPORTED GLOBAL)
    set_target_properties(nvapi PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NVAPI_DIR}
        IMPORTED_LOCATION ${NVAPI_DIR}/amd64/nvapi64.lib
    )
    set(FALCOR_HAS_NVAPI ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_NVAPI OFF PARENT_SCOPE)
endif()

# slang
set(SLANG_DIR ${PACKMAN_DIR}/slang)
if(FALCOR_WINDOWS)
    add_library(slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_IMPLIB ${SLANG_DIR}/bin/windows-x64/release/slang.lib
        IMPORTED_LOCATION ${SLANG_DIR}/bin/windows-x64/release/slang.dll
    )
elseif(FALCOR_LINUX)
    add_library(slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_LOCATION ${SLANG_DIR}/bin/linux-x64/release/libslang.so
    )
endif()

# slang-gfx
if(FALCOR_WINDOWS)
    add_library(slang-gfx SHARED IMPORTED GLOBAL)
    set_target_properties(slang-gfx PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_IMPLIB ${SLANG_DIR}/bin/windows-x64/release/gfx.lib
        IMPORTED_LOCATION ${SLANG_DIR}/bin/windows-x64/release/gfx.dll
    )
elseif(FALCOR_LINUX)
    add_library(slang-gfx SHARED IMPORTED GLOBAL)
    set_target_properties(slang-gfx PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_LOCATION ${SLANG_DIR}/bin/linux-x64/release/libgfx.so
    )
endif()

# pix
set(PIX_DIR ${PACKMAN_DIR}/pix)
if(EXISTS ${PIX_DIR}/Include/WinPixEventRuntime/pix3.h)
    add_library(pix SHARED IMPORTED GLOBAL)
    set_target_properties(pix PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${PIX_DIR}/Include
        IMPORTED_IMPLIB ${PIX_DIR}/bin/x64/WinPixEventRuntime.lib
        IMPORTED_LOCATION ${PIX_DIR}/bin/x64/WinPixEventRuntime.dll
    )
    set(FALCOR_HAS_PIX ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_PIX OFF PARENT_SCOPE)
endif()

# nanovdb
set(NANOVDB_DIR ${PACKMAN_DIR}/nanovdb)
add_library(nanovdb INTERFACE)
target_include_directories(nanovdb INTERFACE ${NANOVDB_DIR}/include)

# rtxgi
set(RTXGI_DIR ${PACKMAN_DIR}/rtxgi)
add_library(rtxgi INTERFACE)
target_include_directories(rtxgi INTERFACE ${RTXGI_DIR}/rtxgi-sdk/include)

# nv-usd
set(NV_USD_RELEASE_DIR ${PACKMAN_DIR}/nv-usd-release)
set(NV_USD_DEBUG_DIR ${PACKMAN_DIR}/nv-usd-debug)
if((EXISTS ${NV_USD_RELEASE_DIR}) AND (EXISTS ${NV_USD_DEBUG_DIR}))
    add_library(nv-usd INTERFACE)
    target_include_directories(nv-usd INTERFACE
        $<$<CONFIG:Release>:${NV_USD_RELEASE_DIR}/include>
        $<$<CONFIG:Debug>:${NV_USD_DEBUG_DIR}/include>
    )
    target_link_directories(nv-usd INTERFACE
        $<$<CONFIG:Release>:${NV_USD_RELEASE_DIR}/lib ${NV_USD_RELEASE_DIR}/plugin/usd>
        $<$<CONFIG:Debug>:${NV_USD_DEBUG_DIR}/lib ${NV_USD_DEBUG_DIR}/plugin/usd>
    )
    target_link_libraries(nv-usd INTERFACE
        usd.lib sdf.lib usdGeom.lib usdImaging.lib;usdRender.lib usdSkel.lib vt.lib ar.lib work.lib hd.lib arch.lib tf.lib usdLux.lib usdShade.lib plug.lib usdMdl.lib gf.lib
        # boost_python
        $<$<CONFIG:Release>:boost_python37-vc141-mt-x64-1_68.lib>
        $<$<CONFIG:Debug>:boost_python37-vc141-mt-gd-x64-1_68.lib>
        # tbb
        $<$<CONFIG:Release>:tbb.lib>
        $<$<CONFIG:Debug>:tbb_debug.lib>
    )
    set(FALCOR_HAS_NV_USD ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_NV_USD OFF PARENT_SCOPE)
endif()

# mdl-sdk
set(MDL_SDK_DIR ${PACKMAN_DIR}/mdl-sdk)
if(EXISTS ${MDL_SDK_DIR})
    add_library(mdl-sdk INTERFACE)
    target_include_directories(mdl-sdk INTERFACE ${MDL_SDK_DIR}/include)
    set(FALCOR_HAS_MDL_SDK ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_MDL_SDK OFF PARENT_SCOPE)
endif()

# dxcompiler
set(DXCOMPILER_DIR ${PACKMAN_DIR}/dxcompiler)
add_library(dxcompiler SHARED IMPORTED GLOBAL)
set_target_properties(dxcompiler PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${DXCOMPILER_DIR}/inc
    IMPORTED_IMPLIB ${DXCOMPILER_DIR}/lib/x64/dxcompiler.lib
    IMPORTED_LOCATION ${DXCOMPILER_DIR}/bin/x64/dxcompiler.dll
)

# TODO: We should use the official DLSS package instead.
# ngx
set(NGX_DIR ${PACKMAN_DIR}/ngx)
if(EXISTS ${NGX_DIR}/include/nvsdk_ngx.h)
    add_library(ngx STATIC IMPORTED GLOBAL)
    set_target_properties(ngx PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NGX_DIR}/include
        IMPORTED_LOCATION ${NGX_DIR}/lib/Windows_x86_64/x86_64/nvsdk_ngx_d.lib
        IMPORTED_LOCATION_DEBUG ${NGX_DIR}/lib/Windows_x86_64/x86_64/nvsdk_ngx_d_dbg.lib
    )
    set(FALCOR_HAS_DLSS ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_DLSS OFF PARENT_SCOPE)
endif()

# nrd
set(NRD_DIR ${PACKMAN_DIR}/nrd)
if(EXISTS ${NRD_DIR}/Include/NRD.h)
    add_library(nrd SHARED IMPORTED GLOBAL)
    set_target_properties(nrd PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NRD_DIR}/Include
        IMPORTED_IMPLIB ${NRD_DIR}/Lib/Release/NRD.lib
        IMPORTED_LOCATION ${NRD_DIR}/Lib/Release/NRD.dll
        IMPORTED_IMPLIB_DEBUG ${NRD_DIR}/Lib/Debug/NRD.lib
        IMPORTED_LOCATION_DEBUG ${NRD_DIR}/Lib/Debug/NRD.dll
    )
    set(FALCOR_HAS_NRD ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_NRD OFF PARENT_SCOPE)
endif()

# optix
set(OPTIX_DIR ${PACKMAN_DIR}/optix)
if(EXISTS ${OPTIX_DIR}/include/optix.h)
    add_library(optix INTERFACE)
    target_include_directories(optix INTERFACE ${OPTIX_DIR}/include)
    set(FALCOR_HAS_OPTIX ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_OPTIX OFF PARENT_SCOPE)
endif()

# nvtt
set(NVTT_DIR ${PACKMAN_DIR}/nvtt)
if(FALCOR_WINDOWS)
    add_library(nvtt SHARED IMPORTED GLOBAL)
    set_target_properties(nvtt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NVTT_DIR}/include
        IMPORTED_IMPLIB ${NVTT_DIR}/lib/x64-v142/nvtt30106.lib
        IMPORTED_LOCATION ${NVTT_DIR}/nvtt30106.dll
    )
elseif(FALCOR_LINUX)
    add_library(nvtt SHARED IMPORTED GLOBAL)
    set_target_properties(nvtt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NVTT_DIR}/include
        IMPORTED_LOCATION ${NVTT_DIR}/libnvtt.so
        IMPORTED_NO_SONAME TRUE
    )
endif()

# agility-sdk
set(D3D12_AGILITY_SDK_DIR ${PACKMAN_DIR}/agility-sdk)
if(EXISTS ${D3D12_AGILITY_SDK_DIR}/build/native/include/d3d12.h)
    add_library(agility-sdk INTERFACE)
    target_include_directories(agility-sdk INTERFACE ${D3D12_AGILITY_SDK_DIR}/build/native/include)
    set(FALCOR_HAS_D3D12_AGILITY_SDK ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_D3D12_AGILITY_SDK OFF PARENT_SCOPE)
endif()

# rtxdi
set(RTXDI_DIR ${PACKMAN_DIR}/rtxdi)
add_library(rtxdi INTERFACE)
target_include_directories(rtxdi INTERFACE ${RTXDI_DIR}/rtxdi-sdk/include)


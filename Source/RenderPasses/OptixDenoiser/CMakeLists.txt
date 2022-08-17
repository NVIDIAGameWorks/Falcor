if(NOT (FALCOR_HAS_CUDA AND FALCOR_HAS_OPTIX))
    return()
endif()

add_renderpass(OptixDenoiser)

target_sources(OptixDenoiser PRIVATE
    BootstrapUtils.cpp
    ConvertBufToTex.ps.slang
    ConvertMotionVectorInputs.cs.slang
    ConvertNormalsToBuf.cs.slang
    ConvertTexToBuf.cs.slang
    CudaUtils.cpp
    CudaUtils.h
    OptixDenoiser.cpp
    OptixDenoiser.h
)

target_copy_shaders(OptixDenoiser RenderPasses/OptixDenoiser)

target_link_libraries(OptixDenoiser PRIVATE optix)

target_source_group(OptixDenoiser "RenderPasses")

if(NOT (FALCOR_HAS_CUDA AND FALCOR_HAS_OPTIX))
    return()
endif()

add_plugin(OptixDenoiser)

target_sources(OptixDenoiser PRIVATE
    ConvertBufToTex.ps.slang
    ConvertMotionVectorInputs.cs.slang
    ConvertNormalsToBuf.cs.slang
    ConvertTexToBuf.cs.slang
    OptixDenoiser.cpp
    OptixDenoiser.h
    OptixUtils.cpp
    OptixUtils.h
)

target_copy_shaders(OptixDenoiser RenderPasses/OptixDenoiser)

target_link_libraries(OptixDenoiser PRIVATE optix)

target_source_group(OptixDenoiser "RenderPasses")

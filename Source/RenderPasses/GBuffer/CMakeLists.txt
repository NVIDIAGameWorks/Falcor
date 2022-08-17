add_renderpass(GBuffer)

target_sources(GBuffer PRIVATE
    GBufferBase.cpp
    GBufferBase.h

    GBuffer/GBuffer.cpp
    GBuffer/GBuffer.h
    GBuffer/GBufferHelpers.slang
    GBuffer/GBufferRaster.3d.slang
    GBuffer/GBufferRaster.cpp
    GBuffer/GBufferRaster.h
    GBuffer/GBufferRT.cpp
    GBuffer/GBufferRT.cs.slang
    GBuffer/GBufferRT.h
    GBuffer/GBufferRT.rt.slang
    GBuffer/GBufferRT.slang

    VBuffer/VBufferRaster.3d.slang
    VBuffer/VBufferRaster.cpp
    VBuffer/VBufferRaster.h
    VBuffer/VBufferRT.cpp
    VBuffer/VBufferRT.cs.slang
    VBuffer/VBufferRT.h
    VBuffer/VBufferRT.rt.slang
    VBuffer/VBufferRT.slang
)

target_copy_shaders(GBuffer RenderPasses/GBuffer)

target_link_libraries(GBuffer PRIVATE DepthPass)

target_source_group(GBuffer "RenderPasses")

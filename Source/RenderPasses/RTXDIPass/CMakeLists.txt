add_renderpass(RTXDIPass)

target_sources(RTXDIPass PRIVATE
    FinalShading.cs.slang
    LoadShadingData.slang
    PrepareSurfaceData.cs.slang
    RTXDIPass.cpp
    RTXDIPass.h
)

target_copy_shaders(RTXDIPass RenderPasses/RTXDIPass)

target_source_group(RTXDIPass "RenderPasses")

add_renderpass(CSM)

target_sources(CSM PRIVATE
    CascadedShadowMap.slang
    CSM.cpp
    CSM.h
    CSMData.slang
    DepthPass.slang
    ShadowPass.slang
    VisibilityPass.ps.slang
)

target_copy_shaders(CSM RenderPasses/CSM)

target_link_libraries(CSM PRIVATE Utils)

target_source_group(CSM "RenderPasses")

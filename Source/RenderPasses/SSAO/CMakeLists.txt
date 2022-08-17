add_renderpass(SSAO)

target_sources(SSAO PRIVATE
    ApplyAO.ps.slang
    SSAO.cpp
    SSAO.h
    SSAO.ps.slang
    SSAOData.slang
)

target_copy_shaders(SSAO RenderPasses/SSAO)

target_link_libraries(SSAO PRIVATE Utils)

target_source_group(SSAO "RenderPasses")

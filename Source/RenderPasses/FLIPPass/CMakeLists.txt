add_renderpass(FLIPPass)

target_sources(FLIPPass PRIVATE
    ComputeLuminance.cs.slang
    flip.hlsli
    FLIPPass.cpp
    FLIPPass.cs.slang
    FLIPPass.h
    ToneMappers.slang
)

target_copy_shaders(FLIPPass RenderPasses/FLIPPass)

target_source_group(FLIPPass "RenderPasses")

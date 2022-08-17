add_renderpass(ToneMapper)

target_sources(ToneMapper PRIVATE
    Luminance.ps.slang
    ToneMapper.cpp
    ToneMapper.h
    ToneMapperParams.slang
    ToneMapping.ps.slang
)

target_copy_shaders(ToneMapper RenderPasses/ToneMapper)

target_source_group(ToneMapper "RenderPasses")

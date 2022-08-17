add_renderpass(SimplePostFX)

target_sources(SimplePostFX PRIVATE
    SimplePostFX.cpp
    SimplePostFX.cs.slang
    SimplePostFX.h
)

target_copy_shaders(SimplePostFX RenderPasses/SimplePostFX)

target_source_group(SimplePostFX "RenderPasses")

add_plugin(TAA)

target_sources(TAA PRIVATE
    TAA.cpp
    TAA.h
    TAA.ps.slang
)

target_copy_shaders(TAA RenderPasses/TAA)

target_source_group(TAA "RenderPasses")

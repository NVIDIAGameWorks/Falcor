add_renderpass(AccumulatePass)

target_sources(AccumulatePass PRIVATE
    Accumulate.cs.slang
    AccumulatePass.cpp
    AccumulatePass.h
)

target_copy_shaders(AccumulatePass RenderPasses/AccumulatePass)

target_source_group(AccumulatePass "RenderPasses")

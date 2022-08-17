add_renderpass(MinimalPathTracer)

target_sources(MinimalPathTracer PRIVATE
    MinimalPathTracer.cpp
    MinimalPathTracer.h
    MinimalPathTracer.rt.slang
)

target_copy_shaders(MinimalPathTracer RenderPasses/MinimalPathTracer)

target_source_group(MinimalPathTracer "RenderPasses")

add_plugin(AAPathTracer)

target_sources(AAPathTracer PRIVATE
    AAPathTracer.cpp
    AAPathTracer.h
    MinimalPathTracer.rt.slang
)

target_copy_shaders(AAPathTracer RenderPasses/AAPathTracer)

target_source_group(AAPathTracer "RenderPasses")

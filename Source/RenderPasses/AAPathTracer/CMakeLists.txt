add_plugin(AAPathTracer)

target_sources(AAPathTracer PRIVATE
    AAPathTracer.cpp
    AAPathTracer.h
    AAPathTracer.rt.slang
)

target_copy_shaders(AAPathTracer RenderPasses/AAPathTracer)

target_source_group(AAPathTracer "RenderPasses")

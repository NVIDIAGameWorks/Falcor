add_plugin(WARDiffPathTracer)

target_sources(WARDiffPathTracer PRIVATE
    Params.slang
    PTUtils.slang
    StaticParams.slang
    WARDiffPathTracer.cpp
    WARDiffPathTracer.h
    WARDiffPathTracer.rt.slang
    WarpedAreaReparam.slang
)

target_copy_shaders(WARDiffPathTracer RenderPasses/WARDiffPathTracer)

target_source_group(WARDiffPathTracer "RenderPasses")

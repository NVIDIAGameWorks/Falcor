add_plugin(PathTracer)

target_sources(PathTracer PRIVATE
    ColorType.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LoadShadingData.slang
    NRDHelpers.slang
    Params.slang
    PathState.slang
    PathTracer.slang
    PathTracer.cpp
    PathTracer.h
    PathTracerNRD.slang
    PathTracerUtils.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    ResolveReSTIRPTPass.cs.slang
    ReSTIRPTData.slang
    LightTileRISHelper.slang
    StaticParams.slang
    TracePass.rt.slang
    TracePassMinimalPayload.rt.slang
)

target_copy_shaders(PathTracer RenderPasses/PathTracer)

target_source_group(PathTracer "RenderPasses")

target_link_libraries(PathTracer
    PRIVATE
        ReSTIRPathTracing
        LightTileResampling
)

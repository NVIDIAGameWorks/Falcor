add_renderpass(WhittedRayTracer)

target_sources(WhittedRayTracer PRIVATE
    WhittedRayTracer.cpp
    WhittedRayTracer.h
    WhittedRayTracer.rt.slang
    WhittedRayTracerTypes.slang
)

target_copy_shaders(WhittedRayTracer RenderPasses/WhittedRayTracer)
target_copy_data_folder(WhittedRayTracer RenderPasses/WhittedRayTracer)

target_source_group(WhittedRayTracer "RenderPasses")

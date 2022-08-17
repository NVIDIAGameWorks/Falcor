add_renderpass(DepthPass)

target_sources(DepthPass PRIVATE
    DepthPass.cpp
    DepthPass.h
    DepthPass.3d.slang
)

target_copy_shaders(DepthPass RenderPasses/DepthPass)

# Export DepthPass symbols for other render passes to link to.
target_compile_definitions(DepthPass PRIVATE BUILD_DEPTH_PASS)

target_source_group(DepthPass "RenderPasses")

add_plugin(WireframePass)

target_sources(WireframePass PRIVATE
    WireframePass.3d.slang
    WireframePass.cpp
    WireframePass.h
)

target_copy_shaders(WireframePass RenderPasses/WireframePass)

target_source_group(WireframePass "RenderPasses")

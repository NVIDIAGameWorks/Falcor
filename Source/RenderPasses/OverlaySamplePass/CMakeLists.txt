add_plugin(OverlaySamplePass)

target_sources(OverlaySamplePass PRIVATE
    OverlaySamplePass.cpp
    OverlaySamplePass.h
)

target_copy_shaders(OverlaySamplePass RenderPasses/OverlaySamplePass)

target_source_group(OverlaySamplePass "RenderPasses")

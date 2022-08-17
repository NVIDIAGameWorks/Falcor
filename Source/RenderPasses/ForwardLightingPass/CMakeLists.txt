add_renderpass(ForwardLightingPass)

target_sources(ForwardLightingPass PRIVATE
    ForwardLightingPass.cpp
    ForwardLightingPass.h
    ForwardLightingPass.3d.slang
)

target_copy_shaders(ForwardLightingPass RenderPasses/ForwardLightingPass)

target_source_group(ForwardLightingPass "RenderPasses")

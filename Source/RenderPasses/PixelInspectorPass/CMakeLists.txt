add_plugin(PixelInspectorPass)

target_sources(PixelInspectorPass PRIVATE
    PixelInspector.cs.slang
    PixelInspectorData.slang
    PixelInspectorPass.cpp
    PixelInspectorPass.h
)

target_copy_shaders(PixelInspectorPass RenderPasses/PixelInspectorPass)

target_source_group(PixelInspectorPass "RenderPasses")

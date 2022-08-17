add_renderpass(SceneDebugger)

target_sources(SceneDebugger PRIVATE
    SceneDebugger.cpp
    SceneDebugger.cs.slang
    SceneDebugger.h
    SharedTypes.slang
)

target_copy_shaders(SceneDebugger RenderPasses/SceneDebugger)

target_source_group(SceneDebugger "RenderPasses")

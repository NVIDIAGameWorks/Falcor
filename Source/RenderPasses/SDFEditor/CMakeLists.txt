add_renderpass(SDFEditor)

target_sources(SDFEditor PRIVATE
    GUIPass.ps.slang
    Marker2DSet.cpp
    Marker2DSet.h
    Marker2DSet.slang
    Marker2DTypes.slang
    SDFEditor.cpp
    SDFEditor.h
    SDFEditorTypes.slang
    SelectionWheel.cpp
    SelectionWheel.h
)

target_copy_shaders(SDFEditor RenderPasses/SDFEditor)

target_source_group(SDFEditor "RenderPasses")

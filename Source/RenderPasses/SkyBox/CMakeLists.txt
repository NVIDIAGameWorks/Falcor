add_renderpass(SkyBox)

target_sources(SkyBox PRIVATE
    SkyBox.cpp
    SkyBox.h
    SkyBox.3d.slang
)

target_copy_shaders(SkyBox RenderPasses/SkyBox)
target_copy_data_folder(SkyBox RenderPasses/SkyBox)

target_source_group(SkyBox "RenderPasses")

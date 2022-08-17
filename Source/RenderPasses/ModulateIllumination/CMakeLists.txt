add_renderpass(ModulateIllumination)

target_sources(ModulateIllumination PRIVATE
    ModulateIllumination.cpp
    ModulateIllumination.h
    ModulateIllumination.cs.slang
)

target_copy_shaders(ModulateIllumination RenderPasses/ModulateIllumination)

target_source_group(ModulateIllumination "RenderPasses")

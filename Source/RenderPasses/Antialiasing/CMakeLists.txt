add_renderpass(Antialiasing)

target_sources(Antialiasing PRIVATE
    Antialiasing.cpp

    FXAA/FXAA.cpp
    FXAA/FXAA.h
    FXAA/FXAA.slang

    TAA/TAA.cpp
    TAA/TAA.h
    TAA/TAA.ps.slang
)

target_copy_shaders(Antialiasing RenderPasses/Antialiasing)

target_source_group(Antialiasing "RenderPasses")

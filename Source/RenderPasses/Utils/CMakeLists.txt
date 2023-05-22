add_plugin(Utils)

target_sources(Utils PRIVATE
    Utils.cpp

    Composite/Composite.cpp
    Composite/Composite.cs.slang
    Composite/Composite.h
    Composite/CompositeMode.slangh

    CrossFade/CrossFade.cpp
    CrossFade/CrossFade.cs.slang
    CrossFade/CrossFade.h

    GaussianBlur/GaussianBlur.cpp
    GaussianBlur/GaussianBlur.h
    GaussianBlur/GaussianBlur.ps.slang
)

target_copy_shaders(Utils RenderPasses/Utils)

target_source_group(Utils "RenderPasses")

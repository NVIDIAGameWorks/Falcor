add_renderpass(SVGFPass)

target_sources(SVGFPass PRIVATE
    SVGFAtrous.ps.slang
    SVGFCommon.slang
    SVGFFilterMoments.ps.slang
    SVGFFinalModulate.ps.slang
    SVGFPackLinearZAndNormal.ps.slang
    SVGFPass.cpp
    SVGFPass.h
    SVGFReproject.ps.slang
)

target_copy_shaders(SVGFPass RenderPasses/SVGFPass)

target_source_group(SVGFPass "RenderPasses")

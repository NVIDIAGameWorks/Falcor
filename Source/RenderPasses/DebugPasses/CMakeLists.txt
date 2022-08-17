add_renderpass(DebugPasses)

target_sources(DebugPasses PRIVATE
    Comparison.ps.slang
    ComparisonPass.cpp
    ComparisonPass.h
    DebugPasses.cpp

    ColorMapPass/ColorMapParams.slang
    ColorMapPass/ColorMapPass.cpp
    ColorMapPass/ColorMapPass.h
    ColorMapPass/ColorMapPass.ps.slang

    InvalidPixelDetectionPass/InvalidPixelDetection.ps.slang
    InvalidPixelDetectionPass/InvalidPixelDetectionPass.cpp
    InvalidPixelDetectionPass/InvalidPixelDetectionPass.h

    SideBySidePass/SideBySide.ps.slang
    SideBySidePass/SideBySidePass.cpp
    SideBySidePass/SideBySidePass.h

    SplitScreenPass/SplitScreen.ps.slang
    SplitScreenPass/SplitScreenPass.cpp
    SplitScreenPass/SplitScreenPass.h
)

target_copy_shaders(DebugPasses RenderPasses/DebugPasses)

target_source_group(DebugPasses "RenderPasses")

add_plugin(ErrorMeasurePass)

target_sources(ErrorMeasurePass PRIVATE
    ErrorMeasurePass.cpp
    ErrorMeasurePass.h
    ErrorMeasurer.cs.slang
)

target_copy_shaders(ErrorMeasurePass RenderPasses/ErrorMeasurePass)

target_source_group(ErrorMeasurePass "RenderPasses")

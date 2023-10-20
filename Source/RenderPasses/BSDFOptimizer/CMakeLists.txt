add_plugin(BSDFOptimizer)

target_sources(BSDFOptimizer PRIVATE
    BSDFOptimizer.cpp
    BSDFOptimizer.cs.slang
    BSDFOptimizer.h
    BSDFOptimizerHelpers.slang
    BSDFOptimizerParams.slang
    BSDFViewer.cs.slang
)

target_copy_shaders(BSDFOptimizer RenderPasses/BSDFOptimizer)

target_source_group(BSDFOptimizer "RenderPasses")

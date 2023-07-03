add_falcor_executable(MultiSampling)

target_sources(MultiSampling PRIVATE
    MultiSampling.cpp
    MultiSampling.h
    MultiSampling.3d.slang
)

target_copy_shaders(MultiSampling Samples/MultiSampling)

target_source_group(MultiSampling "Samples")

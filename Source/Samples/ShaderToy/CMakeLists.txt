add_falcor_executable(ShaderToy)

target_sources(ShaderToy PRIVATE
    ShaderToy.cpp
    ShaderToy.h
    Toy.ps.slang
)

target_copy_shaders(ShaderToy Samples/ShaderToy)

target_source_group(ShaderToy "Samples")

add_falcor_executable(Visualization2D)

target_sources(Visualization2D PRIVATE
    Visualization2D.cpp
    Visualization2D.h
    Visualization2d.ps.slang
    VoxelNormals.ps.slang
)

target_copy_shaders(Visualization2D Samples/Visualization2D)

target_source_group(Visualization2D "Samples")

add_falcor_executable(ProjectTemplate)

target_sources(ProjectTemplate PRIVATE
    ProjectTemplate.cpp
    ProjectTemplate.h
)

target_copy_shaders(ProjectTemplate Samples/ProjectTemplate)

target_source_group(ProjectTemplate "Samples")

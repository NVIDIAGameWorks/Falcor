add_falcor_executable(SampleAppTemplate)

target_sources(SampleAppTemplate PRIVATE
    SampleAppTemplate.cpp
    SampleAppTemplate.h
)

target_copy_shaders(SampleAppTemplate Samples/SampleAppTemplate)

target_source_group(SampleAppTemplate "Samples")

add_falcor_executable(ReSTIR_FG)

target_sources(ReSTIR_FG PRIVATE
    ReSTIR_FG.cpp
    ReSTIR_FG.h
)

target_copy_shaders(ReSTIR_FG Samples/ReSTIR_FG)

target_source_group(ReSTIR_FG "Samples")

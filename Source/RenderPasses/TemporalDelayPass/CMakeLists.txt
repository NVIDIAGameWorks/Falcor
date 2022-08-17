add_renderpass(TemporalDelayPass)

target_sources(TemporalDelayPass PRIVATE
    TemporalDelayPass.cpp
    TemporalDelayPass.h
)

target_source_group(TemporalDelayPass "RenderPasses")

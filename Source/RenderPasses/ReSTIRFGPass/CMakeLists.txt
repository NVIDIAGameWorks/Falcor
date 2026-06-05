add_plugin(ReSTIRFGPass)

target_sources(ReSTIRFGPass PRIVATE
    ReSTIRFGPass.cpp
    ReSTIRFGPass.h

    EvaluateReservoirs.cs.slang
    ResampleReservoirFG.cs.slang
    ResampleReservoirCaustic.cs.slang
    StructsAndHelpers.slang
    GenerateInitialSamples.rt.slang
    TracePhotons.rt.slang
)

target_copy_shaders(ReSTIRFGPass RenderPasses/ReSTIRFGPass)
target_source_group(ReSTIRFGPass "RenderPasses")

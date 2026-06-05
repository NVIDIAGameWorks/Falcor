add_plugin(ReSTIR_FPDG)

target_sources(ReSTIR_FPDG PRIVATE
    ReSTIR_FPDG.cpp
    ReSTIR_FPDG.h

    EvaluateReservoirs.cs.slang
    ResampleReservoirFG.cs.slang
    ResamplePhotonDifferentialsReservoirCaustic.cs.slang
    StructsAndHelpers.slang
    PhotonDifferentials.slang
    GenerateInitialSamples.rt.slang
    TracePass.rt.slang
)

target_copy_shaders(ReSTIR_FPDG RenderPasses/ReSTIR_FPDG)
target_source_group(ReSTIR_FPDG "RenderPasses")

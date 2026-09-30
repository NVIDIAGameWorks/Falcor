add_falcor_executable(Caustics)

target_sources(Caustics PRIVATE
    	Caustics.cpp
    	Caustics.h
    	AllocateTilePhoton.cs.hlsl
    	AnalyseTraceResult.cs.hlsl
    	CausticsRaster.ps.hlsl
    	CausticsRT.rt.hlsl
    	Common.hlsl
    	CompositeRaster.ps.hlsl
    	CompositeRT.rt.hlsl
    	GenerateRayCountMipmap.cs.hlsl
    	GBufferRaster.ps.hlsl
    	PhotonGather.cs.hlsl
	    PhotonScatter.3d.hlsl
	    PhotonTrace.rt.hlsl
	    ResetDrawArgument.cs.hlsl
	    SmoothPhoton.cs.hlsl
	    SpacialFilter.cs.hlsl
	    TemporalFilter.cs.hlsl
	    UpdateRayDensity.cs.hlsl
)

target_copy_shaders(Caustics Samples/Caustics)

target_source_group(Caustics "Samples")

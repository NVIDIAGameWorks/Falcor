add_library(Falcor SHARED)

target_sources(Falcor PRIVATE
    Falcor.natvis
    Falcor.manifest

    Falcor.h

    Core/Assert.h
    Core/ErrorHandling.cpp
    Core/ErrorHandling.h
    Core/Errors.cpp
    Core/Errors.h
    Core/FalcorConfig.h
    Core/GLFW.h
    Core/HotReloadFlags.h
    Core/Macros.h
    Core/Renderer.h
    Core/Sample.cpp
    Core/Sample.h
    Core/Window.cpp
    Core/Window.h

    Core/API/BlendState.cpp
    Core/API/BlendState.h
    Core/API/BlitContext.cpp
    Core/API/BlitContext.h
    Core/API/BlitReduction.3d.slang
    Core/API/Buffer.cpp
    Core/API/Buffer.h
    Core/API/Common.cpp
    Core/API/Common.h
    Core/API/ComputeContext.cpp
    Core/API/ComputeContext.h
    Core/API/ComputeStateObject.cpp
    Core/API/ComputeStateObject.h
    Core/API/CopyContext.cpp
    Core/API/CopyContext.h
    Core/API/DepthStencilState.cpp
    Core/API/DepthStencilState.h
    Core/API/Device.cpp
    Core/API/Device.h
    Core/API/DXGIFormats.h
    Core/API/FBO.cpp
    Core/API/FBO.h
    Core/API/FencedPool.h
    Core/API/Formats.cpp
    Core/API/Formats.h
    Core/API/GpuFence.h
    Core/API/GpuMemoryHeap.cpp
    Core/API/GpuMemoryHeap.h
    Core/API/GpuTimer.cpp
    Core/API/GpuTimer.h
    Core/API/GraphicsStateObject.cpp
    Core/API/GraphicsStateObject.h
    Core/API/LowLevelContextData.h
    Core/API/ParameterBlock.cpp
    Core/API/ParameterBlock.h
    Core/API/QueryHeap.h
    Core/API/RasterizerState.cpp
    Core/API/RasterizerState.h
    Core/API/Raytracing.h
    Core/API/RenderContext.cpp
    Core/API/RenderContext.h
    Core/API/Resource.cpp
    Core/API/Resource.h
    Core/API/ResourceViews.cpp
    Core/API/ResourceViews.h
    Core/API/RtAccelerationStructure.cpp
    Core/API/RtAccelerationStructure.h
    Core/API/RtAccelerationStructurePostBuildInfoPool.cpp
    Core/API/RtAccelerationStructurePostBuildInfoPool.h
    Core/API/RtStateObject.cpp
    Core/API/RtStateObject.h
    Core/API/Sampler.cpp
    Core/API/Sampler.h
    Core/API/Shader.h
    Core/API/ShaderResourceType.h
    Core/API/ShaderTable.h
    Core/API/Texture.cpp
    Core/API/Texture.h
    Core/API/VAO.cpp
    Core/API/VAO.h
    Core/API/VertexLayout.cpp
    Core/API/VertexLayout.h

    Core/BufferTypes/VariablesBufferUI.cpp
    Core/BufferTypes/VariablesBufferUI.h

    Core/Platform/MonitorInfo.cpp
    Core/Platform/MonitorInfo.h
    Core/Platform/OS.cpp
    Core/Platform/OS.h
    Core/Platform/PlatformHandles.h
    Core/Platform/ProgressBar.cpp
    Core/Platform/ProgressBar.h

    Core/Program/ComputeProgram.cpp
    Core/Program/ComputeProgram.h
    Core/Program/CUDAProgram.cpp
    Core/Program/CUDAProgram.h
    Core/Program/GraphicsProgram.cpp
    Core/Program/GraphicsProgram.h
    Core/Program/Program.cpp
    Core/Program/Program.h
    Core/Program/ProgramReflection.cpp
    Core/Program/ProgramReflection.h
    Core/Program/ProgramVars.cpp
    Core/Program/ProgramVars.h
    Core/Program/ProgramVersion.cpp
    Core/Program/ProgramVersion.h
    Core/Program/RtBindingTable.cpp
    Core/Program/RtBindingTable.h
    Core/Program/RtProgram.cpp
    Core/Program/RtProgram.h
    Core/Program/ShaderVar.cpp
    Core/Program/ShaderVar.h

    Core/State/ComputeState.cpp
    Core/State/ComputeState.h
    Core/State/GraphicsState.cpp
    Core/State/GraphicsState.h
    Core/State/StateGraph.h

    RenderGraph/RenderGraph.cpp
    RenderGraph/RenderGraph.h
    RenderGraph/RenderGraphCompiler.cpp
    RenderGraph/RenderGraphCompiler.h
    RenderGraph/RenderGraphExe.cpp
    RenderGraph/RenderGraphExe.h
    RenderGraph/RenderGraphImportExport.cpp
    RenderGraph/RenderGraphImportExport.h
    RenderGraph/RenderGraphIR.cpp
    RenderGraph/RenderGraphIR.h
    RenderGraph/RenderGraphUI.cpp
    RenderGraph/RenderGraphUI.h
    RenderGraph/RenderPass.cpp
    RenderGraph/RenderPass.h
    RenderGraph/RenderPassHelpers.cpp
    RenderGraph/RenderPassHelpers.h
    RenderGraph/RenderPassLibrary.cpp
    RenderGraph/RenderPassLibrary.h
    RenderGraph/RenderPassReflection.cpp
    RenderGraph/RenderPassReflection.h
    RenderGraph/RenderPassStandardFlags.h
    RenderGraph/ResourceCache.cpp
    RenderGraph/ResourceCache.h

    RenderGraph/BasePasses/BaseGraphicsPass.cpp
    RenderGraph/BasePasses/BaseGraphicsPass.h
    RenderGraph/BasePasses/ComputePass.cpp
    RenderGraph/BasePasses/ComputePass.h
    RenderGraph/BasePasses/FullScreenPass.cpp
    RenderGraph/BasePasses/FullScreenPass.gs.slang
    RenderGraph/BasePasses/FullScreenPass.h
    RenderGraph/BasePasses/FullScreenPass.vs.slang
    RenderGraph/BasePasses/RasterPass.cpp
    RenderGraph/BasePasses/RasterPass.h
    RenderGraph/BasePasses/RasterScenePass.cpp
    RenderGraph/BasePasses/RasterScenePass.h

    Rendering/Lights/EmissiveLightSampler.cpp
    Rendering/Lights/EmissiveLightSampler.h
    Rendering/Lights/EmissiveLightSampler.slang
    Rendering/Lights/EmissiveLightSamplerHelpers.slang
    Rendering/Lights/EmissiveLightSamplerInterface.slang
    Rendering/Lights/EmissiveLightSamplerType.slangh
    Rendering/Lights/EmissivePowerSampler.cpp
    Rendering/Lights/EmissivePowerSampler.h
    Rendering/Lights/EmissivePowerSampler.slang
    Rendering/Lights/EmissiveUniformSampler.cpp
    Rendering/Lights/EmissiveUniformSampler.h
    Rendering/Lights/EmissiveUniformSampler.slang
    Rendering/Lights/EnvMapIntegration.ps.slang
    Rendering/Lights/EnvMapLighting.cpp
    Rendering/Lights/EnvMapLighting.h
    Rendering/Lights/EnvMapLighting.slang
    Rendering/Lights/EnvMapSampler.cpp
    Rendering/Lights/EnvMapSampler.h
    Rendering/Lights/EnvMapSampler.slang
    Rendering/Lights/EnvMapSamplerSetup.cs.slang
    Rendering/Lights/LightBVH.cpp
    Rendering/Lights/LightBVH.h
    Rendering/Lights/LightBVH.slang
    Rendering/Lights/LightBVHBuilder.cpp
    Rendering/Lights/LightBVHBuilder.h
    Rendering/Lights/LightBVHRefit.cs.slang
    Rendering/Lights/LightBVHSampler.cpp
    Rendering/Lights/LightBVHSampler.h
    Rendering/Lights/LightBVHSampler.slang
    Rendering/Lights/LightBVHSamplerSharedDefinitions.slang
    Rendering/Lights/LightBVHTypes.slang
    Rendering/Lights/LightHelpers.slang

    Rendering/Materials/AnisotropicGGX.slang
    Rendering/Materials/BCSDFConfig.slangh
    Rendering/Materials/BSDFIntegrator.cpp
    Rendering/Materials/BSDFIntegrator.cs.slang
    Rendering/Materials/BSDFIntegrator.h
    Rendering/Materials/BxDF.slang
    Rendering/Materials/BxDFConfig.slangh
    Rendering/Materials/ClothBRDF.slang
    Rendering/Materials/ClothBSDF.slang
    Rendering/Materials/ClothMaterial.slang
    Rendering/Materials/Fresnel.slang
    Rendering/Materials/HairBSDF.slang
    Rendering/Materials/HairChiang16.slang
    Rendering/Materials/HairMaterial.slang
    Rendering/Materials/IBSDF.slang
    Rendering/Materials/IBxDF.slang
    Rendering/Materials/IMaterial.slang
    Rendering/Materials/InteriorList.slang
    Rendering/Materials/InteriorListHelpers.slang
    Rendering/Materials/LayeredBxDF.slang
    Rendering/Materials/LobeType.slang
    Rendering/Materials/MERLBSDF.slang
    Rendering/Materials/MERLMaterial.slang
    Rendering/Materials/Microfacet.slang
    Rendering/Materials/RGLAcquisition.cpp
    Rendering/Materials/RGLAcquisition.cs.slang
    Rendering/Materials/RGLAcquisition.h
    Rendering/Materials/RGLBSDF.slang
    Rendering/Materials/RGLCommon.slang
    Rendering/Materials/RGLMaterial.slang
    Rendering/Materials/StandardBSDF.slang
    Rendering/Materials/StandardMaterial.slang
    Rendering/Materials/TexLODHelpers.slang
    Rendering/Materials/TexLODTypes.cpp
    Rendering/Materials/TexLODTypes.slang

    Rendering/Materials/PBRT/PBRTDiffuseMaterial.slang
    Rendering/Materials/PBRT/PBRTDiffuseBSDF.slang
    Rendering/Materials/PBRT/PBRTDiffuseTransmissionMaterial.slang
    Rendering/Materials/PBRT/PBRTDiffuseTransmissionBSDF.slang
    Rendering/Materials/PBRT/PBRTConductorMaterial.slang
    Rendering/Materials/PBRT/PBRTConductorBSDF.slang
    Rendering/Materials/PBRT/PBRTDielectricMaterial.slang
    Rendering/Materials/PBRT/PBRTDielectricBSDF.slang
    Rendering/Materials/PBRT/PBRTCoatedConductorMaterial.slang
    Rendering/Materials/PBRT/PBRTCoatedConductorBSDF.slang
    Rendering/Materials/PBRT/PBRTCoatedDiffuseMaterial.slang
    Rendering/Materials/PBRT/PBRTCoatedDiffuseBSDF.slang

    Rendering/RTXDI/EnvLightUpdater.cs.slang
    Rendering/RTXDI/LightUpdater.cs.slang
    Rendering/RTXDI/PackedTypes.slang
    Rendering/RTXDI/PolymorphicLight.slang
    Rendering/RTXDI/ReflectTypes.cs.slang
    Rendering/RTXDI/RTXDI.cpp
    Rendering/RTXDI/RTXDI.h
    Rendering/RTXDI/RTXDI.slang
    Rendering/RTXDI/RTXDIApplicationBridge.slangh
    Rendering/RTXDI/RTXDISDK.cpp
    Rendering/RTXDI/RTXDISetup.cs.slang
    Rendering/RTXDI/SurfaceData.slang

    Rendering/RTXGI/DirectLighting.slang
    Rendering/RTXGI/RTXGIDefines.slangh
    Rendering/RTXGI/RTXGISDK.cpp
    Rendering/RTXGI/RTXGIVolume.cpp
    Rendering/RTXGI/RTXGIVolume.h
    Rendering/RTXGI/RTXGIVolume.slang
    Rendering/RTXGI/UpdateProbes.rt.slang
    Rendering/RTXGI/UpdateProbesDebugData.slang

    Rendering/Utils/PixelStats.cpp
    Rendering/Utils/PixelStats.cs.slang
    Rendering/Utils/PixelStats.h
    Rendering/Utils/PixelStats.slang
    Rendering/Utils/PixelStatsShared.slang

    Rendering/Volumes/HomogeneousVolumeSampler.slang
    Rendering/Volumes/IPhaseFunction.slang
    Rendering/Volumes/PhaseFunction.slang
    Rendering/Volumes/GridVolumeSampler.cpp
    Rendering/Volumes/GridVolumeSampler.h
    Rendering/Volumes/GridVolumeSampler.slang
    Rendering/Volumes/GridVolumeSamplerParams.slang

    RenderPasses/ResolvePass.cpp
    RenderPasses/ResolvePass.h

    RenderPasses/Shared/Denoising/NRDBuffers.slang
    RenderPasses/Shared/Denoising/NRDConstants.slang
    RenderPasses/Shared/Denoising/NRDData.slang
    RenderPasses/Shared/Denoising/NRDHelpers.slang

    Scene/HitInfo.cpp
    Scene/HitInfo.h
    Scene/HitInfo.slang
    Scene/HitInfoType.slang
    Scene/Importer.cpp
    Scene/Importer.h
    Scene/Intersection.slang
    Scene/NullTrace.cs.slang
    Scene/Raster.slang
    Scene/Raytracing.slang
    Scene/RaytracingInline.slang
    Scene/Scene.cpp
    Scene/Scene.h
    Scene/Scene.slang
    Scene/SceneBlock.slang
    Scene/SceneBuilder.cpp
    Scene/SceneBuilder.h
    Scene/SceneCache.cpp
    Scene/SceneCache.h
    Scene/SceneDefines.slangh
    Scene/SceneIDs.h
    Scene/SceneRayQueryInterface.slang
    Scene/SceneTypes.slang
    Scene/Shading.slang
    Scene/ShadingData.slang
    Scene/Transform.cpp
    Scene/Transform.h
    Scene/TriangleMesh.cpp
    Scene/TriangleMesh.h
    Scene/VertexAttrib.slangh

    Scene/Animation/Animatable.cpp
    Scene/Animation/Animatable.h
    Scene/Animation/AnimatedVertexCache.cpp
    Scene/Animation/AnimatedVertexCache.h
    Scene/Animation/Animation.cpp
    Scene/Animation/Animation.h
    Scene/Animation/AnimationController.cpp
    Scene/Animation/AnimationController.h
    Scene/Animation/SharedTypes.slang
    Scene/Animation/Skinning.slang
    Scene/Animation/UpdateCurveAABBs.slang
    Scene/Animation/UpdateCurvePolyTubeVertices.slang
    Scene/Animation/UpdateCurveVertices.slang
    Scene/Animation/UpdateMeshVertices.slang

    Scene/Camera/Camera.cpp
    Scene/Camera/Camera.h
    Scene/Camera/CameraController.cpp
    Scene/Camera/CameraController.h
    Scene/Camera/Camera.slang
    Scene/Camera/CameraData.slang

    Scene/Curves/CurveConfig.h
    Scene/Curves/CurveTessellation.cpp
    Scene/Curves/CurveTessellation.h

    Scene/Displacement/DisplacementData.slang
    Scene/Displacement/DisplacementMapping.slang
    Scene/Displacement/DisplacementUpdate.cs.slang
    Scene/Displacement/DisplacementUpdateTask.slang

    Scene/Importers/AssimpImporter.cpp
    Scene/Importers/AssimpImporter.h
    Scene/Importers/PythonImporter.cpp
    Scene/Importers/PythonImporter.h

    Scene/Importers/PBRTImporter/Builder.cpp
    Scene/Importers/PBRTImporter/Builder.h
    Scene/Importers/PBRTImporter/EnvMapConverter.cs.slang
    Scene/Importers/PBRTImporter/EnvMapConverter.h
    Scene/Importers/PBRTImporter/Helpers.h
    Scene/Importers/PBRTImporter/LoopSubdivide.cpp
    Scene/Importers/PBRTImporter/LoopSubdivide.h
    Scene/Importers/PBRTImporter/Parameters.cpp
    Scene/Importers/PBRTImporter/Parameters.h
    Scene/Importers/PBRTImporter/Parser.cpp
    Scene/Importers/PBRTImporter/Parser.h
    Scene/Importers/PBRTImporter/PBRTImporter.cpp
    Scene/Importers/PBRTImporter/PBRTImporter.h
    Scene/Importers/PBRTImporter/Types.h

    Scene/Lights/BakeIesProfile.cs.slang
    Scene/Lights/BuildTriangleList.cs.slang
    Scene/Lights/EmissiveIntegrator.3d.slang
    Scene/Lights/EnvMap.cpp
    Scene/Lights/EnvMap.h
    Scene/Lights/EnvMap.slang
    Scene/Lights/EnvMapData.slang
    Scene/Lights/FinalizeIntegration.cs.slang
    Scene/Lights/Light.cpp
    Scene/Lights/Light.h
    Scene/Lights/LightCollection.cpp
    Scene/Lights/LightCollection.h
    Scene/Lights/LightCollection.slang
    Scene/Lights/LightCollectionShared.slang
    Scene/Lights/LightData.slang
    Scene/Lights/LightProfile.cpp
    Scene/Lights/LightProfile.h
    Scene/Lights/LightProfile.slang
    Scene/Lights/MeshLightData.slang
    Scene/Lights/UpdateTriangleVertices.cs.slang

    Scene/Material/AlphaTest.slang
    Scene/Material/BasicMaterial.cpp
    Scene/Material/BasicMaterial.h
    Scene/Material/BasicMaterialData.slang
    Scene/Material/ClothMaterial.cpp
    Scene/Material/ClothMaterial.h
    Scene/Material/HairMaterial.cpp
    Scene/Material/HairMaterial.h
    Scene/Material/HomogeneousVolumeData.slang
    Scene/Material/Material.cpp
    Scene/Material/Material.h
    Scene/Material/MaterialData.slang
    Scene/Material/MaterialFactory.slang
    Scene/Material/MaterialSystem.cpp
    Scene/Material/MaterialSystem.h
    Scene/Material/MaterialSystem.slang
    Scene/Material/MaterialTextureLoader.cpp
    Scene/Material/MaterialTextureLoader.h
    Scene/Material/MaterialTypes.slang
    Scene/Material/MERLMaterial.cpp
    Scene/Material/MERLMaterial.h
    Scene/Material/MERLMaterialData.slang
    Scene/Material/RGLCommon.cpp
    Scene/Material/RGLCommon.h
    Scene/Material/RGLFile.cpp
    Scene/Material/RGLFile.h
    Scene/Material/RGLMaterial.cpp
    Scene/Material/RGLMaterial.h
    Scene/Material/RGLMaterialData.slang
    Scene/Material/ShadingUtils.slang
    Scene/Material/StandardMaterial.cpp
    Scene/Material/StandardMaterial.h
    Scene/Material/TextureHandle.slang
    Scene/Material/TextureSampler.slang

    Scene/Material/PBRT/PBRTDiffuseMaterial.h
    Scene/Material/PBRT/PBRTDiffuseMaterial.cpp
    Scene/Material/PBRT/PBRTDiffuseTransmissionMaterial.h
    Scene/Material/PBRT/PBRTDiffuseTransmissionMaterial.cpp
    Scene/Material/PBRT/PBRTConductorMaterial.h
    Scene/Material/PBRT/PBRTConductorMaterial.cpp
    Scene/Material/PBRT/PBRTDielectricMaterial.h
    Scene/Material/PBRT/PBRTDielectricMaterial.cpp
    Scene/Material/PBRT/PBRTCoatedConductorMaterial.h
    Scene/Material/PBRT/PBRTCoatedConductorMaterial.cpp
    Scene/Material/PBRT/PBRTCoatedDiffuseMaterial.h
    Scene/Material/PBRT/PBRTCoatedDiffuseMaterial.cpp

    Scene/SDFs/NormalizedDenseSDFGrid/NDSDFGrid.cpp
    Scene/SDFs/NormalizedDenseSDFGrid/NDSDFGrid.h
    Scene/SDFs/NormalizedDenseSDFGrid/NDSDFGrid.slang

    Scene/SDFs/SparseBrickSet/BC4Encode.slang
    Scene/SDFs/SparseBrickSet/SDFSBS.cpp
    Scene/SDFs/SparseBrickSet/SDFSBS.h
    Scene/SDFs/SparseBrickSet/SDFSBS.slang
    Scene/SDFs/SparseBrickSet/SDFSBSAssignBrickValidityFromSDFieldPass.cs.slang
    Scene/SDFs/SparseBrickSet/SDFSBSCompactifyChunks.cs.slang
    Scene/SDFs/SparseBrickSet/SDFSBSComputeIntervalSDFieldFromGrid.cs.slang
    Scene/SDFs/SparseBrickSet/SDFSBSCopyIndirectionBuffer.cs.slang
    Scene/SDFs/SparseBrickSet/SDFSBSCreateBricksFromChunks.cs.slang
    Scene/SDFs/SparseBrickSet/SDFSBSCreateBricksFromSDField.cs.slang
    Scene/SDFs/SparseBrickSet/SDFSBSCreateChunksFromPrimitives.cs.slang
    Scene/SDFs/SparseBrickSet/SDFSBSExpandSDFieldData.cs.slang
    Scene/SDFs/SparseBrickSet/SDFSBSPruneEmptyBricks.cs.slang
    Scene/SDFs/SparseBrickSet/SDFSBSResetBrickValidity.cs.slang

    Scene/SDFs/SparseVoxelOctree/SDFSVO.cpp
    Scene/SDFs/SparseVoxelOctree/SDFSVO.h
    Scene/SDFs/SparseVoxelOctree/SDFSVO.slang
    Scene/SDFs/SparseVoxelOctree/SDFSVOBuildLevelFromTexture.cs.slang
    Scene/SDFs/SparseVoxelOctree/SDFSVOBuildOctree.cs.slang
    Scene/SDFs/SparseVoxelOctree/SDFSVOHashTable.slang
    Scene/SDFs/SparseVoxelOctree/SDFSVOLocationCodeSorter.cs.slang
    Scene/SDFs/SparseVoxelOctree/SDFSVOWriteSVOOffsets.cs.slang

    Scene/SDFs/SparseVoxelSet/SDFSVS.cpp
    Scene/SDFs/SparseVoxelSet/SDFSVS.h
    Scene/SDFs/SparseVoxelSet/SDFSVS.slang
    Scene/SDFs/SparseVoxelSet/SDFSVSVoxelizer.cs.slang

    Scene/SDFs/EvaluateSDFPrimitives.cs.slang
    Scene/SDFs/SDF3DPrimitive.slang
    Scene/SDFs/SDF3DPrimitiveCommon.slang
    Scene/SDFs/SDF3DPrimitiveFactory.cpp
    Scene/SDFs/SDF3DPrimitiveFactory.h
    Scene/SDFs/SDFGrid.cpp
    Scene/SDFs/SDFGrid.h
    Scene/SDFs/SDFGrid.slang
    Scene/SDFs/SDFGridBase.slang
    Scene/SDFs/SDFGridHitData.slang
    Scene/SDFs/SDFGridNoDefines.slangh
    Scene/SDFs/SDFSurfaceVoxelCounter.cs.slang
    Scene/SDFs/SDFVoxelCommon.slang
    Scene/SDFs/SDFVoxelHitUtils.slang
    Scene/SDFs/SDFVoxelTypes.slang

    Scene/Volume/BC4Encode.h
    Scene/Volume/BrickedGrid.h
    Scene/Volume/Grid.cpp
    Scene/Volume/Grid.h
    Scene/Volume/Grid.slang
    Scene/Volume/GridConverter.h
    Scene/Volume/GridVolume.cpp
    Scene/Volume/GridVolume.h
    Scene/Volume/GridVolume.slang
    Scene/Volume/GridVolumeData.slang

    Testing/UnitTest.cpp
    Testing/UnitTest.cs.slang
    Testing/UnitTest.h

    Utils/AlignedAllocator.h
    Utils/Attributes.slang
    Utils/BinaryFileStream.h
    Utils/CryptoUtils.cpp
    Utils/CryptoUtils.h
    Utils/HostDeviceShared.slangh
    Utils/InternalDictionary.h
    Utils/Logger.cpp
    Utils/Logger.h
    Utils/NumericRange.h
    Utils/NVAPI.slang
    Utils/NVAPI.slangh
    Utils/ObjectID.h
    Utils/Settings.h
    Utils/StringFormatters.h
    Utils/StringUtils.cpp
    Utils/StringUtils.h
    Utils/TermColor.cpp
    Utils/TermColor.h
    Utils/Threading.cpp
    Utils/Threading.h

    Utils/Algorithm/BitonicSort.cpp
    Utils/Algorithm/BitonicSort.cs.slang
    Utils/Algorithm/BitonicSort.h
    Utils/Algorithm/ComputeParallelReduction.cpp
    Utils/Algorithm/ComputeParallelReduction.h
    Utils/Algorithm/DirectedGraph.h
    Utils/Algorithm/DirectedGraphTraversal.h
    Utils/Algorithm/ParallelReduction.cpp
    Utils/Algorithm/ParallelReduction.cs.slang
    Utils/Algorithm/ParallelReduction.h
    Utils/Algorithm/ParallelReduction.ps.slang
    Utils/Algorithm/ParallelReductionType.slangh
    Utils/Algorithm/PrefixSum.cpp
    Utils/Algorithm/PrefixSum.cs.slang
    Utils/Algorithm/PrefixSum.h

    Utils/Color/ColorHelpers.slang
    Utils/Color/ColorMap.slang
    Utils/Color/ColorUtils.h
    Utils/Color/SampledSpectrum.h
    Utils/Color/Spectrum.cpp
    Utils/Color/Spectrum.h
    Utils/Color/SpectrumUtils.cpp
    Utils/Color/SpectrumUtils.h
    Utils/Color/SpectrumUtils.slang

    Utils/Debug/DebugConsole.h
    Utils/Debug/PixelDebug.cpp
    Utils/Debug/PixelDebug.h
    Utils/Debug/PixelDebug.slang
    Utils/Debug/PixelDebugTypes.slang
    Utils/Debug/ReflectPixelDebugTypes.cs.slang

    Utils/Geometry/GeometryHelpers.slang
    Utils/Geometry/IntersectionHelpers.slang

    Utils/Image/AsyncTextureLoader.cpp
    Utils/Image/AsyncTextureLoader.h
    Utils/Image/Bitmap.cpp
    Utils/Image/Bitmap.h
    Utils/Image/CopyColorChannel.cs.slang
    Utils/Image/ImageIO.cpp
    Utils/Image/ImageIO.h
    Utils/Image/ImageProcessing.cpp
    Utils/Image/ImageProcessing.h
    Utils/Image/TextureAnalyzer.cpp
    Utils/Image/TextureAnalyzer.cs.slang
    Utils/Image/TextureAnalyzer.h
    Utils/Image/TextureManager.cpp
    Utils/Image/TextureManager.h

    Utils/Math/AABB.cpp
    Utils/Math/AABB.h
    Utils/Math/AABB.slang
    Utils/Math/BitTricks.slang
    Utils/Math/Common.h
    Utils/Math/CubicSpline.h
    Utils/Math/FalcorMath.h
    Utils/Math/Float16.h
    Utils/Math/FNVHash.h
    Utils/Math/FormatConversion.slang
    Utils/Math/HalfUtils.slang
    Utils/Math/HashUtils.slang
    Utils/Math/IntervalArithmetic.slang
    Utils/Math/MathConstants.slangh
    Utils/Math/MathHelpers.h
    Utils/Math/MathHelpers.slang
    Utils/Math/MatrixUtils.slang
    Utils/Math/Matrix.h
    Utils/Math/PackedFormats.h
    Utils/Math/PackedFormats.slang
    Utils/Math/Quaternion.slang
    Utils/Math/Ray.h
    Utils/Math/Ray.slang
    Utils/Math/SphericalHarmonics.slang
    Utils/Math/Vector.h

    Utils/Math/Matrix/Matrix.h

    Utils/Perception/Experiment.cpp
    Utils/Perception/Experiment.h
    Utils/Perception/SingleThresholdMeasurement.cpp
    Utils/Perception/SingleThresholdMeasurement.h

    Utils/SampleGenerators/CPUSampleGenerator.h
    Utils/SampleGenerators/DxSamplePattern.cpp
    Utils/SampleGenerators/DxSamplePattern.h
    Utils/SampleGenerators/HaltonSamplePattern.cpp
    Utils/SampleGenerators/HaltonSamplePattern.h
    Utils/SampleGenerators/StratifiedSamplePattern.cpp
    Utils/SampleGenerators/StratifiedSamplePattern.h

    Utils/Sampling/AliasTable.cpp
    Utils/Sampling/AliasTable.h
    Utils/Sampling/AliasTable.slang
    Utils/Sampling/SampleGenerator.cpp
    Utils/Sampling/SampleGenerator.h
    Utils/Sampling/SampleGenerator.slang
    Utils/Sampling/SampleGeneratorInterface.slang
    Utils/Sampling/SampleGeneratorType.slangh
    Utils/Sampling/TinyUniformSampleGenerator.slang
    Utils/Sampling/UniformSampleGenerator.slang

    Utils/Sampling/LowDiscrepancy/HammersleySequence.slang

    Utils/Sampling/Pseudorandom/LCG.slang
    Utils/Sampling/Pseudorandom/SplitMix64.slang
    Utils/Sampling/Pseudorandom/Xorshift32.slang
    Utils/Sampling/Pseudorandom/Xoshiro.slang

    Utils/Scripting/Console.cpp
    Utils/Scripting/Console.h
    Utils/Scripting/Dictionary.h
    Utils/Scripting/ScriptBindings.cpp
    Utils/Scripting/ScriptBindings.h
    Utils/Scripting/Scripting.cpp
    Utils/Scripting/Scripting.h
    Utils/Scripting/ScriptWriter.h

    Utils/SDF/SDF2DDraw.slang
    Utils/SDF/SDF2DPrimitives.slang
    Utils/SDF/SDF3DShapes.slang
    Utils/SDF/SDFOperations.slang
    Utils/SDF/SDFOperationType.slang

    Utils/Timing/Clock.cpp
    Utils/Timing/Clock.h
    Utils/Timing/CpuTimer.h
    Utils/Timing/FrameRate.cpp
    Utils/Timing/FrameRate.h
    Utils/Timing/GpuTimer.slang
    Utils/Timing/Profiler.cpp
    Utils/Timing/Profiler.h
    Utils/Timing/ProfilerUI.cpp
    Utils/Timing/ProfilerUI.h
    Utils/Timing/TimeReport.cpp
    Utils/Timing/TimeReport.h

    Utils/UI/Font.cpp
    Utils/UI/Font.h
    Utils/UI/Gui.cpp
    Utils/UI/Gui.h
    Utils/UI/Gui.slang
    Utils/UI/InputState.cpp
    Utils/UI/InputState.h
    Utils/UI/InputTypes.h
    Utils/UI/PixelZoom.cpp
    Utils/UI/PixelZoom.h
    Utils/UI/SpectrumUI.cpp
    Utils/UI/SpectrumUI.h
    Utils/UI/TextRenderer.cpp
    Utils/UI/TextRenderer.h
    Utils/UI/TextRenderer.3d.slang

    Utils/Video/VideoEncoder.cpp
    Utils/Video/VideoEncoder.h
    Utils/Video/VideoEncoderUI.cpp
    Utils/Video/VideoEncoderUI.h
)


if(FALCOR_WINDOWS)
    target_sources(Falcor PRIVATE
        Core/Platform/Windows/ProgressBarWin.cpp
        Core/Platform/Windows/Windows.cpp
    )
endif()

if(FALCOR_LINUX)
    target_sources(Falcor PRIVATE
        Core/Platform/Linux/Linux.cpp
        Core/Platform/Linux/ProgressBarLinux.cpp
    )
endif()

if(FALCOR_RENDER_BACKEND STREQUAL "D3D12")
    target_compile_definitions(Falcor PUBLIC FALCOR_D3D12)
    target_sources(Falcor PRIVATE
        Core/API/D3D12/D3D12API.cpp
        Core/API/D3D12/D3D12API.h
        Core/API/D3D12/D3D12ApiData.h
        Core/API/D3D12/D3D12Buffer.cpp
        Core/API/D3D12/D3D12ComputeContext.cpp
        Core/API/D3D12/D3D12ComputeStateObject.cpp
        Core/API/D3D12/D3D12CopyContext.cpp
        Core/API/D3D12/D3D12DescriptorSet.cpp
        Core/API/D3D12/D3D12Device.cpp
        Core/API/D3D12/D3D12Fbo.cpp
        Core/API/D3D12/D3D12Formats.cpp
        Core/API/D3D12/D3D12GpuFence.cpp
        Core/API/D3D12/D3D12GpuMemoryHeap.cpp
        Core/API/D3D12/D3D12GpuTimer.cpp
        Core/API/D3D12/D3D12GraphicsStateObject.cpp
        Core/API/D3D12/D3D12IndirectCommands.cpp
        Core/API/D3D12/D3D12LowLevelContextData.cpp
        Core/API/D3D12/D3D12NvApiExDesc.h
        Core/API/D3D12/D3D12ParameterBlock.cpp
        Core/API/D3D12/D3D12ProgramVars.cpp
        Core/API/D3D12/D3D12QueryHeap.cpp
        Core/API/D3D12/D3D12RasterizerState.cpp
        Core/API/D3D12/D3D12RenderContext.cpp
        Core/API/D3D12/D3D12Resource.cpp
        Core/API/D3D12/D3D12Resource.h
        Core/API/D3D12/D3D12ResourceViews.cpp
        Core/API/D3D12/D3D12RtAccelerationStructure.cpp
        Core/API/D3D12/D3D12RtAccelerationStructure.h
        Core/API/D3D12/D3D12RtAccelerationStructurePostBuildInfoPool.cpp
        Core/API/D3D12/D3D12RtStateObject.cpp
        Core/API/D3D12/D3D12Sampler.cpp
        Core/API/D3D12/D3D12Shader.cpp
        Core/API/D3D12/D3D12ShaderTable.cpp
        Core/API/D3D12/D3D12State.cpp
        Core/API/D3D12/D3D12State.h
        Core/API/D3D12/D3D12Texture.cpp
        Core/API/D3D12/D3D12Vao.cpp
    )
endif()

if(FALCOR_RENDER_BACKEND STREQUAL "GFX-D3D12" OR FALCOR_RENDER_BACKEND STREQUAL "GFX-VK")
    if(FALCOR_RENDER_BACKEND STREQUAL "GFX-D3D12")
        target_compile_definitions(Falcor PUBLIC FALCOR_GFX FALCOR_GFX_D3D12)
    endif()
    if(FALCOR_RENDER_BACKEND STREQUAL "GFX-VK")
        target_compile_definitions(Falcor PUBLIC FALCOR_GFX FALCOR_GFX_VK)
    endif()

    target_sources(Falcor PRIVATE
        Core/API/GFX/GFXBuffer.cpp
        Core/API/GFX/GFXComputeContext.cpp
        Core/API/GFX/GFXComputeStateObject.cpp
        Core/API/GFX/GFXCopyContext.cpp
        Core/API/GFX/GFXD3D12DescriptorSet.cpp
        Core/API/GFX/GFXDevice.cpp
        Core/API/GFX/GFXDeviceApiData.h
        Core/API/GFX/GFXFbo.cpp
        Core/API/GFX/GFXFormats.cpp
        Core/API/GFX/GFXFormats.h
        Core/API/GFX/GFXGpuFence.cpp
        Core/API/GFX/GFXGpuMemoryHeap.cpp
        Core/API/GFX/GFXGpuTimer.cpp
        Core/API/GFX/GFXGraphicsStateObject.cpp
        Core/API/GFX/GFXLowLevelContextApiData.h
        Core/API/GFX/GFXLowLevelContextData.cpp
        Core/API/GFX/GFXParameterBlock.cpp
        Core/API/GFX/GFXProgramVars.cpp
        Core/API/GFX/GFXQueryHeap.cpp
        Core/API/GFX/GFXRasterizerState.cpp
        Core/API/GFX/GFXRenderContext.cpp
        Core/API/GFX/GFXResource.cpp
        Core/API/GFX/GFXResource.h
        Core/API/GFX/GFXResourceViews.cpp
        Core/API/GFX/GFXRtAccelerationStructure.cpp
        Core/API/GFX/GFXRtAccelerationStructure.h
        Core/API/GFX/GFXRtAccelerationStructurePostBuildInfoPool.cpp
        Core/API/GFX/GFXRtStateObject.cpp
        Core/API/GFX/GFXSampler.cpp
        Core/API/GFX/GFXShader.cpp
        Core/API/GFX/GFXTexture.cpp
        Core/API/GFX/GFXVao.cpp
    )
endif()

if(FALCOR_HAS_D3D12)
    target_sources(Falcor PRIVATE
        Core/API/Shared/D3D12DescriptorData.h
        Core/API/Shared/D3D12DescriptorHeap.cpp
        Core/API/Shared/D3D12DescriptorHeap.h
        Core/API/Shared/D3D12DescriptorPool.cpp
        Core/API/Shared/D3D12DescriptorPool.h
        Core/API/Shared/D3D12DescriptorSet.h
        Core/API/Shared/D3D12Handles.h
        Core/API/Shared/D3D12RootSignature.cpp
        Core/API/Shared/D3D12RootSignature.h
        Core/API/Shared/MockedD3D12StagingBuffer.cpp
        Core/API/Shared/MockedD3D12StagingBuffer.h
    )
endif()

if(FALCOR_HAS_NV_USD)
    target_sources(Falcor PRIVATE
        Scene/Importers/USDImporter/CreateSpecularTexture.cs.slang
        Scene/Importers/USDImporter/CreateSpecularTransmissionTexture.cs.slang
        Scene/Importers/USDImporter/ImporterContext.cpp
        Scene/Importers/USDImporter/ImporterContext.h
        Scene/Importers/USDImporter/PackBaseColorAlpha.cs.slang
        Scene/Importers/USDImporter/PreviewSurfaceConverter.cpp
        Scene/Importers/USDImporter/PreviewSurfaceConverter.h
        Scene/Importers/USDImporter/SampleTexture.slang
        Scene/Importers/USDImporter/USDHelpers.h
        Scene/Importers/USDImporter/USDImporter.cpp
        Scene/Importers/USDImporter/USDImporter.h
        Scene/Importers/USDImporter/Utils.h
    )
endif()

target_source_group(Falcor "/")

# ImGui source files
# Note: These added after calling target_source_group macro as they are outside the current folder.
target_sources(Falcor PRIVATE
    ../../external/imgui/imconfig.h
    ../../external/imgui/imgui.h
    ../../external/imgui/imgui_internal.h
    ../../external/imgui/imstb_rectpack.h
    ../../external/imgui/imstb_textedit.h
    ../../external/imgui/imstb_truetype.h
    ../../external/imgui/imgui.cpp
    ../../external/imgui/imgui_draw.cpp
    ../../external/imgui/imgui_widgets.cpp
    ../../external/imgui_addons/imguinodegrapheditor/imguinodegrapheditor.h
    ../../external/imgui_addons/imguinodegrapheditor/imguinodegrapheditor.cpp
)

target_copy_shaders(Falcor .)
target_copy_data_folder(Falcor)

target_compile_features(Falcor
    PUBLIC
        cxx_std_17
    PRIVATE
        cxx_std_17
)

target_compile_options(Falcor
    PUBLIC
        # MSVC flags.
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zi>      # generate debug symbols
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W2>      # increase warning level
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX>      # warnings as errors
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4251>  # 'type' : class 'type1' needs to have dll-interface to be used by clients of class 'type2'
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/MP>      # enable multi-processor compilation
        # Clang flags.
        $<$<CXX_COMPILER_ID:Clang>:-fms-extensions>     # enable MS extensions (among other things allow glm swizzle to work)
        $<$<CXX_COMPILER_ID:Clang>:-fvisibility=hidden> # hide symbols by default
    PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>  # big object files
    )

if(FALCOR_PRECOMPILED_HEADERS)
target_precompile_headers(Falcor
    PRIVATE
        Falcor.h
)
endif()

target_link_options(Falcor
    PUBLIC
        # MSVC flags.
        $<$<CXX_COMPILER_ID:MSVC>:/DEBUG>           # generate debug information
)

target_compile_definitions(Falcor
    PUBLIC
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        $<$<CONFIG:Release>:NDEBUG>
        $<$<CONFIG:Debug>:_DEBUG>
        # Windows.
        $<$<PLATFORM_ID:Windows>:NOMINMAX>  # do not define min/max macros
        $<$<PLATFORM_ID:Windows>:UNICODE>   # force character map to unicode
        # MSVC C++ library.
        $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>
        $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>
        $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
        # Clang.
        $<$<CXX_COMPILER_ID:Clang>:_MSC_EXTENSIONS> # enable MS extensions
        # TBB.
        $<$<CONFIG:Debug>:TBB_USE_DEBUG=0>
        # Falcor feature flags.
        FALCOR_HAS_D3D12=$<BOOL:${FALCOR_HAS_D3D12}>
        FALCOR_HAS_VULKAN=$<BOOL:${FALCOR_HAS_VULKAN}>
        FALCOR_HAS_NVAPI=$<BOOL:${FALCOR_HAS_NVAPI}>
        FALCOR_HAS_CUDA=$<BOOL:${FALCOR_HAS_CUDA}>
        FALCOR_HAS_D3D12_AGILITY_SDK=$<BOOL:${FALCOR_HAS_D3D12_AGILITY_SDK}>
        # TODO: RTXDI is always available, we might want to remove the feature flag.
        FALCOR_HAS_RTXDI=1
    PRIVATE
        #$<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>
        _PROJECT_DIR_="${CMAKE_CURRENT_SOURCE_DIR}/"
        FALCOR_DLL
        $<$<PLATFORM_ID:Windows>:IMGUI_API=__declspec\(dllexport\)>
        $<$<PLATFORM_ID:Linux>:IMGUI_API=__attribute__\(\(visibility\("default"\)\)\)>
)

target_link_libraries(Falcor
    PUBLIC
        fmt pybind11::embed args glm
        falcor_deps
        python
        slang slang-gfx
        imgui imgui_addons
        nanovdb
        nvtt
        external_includes
        rtxdi
        rtxgi
        $<$<BOOL:${FALCOR_HAS_D3D12}>:d3d12>
        $<$<BOOL:${FALCOR_HAS_CUDA}>:CUDA::cuda_driver>
        $<$<BOOL:${FALCOR_HAS_CUDA}>:CUDA::cudart_static>
        $<$<BOOL:${FALCOR_HAS_D3D12_AGILITY_SDK}>:agility-sdk>
        $<$<BOOL:${FALCOR_HAS_NVAPI}>:nvapi>
        $<$<BOOL:${FALCOR_HAS_PIX}>:pix>
        $<$<BOOL:${FALCOR_HAS_NV_USD}>:nv-usd>
    PRIVATE
        glfw mikktspace
        # Windows system libraries.
        $<$<PLATFORM_ID:Windows>:shcore.lib>
        $<$<PLATFORM_ID:Windows>:shlwapi.lib>
        $<$<PLATFORM_ID:Windows>:comctl32.lib>
        $<$<PLATFORM_ID:Windows>:setupapi.lib>  # Used in MonitorInfo
        # Linux system libraries.
        $<$<PLATFORM_ID:Linux>:gtk3>
)


target_include_directories(Falcor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(Falcor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${FALCOR_LIBRARY_OUTPUT_DIRECTORY}
)

add_dependencies(Falcor packman_dependencies)

validate_headers(Falcor)

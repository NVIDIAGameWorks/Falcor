add_library(Falcor SHARED)

target_sources(Falcor PRIVATE
    Falcor.natvis
    Falcor.manifest

    Falcor.h
    GlobalState.cpp
    GlobalState.h

    Core/AssetResolver.cpp
    Core/AssetResolver.h
    Core/Enum.h
    Core/Error.cpp
    Core/Error.h
    Core/GLFW.h
    Core/HotReloadFlags.h
    Core/Macros.h
    Core/Object.cpp
    Core/Object.h
    Core/ObjectPython.h
    Core/Plugin.cpp
    Core/Plugin.h
    Core/SampleApp.cpp
    Core/SampleApp.h
    Core/Testbed.cpp
    Core/Testbed.h
    Core/Version.cpp
    Core/Version.h
    Core/Window.cpp
    Core/Window.h

    Core/API/Aftermath.cpp
    Core/API/Aftermath.h
    Core/API/BlendState.cpp
    Core/API/BlendState.h
    Core/API/BlitContext.cpp
    Core/API/BlitContext.h
    Core/API/BlitReduction.3d.slang
    Core/API/Buffer.cpp
    Core/API/Buffer.h
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
    Core/API/FBO.cpp
    Core/API/FBO.h
    Core/API/Fence.cpp
    Core/API/Fence.h
    Core/API/FencedPool.h
    Core/API/Formats.cpp
    Core/API/Formats.h
    Core/API/GFXAPI.cpp
    Core/API/GFXAPI.h
    Core/API/GFXHelpers.cpp
    Core/API/GFXHelpers.h
    Core/API/GpuMemoryHeap.cpp
    Core/API/GpuMemoryHeap.h
    Core/API/GpuTimer.cpp
    Core/API/GpuTimer.h
    Core/API/GraphicsStateObject.cpp
    Core/API/GraphicsStateObject.h
    Core/API/LowLevelContextData.cpp
    Core/API/LowLevelContextData.h
    Core/API/NativeFormats.h
    Core/API/NativeHandle.h
    Core/API/NativeHandleTraits.h
    Core/API/NvApiExDesc.h
    Core/API/ParameterBlock.cpp
    Core/API/ParameterBlock.h
    Core/API/PythonHelpers.cpp
    Core/API/PythonHelpers.h
    Core/API/QueryHeap.cpp
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
    Core/API/ShaderResourceType.h
    Core/API/ShaderTable.h
    Core/API/Swapchain.cpp
    Core/API/Swapchain.h
    Core/API/Texture.cpp
    Core/API/Texture.h
    Core/API/Types.cpp
    Core/API/Types.h
    Core/API/VAO.cpp
    Core/API/VAO.h
    Core/API/VertexLayout.cpp
    Core/API/VertexLayout.h

    Core/Pass/BaseGraphicsPass.cpp
    Core/Pass/BaseGraphicsPass.h
    Core/Pass/ComputePass.cpp
    Core/Pass/ComputePass.h
    Core/Pass/FullScreenPass.cpp
    Core/Pass/FullScreenPass.gs.slang
    Core/Pass/FullScreenPass.h
    Core/Pass/FullScreenPass.vs.slang
    Core/Pass/RasterPass.cpp
    Core/Pass/RasterPass.h

    Core/Platform/LockFile.cpp
    Core/Platform/LockFile.h
    Core/Platform/MemoryMappedFile.cpp
    Core/Platform/MemoryMappedFile.h
    Core/Platform/MonitorInfo.cpp
    Core/Platform/MonitorInfo.h
    Core/Platform/OS.cpp
    Core/Platform/OS.h
    Core/Platform/PlatformHandles.h
    Core/Platform/ProgressBar.cpp
    Core/Platform/ProgressBar.h

    Core/Program/DefineList.h
    Core/Program/Program.cpp
    Core/Program/Program.h
    Core/Program/ProgramManager.cpp
    Core/Program/ProgramManager.h
    Core/Program/ProgramReflection.cpp
    Core/Program/ProgramReflection.h
    Core/Program/ProgramVars.cpp
    Core/Program/ProgramVars.h
    Core/Program/ProgramVersion.cpp
    Core/Program/ProgramVersion.h
    Core/Program/RtBindingTable.cpp
    Core/Program/RtBindingTable.h
    Core/Program/ShaderVar.cpp
    Core/Program/ShaderVar.h

    Core/State/ComputeState.cpp
    Core/State/ComputeState.h
    Core/State/GraphicsState.cpp
    Core/State/GraphicsState.h
    Core/State/StateGraph.h

    DiffRendering/AggregateGradients.cs.slang
    DiffRendering/DiffDebug.slang
    DiffRendering/DiffMaterialData.slang
    DiffRendering/DiffSceneIO.slang
    DiffRendering/DiffSceneQuery.slang
    DiffRendering/GradientIOWrapper.slang
    DiffRendering/SceneGradientInfo.slang
    DiffRendering/SceneGradients.cpp
    DiffRendering/SceneGradients.h
    DiffRendering/SceneGradients.slang
    DiffRendering/SharedTypes.slang

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
    RenderGraph/RenderPassReflection.cpp
    RenderGraph/RenderPassReflection.h
    RenderGraph/RenderPassStandardFlags.h
    RenderGraph/ResourceCache.cpp
    RenderGraph/ResourceCache.h

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
    Rendering/Materials/BSDFConfig.slangh
    Rendering/Materials/ClothBRDF.slang
    Rendering/Materials/ClothMaterialInstance.slang
    Rendering/Materials/ClothMaterial.slang
    Rendering/Materials/Fresnel.slang
    Rendering/Materials/HairMaterialInstance.slang
    Rendering/Materials/HairChiang16.slang
    Rendering/Materials/HairMaterial.slang
    Rendering/Materials/IMaterialInstance.slang
    Rendering/Materials/IBSDF.slang
    Rendering/Materials/IMaterial.slang
    Rendering/Materials/INDF.slang
    Rendering/Materials/InteriorList.slang
    Rendering/Materials/InteriorListHelpers.slang
    Rendering/Materials/IsotropicGGX.slang
    Rendering/Materials/LayeredBSDF.slang
    Rendering/Materials/LobeType.slang
    Rendering/Materials/MERLCommon.slang
    Rendering/Materials/MERLMaterialInstance.slang
    Rendering/Materials/MERLMaterial.slang
    Rendering/Materials/MERLMixMaterialInstance.slang
    Rendering/Materials/MERLMixMaterial.slang
    Rendering/Materials/Microfacet.slang
    Rendering/Materials/NDF.slang
    Rendering/Materials/RGLAcquisition.cpp
    Rendering/Materials/RGLAcquisition.cs.slang
    Rendering/Materials/RGLAcquisition.h
    Rendering/Materials/RGLMaterialInstance.slang
    Rendering/Materials/RGLCommon.slang
    Rendering/Materials/RGLMaterial.slang
    Rendering/Materials/StandardMaterialInstance.slang
    Rendering/Materials/StandardMaterial.slang
    Rendering/Materials/TexLODHelpers.slang
    Rendering/Materials/TexLODTypes.slang

    Rendering/Materials/PBRT/PBRTDiffuseMaterial.slang
    Rendering/Materials/PBRT/PBRTDiffuseMaterialInstance.slang
    Rendering/Materials/PBRT/PBRTDiffuseTransmissionMaterial.slang
    Rendering/Materials/PBRT/PBRTDiffuseTransmissionMaterialInstance.slang
    Rendering/Materials/PBRT/PBRTConductorMaterial.slang
    Rendering/Materials/PBRT/PBRTConductorMaterialInstance.slang
    Rendering/Materials/PBRT/PBRTDielectricMaterial.slang
    Rendering/Materials/PBRT/PBRTDielectricMaterialInstance.slang
    Rendering/Materials/PBRT/PBRTCoatedConductorMaterial.slang
    Rendering/Materials/PBRT/PBRTCoatedConductorMaterialInstance.slang
    Rendering/Materials/PBRT/PBRTCoatedDiffuseMaterial.slang
    Rendering/Materials/PBRT/PBRTCoatedDiffuseMaterialInstance.slang

    Rendering/Materials/BSDFs/BeerBTDF.slang
    Rendering/Materials/BSDFs/DielectricPlateBSDF.slang
    Rendering/Materials/BSDFs/DiffuseSpecularBRDF.slang
    Rendering/Materials/BSDFs/DisneyDiffuseBRDF.slang
    Rendering/Materials/BSDFs/FrostbiteDiffuseBRDF.slang
    Rendering/Materials/BSDFs/LambertDiffuseBRDF.slang
    Rendering/Materials/BSDFs/LambertDiffuseBTDF.slang
    Rendering/Materials/BSDFs/OrenNayarBRDF.slang
    Rendering/Materials/BSDFs/SheenBSDF.slang
    Rendering/Materials/BSDFs/SimpleBTDF.slang
    Rendering/Materials/BSDFs/SpecularMicrofacet.slang
    Rendering/Materials/BSDFs/StandardBSDF.slang

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
    Scene/ImporterError.h
    Scene/Intersection.slang
    Scene/MeshIO.cs.slang
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
    Scene/SceneBuilderDump.cpp
    Scene/SceneBuilderDump.h
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
    Scene/Material/DiffuseSpecularData.slang
    Scene/Material/DiffuseSpecularUtils.cpp
    Scene/Material/DiffuseSpecularUtils.h
    Scene/Material/HairMaterial.cpp
    Scene/Material/HairMaterial.h
    Scene/Material/Material.cpp
    Scene/Material/Material.h
    Scene/Material/MaterialData.slang
    Scene/Material/MaterialFactory.slang
    Scene/Material/MaterialParamLayout.h
    Scene/Material/MaterialParamLayout.slang
    Scene/Material/MaterialParamLayout.slangh
    Scene/Material/MaterialSystem.cpp
    Scene/Material/MaterialSystem.h
    Scene/Material/MaterialSystem.slang
    Scene/Material/MaterialTextureLoader.cpp
    Scene/Material/MaterialTextureLoader.h
    Scene/Material/MaterialTypeRegistry.cpp
    Scene/Material/MaterialTypeRegistry.h
    Scene/Material/MaterialTypes.slang
    Scene/Material/MERLFile.cpp
    Scene/Material/MERLFile.h
    Scene/Material/MERLMaterial.cpp
    Scene/Material/MERLMaterial.h
    Scene/Material/MERLMaterialData.slang
    Scene/Material/MERLMixMaterial.cpp
    Scene/Material/MERLMixMaterial.h
    Scene/Material/MERLMixMaterialData.slang
    Scene/Material/RGLCommon.cpp
    Scene/Material/RGLCommon.h
    Scene/Material/RGLFile.cpp
    Scene/Material/RGLFile.h
    Scene/Material/RGLMaterial.cpp
    Scene/Material/RGLMaterial.h
    Scene/Material/RGLMaterialData.slang
    Scene/Material/SerializedMaterialParams.h
    Scene/Material/ShadingUtils.slang
    Scene/Material/StandardMaterial.cpp
    Scene/Material/StandardMaterial.h
    Scene/Material/StandardMaterialParamLayout.slang
    Scene/Material/TextureHandle.slang
    Scene/Material/TextureSampler.slang
    Scene/Material/VolumeProperties.slang

    Scene/Material/PBRT/PBRTCoatedConductorMaterial.cpp
    Scene/Material/PBRT/PBRTCoatedConductorMaterial.h
    Scene/Material/PBRT/PBRTCoatedDiffuseMaterial.cpp
    Scene/Material/PBRT/PBRTCoatedDiffuseMaterial.h
    Scene/Material/PBRT/PBRTConductorMaterial.cpp
    Scene/Material/PBRT/PBRTConductorMaterial.h
    Scene/Material/PBRT/PBRTConductorMaterialParamLayout.slang
    Scene/Material/PBRT/PBRTDielectricMaterial.cpp
    Scene/Material/PBRT/PBRTDielectricMaterial.h
    Scene/Material/PBRT/PBRTDiffuseMaterial.cpp
    Scene/Material/PBRT/PBRTDiffuseMaterial.h
    Scene/Material/PBRT/PBRTDiffuseMaterialParamLayout.slang
    Scene/Material/PBRT/PBRTDiffuseTransmissionMaterial.cpp
    Scene/Material/PBRT/PBRTDiffuseTransmissionMaterial.h

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
    Utils/BufferAllocator.cpp
    Utils/BufferAllocator.h
    Utils/CryptoUtils.cpp
    Utils/CryptoUtils.h
    Utils/Dictionary.h
    Utils/fast_vector.h
    Utils/HostDeviceShared.slangh
    Utils/Logger.cpp
    Utils/Logger.h
    Utils/NumericRange.h
    Utils/NVAPI.slang
    Utils/NVAPI.slangh
    Utils/ObjectID.h
    Utils/ObjectIDPython.h
    Utils/PathResolving.cpp
    Utils/PathResolving.h
    Utils/Properties.cpp
    Utils/Properties.h
    Utils/Settings.cpp
    Utils/Settings.h
    Utils/SharedCache.h
    Utils/SlangUtils.slang
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
    Utils/Algorithm/DirectedGraph.h
    Utils/Algorithm/DirectedGraphTraversal.h
    Utils/Algorithm/ParallelReduction.cpp
    Utils/Algorithm/ParallelReduction.cs.slang
    Utils/Algorithm/ParallelReduction.h
    Utils/Algorithm/ParallelReductionType.slangh
    Utils/Algorithm/PrefixSum.cpp
    Utils/Algorithm/PrefixSum.cs.slang
    Utils/Algorithm/PrefixSum.h
    Utils/Algorithm/UnionFind.h

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
    Utils/Debug/WarpProfiler.cpp
    Utils/Debug/WarpProfiler.h
    Utils/Debug/WarpProfiler.slang

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
    Utils/Math/DiffMathHelpers.slang
    Utils/Math/FalcorMath.h
    Utils/Math/Float16.cpp
    Utils/Math/Float16.h
    Utils/Math/FNVHash.h
    Utils/Math/FormatConversion.h
    Utils/Math/FormatConversion.slang
    Utils/Math/HalfUtils.slang
    Utils/Math/HashUtils.slang
    Utils/Math/IntervalArithmetic.slang
    Utils/Math/MathConstants.slangh
    Utils/Math/MathHelpers.h
    Utils/Math/MathHelpers.slang
    Utils/Math/Matrix.h
    Utils/Math/MatrixMath.h
    Utils/Math/MatrixTypes.h
    Utils/Math/MatrixUtils.slang
    Utils/Math/PackedFormats.h
    Utils/Math/PackedFormats.slang
    Utils/Math/Quaternion.h
    Utils/Math/Quaternion.slang
    Utils/Math/QuaternionMath.h
    Utils/Math/QuaternionTypes.h
    Utils/Math/Ray.h
    Utils/Math/Ray.slang
    Utils/Math/Rectangle.cpp
    Utils/Math/Rectangle.h
    Utils/Math/ScalarMath.h
    Utils/Math/ScalarTypes.h
    Utils/Math/ShadingFrame.slang
    Utils/Math/SphericalHarmonics.slang
    Utils/Math/Vector.h
    Utils/Math/VectorMath.h
    Utils/Math/VectorTypes.h

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
    Utils/Scripting/ndarray.cpp
    Utils/Scripting/ndarray.h
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
    Utils/UI/PythonUI.cpp
    Utils/UI/PythonUI.h
    Utils/UI/SpectrumUI.cpp
    Utils/UI/SpectrumUI.h
    Utils/UI/TextRenderer.3d.slang
    Utils/UI/TextRenderer.cpp
    Utils/UI/TextRenderer.h
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

if(FALCOR_HAS_D3D12)
    target_sources(Falcor PRIVATE
        Core/API/Shared/D3D12ConstantBufferView.h
        Core/API/Shared/D3D12ConstantBufferView.cpp
        Core/API/Shared/D3D12DescriptorData.h
        Core/API/Shared/D3D12DescriptorHeap.cpp
        Core/API/Shared/D3D12DescriptorHeap.h
        Core/API/Shared/D3D12DescriptorPool.cpp
        Core/API/Shared/D3D12DescriptorPool.h
        Core/API/Shared/D3D12DescriptorSet.cpp
        Core/API/Shared/D3D12DescriptorSet.h
        Core/API/Shared/D3D12Handles.h
        Core/API/Shared/D3D12RootSignature.cpp
        Core/API/Shared/D3D12RootSignature.h
        Core/API/Shared/MockedD3D12StagingBuffer.cpp
        Core/API/Shared/MockedD3D12StagingBuffer.h
    )
endif()

if(FALCOR_HAS_CUDA)
    target_sources(Falcor PRIVATE
        Utils/CudaRuntime.h
        Utils/CudaUtils.cpp
        Utils/CudaUtils.h
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
    ../../external/imgui/imgui_tables.cpp
    ../../external/imgui_addons/imguinodegrapheditor/imguinodegrapheditor.h
    ../../external/imgui_addons/imguinodegrapheditor/imguinodegrapheditor.cpp
)

target_copy_shaders(Falcor .)

target_compile_features(Falcor
    PUBLIC
        cxx_std_17
    PRIVATE
        cxx_std_17
)

target_compile_options(Falcor
    PUBLIC
        # MSVC flags.
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:
            /Zi                             # generate debug symbols
            /MP                             # enable multi-processor compilation
            # Configure warnings
            /WX                             # warnings as errors
            /W4                             # increase warning level
            /wd4251                         # 'type' : class 'type1' needs to have dll-interface to be used by clients of class 'type2'
            /wd4244                         # 'conversion' conversion from 'type1' to 'type2', possible loss of data
            /wd4267                         # 'var' : conversion from 'size_t' to 'type', possible loss of data
            /wd4100                         # unreferenced formal parameter
            /wd4201                         # nonstandard extension used: nameless struct/union
            /wd4245                         # conversion from 'type1' to 'type2', signed/unsigned mismatch
            /wd4189                         # local variable is initialized but not referenced
            /wd4127                         # conditional expression is constant
            /wd4701                         # potentially uninitialized local variable 'name' used
            /wd4703                         # potentially uninitialized local pointer variable 'name' used
            /wd4324                         # structure was padded due to alignment specifier
            /wd4505                         # unreferenced local function has been removed
            /wd4702                         # unreachable code
            /wd4389                         # signed/unsigned mismatch
            /wd4459                         # declaration of 'identifier' hides global declaration
            /wd4268                         # 'identifier' : 'const' static/global data initialized with compiler generated default constructor fills the object with zeros
            /external:templates-            # allows warnings from system headers when they occur in a template that's instantiated in your code
            # Enable warnings that are disabled with /W4
            /we4263                         # 'function': member function does not override any base class virtual member function
            /we4264                         # 'virtual_function': no override available for virtual member function from base 'class'; function is hidden
            /we5038                         # data member 'member1' will be initialized after data member 'member2'
            /we5259                         # 'specialized-type': explicit specialization requires 'template <>'
            /we5263                         # calling 'std::move' on a temporary object prevents copy elision
        >
        # Clang/GCC flags.
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
            -fms-extensions                 # enable MS extensions (among other things allow anonymous structs)
            -fvisibility=hidden             # hide symbols by default
            -Wall                           # set warning level
            -Wno-unused-function
            -Wno-unused-variable
            -Wno-unused-but-set-variable
            -Wno-switch
            -Wno-missing-braces
            -Wno-invalid-offsetof
        >
        # Clang flags.
        $<$<CXX_COMPILER_ID:Clang>:
            -Wno-unused-private-field
            -Wno-braced-scalar-init
            -Wno-self-assign-overloaded
        >
        # GCC flags.
        $<$<CXX_COMPILER_ID:GNU>:
            -fpermissive
            -Wno-sign-compare
            -Wno-literal-suffix
            -Wno-class-memaccess
            -Wno-strict-aliasing
            -Wno-maybe-uninitialized
            -Wno-stringop-truncation
        >
    PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>  # big object files
    )

if(FALCOR_ENABLE_ASAN)
    target_compile_options(Falcor
        PUBLIC
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:
                /fsanitize=address
            >
            $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
                -fsanitize=address
            >
    )
    target_link_options(Falcor
        PUBLIC
            $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
                -fsanitize=address
            >
    )
    target_compile_definitions(Falcor
        PUBLIC
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:
                _DISABLE_VECTOR_ANNOTATION
                _DISABLE_STRING_ANNOTATION
            >
    )
endif()

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

if(FALCOR_ENABLE_ASSERTS STREQUAL "AUTO")
    set(FALCOR_ENABLE_ASSERTS_ $<IF:$<CONFIG:Debug>,ON,OFF>)
else()
    set(FALCOR_ENABLE_ASSERTS_ ${FALCOR_ENABLE_ASSERTS})
endif()

target_compile_definitions(Falcor
    PUBLIC
        $<$<CONFIG:Release>:NDEBUG>
        $<$<CONFIG:Debug>:_DEBUG>
        # Windows.
        $<$<PLATFORM_ID:Windows>:NOMINMAX>  # do not define min/max macros
        $<$<PLATFORM_ID:Windows>:UNICODE>   # force character map to unicode
        # MSVC C++ library.
        $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>
        $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>
        $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
        $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING>
        # Clang.
        $<$<CXX_COMPILER_ID:Clang>:_MSC_EXTENSIONS> # enable MS extensions
        # Falcor feature flags.
        FALCOR_ENABLE_ASSERTS=$<BOOL:${FALCOR_ENABLE_ASSERTS_}>
        FALCOR_ENABLE_PROFILER=$<BOOL:${FALCOR_ENABLE_PROFILER}>
        FALCOR_HAS_D3D12=$<BOOL:${FALCOR_HAS_D3D12}>
        FALCOR_HAS_VULKAN=$<BOOL:${FALCOR_HAS_VULKAN}>
        FALCOR_HAS_AFTERMATH=$<BOOL:${FALCOR_HAS_AFTERMATH}>
        FALCOR_HAS_NVAPI=$<BOOL:${FALCOR_HAS_NVAPI}>
        FALCOR_HAS_CUDA=$<BOOL:${FALCOR_HAS_CUDA}>
        FALCOR_HAS_D3D12_AGILITY_SDK=$<BOOL:${FALCOR_HAS_D3D12_AGILITY_SDK}>
        # TODO: RTXDI is always available, we might want to remove the feature flag.
        FALCOR_HAS_RTXDI=1
    PRIVATE
        #$<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>
        FALCOR_PROJECT_DIR="${CMAKE_SOURCE_DIR}/"
        FALCOR_DLL
        $<$<PLATFORM_ID:Windows>:IMGUI_API=__declspec\(dllexport\)>
        $<$<PLATFORM_ID:Linux>:IMGUI_API=__attribute__\(\(visibility\("default"\)\)\)>
)

if(FALCOR_USE_SYSTEM_PYTHON)
    target_compile_definitions(Falcor
        PRIVATE
            FALCOR_PYTHON_EXECUTABLE="${Python_EXECUTABLE}"
    )
endif()

target_link_libraries(Falcor
    PUBLIC
        fmt pybind11::embed Vulkan::Headers
        slang slang-gfx
        imgui imgui_addons
        nanovdb
        external_includes
        rtxdi
        $<$<BOOL:${FALCOR_HAS_CUDA}>:CUDA::cuda_driver>
        $<$<BOOL:${FALCOR_HAS_CUDA}>:CUDA::cudart_static>
    PRIVATE
        git_version
        FreeImage assimp OpenEXR OpenVDB lz4 zlib pugixml
        glfw mikktspace nvtt
        $<$<BOOL:${FALCOR_HAS_D3D12}>:d3d12>
        $<$<BOOL:${FALCOR_HAS_D3D12_AGILITY_SDK}>:agility-sdk>
        $<$<BOOL:${FALCOR_HAS_AFTERMATH}>:aftermath>
        $<$<BOOL:${FALCOR_HAS_NVAPI}>:nvapi>
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

# Falcor python module

# The `falcor` python package is a wrapper around the `falcor_ext` extension module.
# The reason for this is that as of Python 3.8, we need to use os.add_dll_directory()
# to add the directory containing the main falcor DLL to the DLL search path.
# This is not possible to do from within the extension module, so we need to do it
# from the `falcor` package __init__.py file (which is copied from falcor__init__.py).

pybind11_add_module(FalcorPython FalcorPython.cpp)

target_link_libraries(FalcorPython PRIVATE Falcor)

set_target_properties(FalcorPython PROPERTIES
    OUTPUT_NAME falcor_ext
    ARCHIVE_OUTPUT_NAME FalcorPython # Visual Studio generator clashes with the main Falcor library otherwise
    RUNTIME_OUTPUT_DIRECTORY ${FALCOR_OUTPUT_DIRECTORY}/python/falcor
    LIBRARY_OUTPUT_DIRECTORY ${FALCOR_OUTPUT_DIRECTORY}/python/falcor
)

# Generate falcor package __init__.py file.
file(GENERATE OUTPUT ${FALCOR_OUTPUT_DIRECTORY}/python/falcor/__init__.py INPUT ${CMAKE_SOURCE_DIR}/build_scripts/falcor__init__.py)

# Generate python stub files.
if(FALCOR_WINDOWS)
    add_custom_command(
        TARGET FalcorPython POST_BUILD
        COMMAND ${FALCOR_OUTPUT_DIRECTORY}/setpath.bat & ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build_scripts/generate_stubs.py ${FALCOR_OUTPUT_DIRECTORY}/python || (exit 0)
    )
endif()
if(FALCOR_LINUX)
    add_custom_command(
        TARGET FalcorPython POST_BUILD
        WORKING_DIRECTORY ${FALCOR_OUTPUT_DIRECTORY}
        COMMAND ${CMAKE_SOURCE_DIR}/build_scripts/wrap_setpath.sh ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build_scripts/generate_stubs.py ${FALCOR_OUTPUT_DIRECTORY}/python || (exit 0)
    )
endif()

# Generate setpath scripts.
if(FALCOR_WINDOWS)
    file(GENERATE OUTPUT ${FALCOR_OUTPUT_DIRECTORY}/setpath.bat INPUT ${CMAKE_SOURCE_DIR}/build_scripts/setpath.bat.in)
    file(GENERATE OUTPUT ${FALCOR_OUTPUT_DIRECTORY}/setpath.ps1 INPUT ${CMAKE_SOURCE_DIR}/build_scripts/setpath.ps1.in)
endif()
if(FALCOR_LINUX)
    file(GENERATE OUTPUT ${FALCOR_OUTPUT_DIRECTORY}/setpath.sh INPUT ${CMAKE_SOURCE_DIR}/build_scripts/setpath.sh.in)
endif()

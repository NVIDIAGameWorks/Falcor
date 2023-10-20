add_falcor_executable(FalcorTest)

target_sources(FalcorTest PRIVATE
    FalcorTest.cpp

    Tests/Core/AftermathTests.cpp
    Tests/Core/AftermathTests.cs.slang
    Tests/Core/AssetResolverTests.cpp
    Tests/Core/BlitTests.cpp
    Tests/Core/BlitTests.cs.slang
    Tests/Core/BufferAccessTests.cpp
    Tests/Core/BufferTests.cpp
    Tests/Core/BufferTests.cs.slang
    Tests/Core/ConstantBufferTests.cpp
    Tests/Core/ConstantBufferTests.cs.slang
    Tests/Core/CoreTests.cpp
    Tests/Core/DDSReadTests.cpp
    Tests/Core/DDSReadTests.cs.slang
    Tests/Core/EnumTests.cpp
    Tests/Core/LargeBuffer.cpp
    Tests/Core/LargeBuffer.cs.slang
    Tests/Core/ObjectTests.cpp
    Tests/Core/ParamBlockCB.cpp
    Tests/Core/ParamBlockCB.cs.slang
    Tests/Core/ParamBlockDefinition.slang
    Tests/Core/ParamBlockReflection.cs.slang
    Tests/Core/PluginTests.cpp
    Tests/Core/ResourceAliasing.cpp
    Tests/Core/ResourceAliasing.cs.slang
    Tests/Core/RootBufferParamBlockTests.cpp
    Tests/Core/RootBufferParamBlockTests.cs.slang
    Tests/Core/RootBufferStructTests.cpp
    Tests/Core/RootBufferStructTests.cs.slang
    Tests/Core/RootBufferTests.cpp
    Tests/Core/RootBufferTests.cs.slang
    Tests/Core/TextureLoadTests.cs.slang
    Tests/Core/TextureTests.cpp
    Tests/Core/TextureTests.cs.slang

    Tests/DebugPasses/InvalidPixelDetectionTests.cpp

    Tests/DiffRendering/SceneGradientsTest.cpp
    Tests/DiffRendering/SceneGradientsTest.cs.slang

    Tests/DiffRendering/Material/DiffMaterialTests.cpp
    Tests/DiffRendering/Material/DiffMaterialTests.cs.slang

    Tests/Platform/LockFileTests.cpp
    Tests/Platform/MemoryMappedFileTests.cpp
    Tests/Platform/MonitorInfoTests.cpp
    Tests/Platform/OSTests.cpp

    Tests/Rendering/Materials/BSDFIntegratorTests.cpp
    Tests/Rendering/Materials/RGLAcquisitionTests.cpp
    Tests/Rendering/Materials/MicrofacetTests.cpp
    Tests/Rendering/Materials/MicrofacetTests.cs.slang

    Tests/Sampling/AliasTableTests.cpp
    Tests/Sampling/AliasTableTests.cs.slang
    Tests/Sampling/LowDiscrepancyTests.cpp
    Tests/Sampling/LowDiscrepancyTests.cs.slang
    Tests/Sampling/PointSetsTests.cpp
    Tests/Sampling/PointSetsTests.cs.slang
    Tests/Sampling/PseudorandomTests.cpp
    Tests/Sampling/PseudorandomTests.cs.slang
    Tests/Sampling/SampleGeneratorTests.cpp
    Tests/Sampling/SampleGeneratorTests.cs.slang

    Tests/Scene/EnvMapTests.cpp

    Tests/Scene/Material/BSDFTests.cpp
    Tests/Scene/Material/BSDFTests.cs.slang
    Tests/Scene/Material/HairChiang16Tests.cpp
    Tests/Scene/Material/HairChiang16Tests.cs.slang
    Tests/Scene/Material/MERLFileTests.cpp

    Tests/Slang/CastFloat16.cpp
    Tests/Slang/CastFloat16.cs.slang
    Tests/Slang/Float16Tests.cpp
    Tests/Slang/Float16Tests.cs.slang
    Tests/Slang/Float64Tests.cpp
    Tests/Slang/Float64Tests.cs.slang
    Tests/Slang/InheritanceTests.cpp
    Tests/Slang/InheritanceTests.cs.slang
    Tests/Slang/Int64Tests.cpp
    Tests/Slang/Int64Tests.cs.slang
    Tests/Slang/NestedStructs.cpp
    Tests/Slang/NestedStructs.cs.slang
    Tests/Slang/ShaderModel.cpp
    Tests/Slang/ShaderModel.cs.slang
    Tests/Slang/ShaderString.cpp
    Tests/Slang/ShaderStringDynamic.cs.slang
    Tests/Slang/ShaderStringDynamicType.slang
    Tests/Slang/ShaderStringImport.cs.slang
    Tests/Slang/ShaderStringImported.cs.slang
    Tests/Slang/ShaderStringImported.slang
    Tests/Slang/ShaderStringInline.cs.slang
    Tests/Slang/ShaderStringModule.cs.slang
    Tests/Slang/ShaderStringUtil.slang
    Tests/Slang/SlangExtension.cpp
    Tests/Slang/SlangExtension.cs.slang
    Tests/Slang/SlangInheritance.cpp
    Tests/Slang/SlangInheritance.cs.slang
    Tests/Slang/SlangMutatingTests.cpp
    Tests/Slang/SlangMutatingTests.cs.slang
    Tests/Slang/SlangReinterpretCast.cpp
    Tests/Slang/SlangReinterpretCast.cs.slang
    Tests/Slang/SlangShared.slang
    Tests/Slang/SlangTests.cpp
    Tests/Slang/SlangTests.cs.slang
    Tests/Slang/StructuredBufferMatrix.cpp
    Tests/Slang/StructuredBufferMatrix.cs.slang
    Tests/Slang/TemplatedLoad.cpp
    Tests/Slang/TemplatedLoad.cs.slang
    Tests/Slang/TraceRayFlags.cpp
    Tests/Slang/TraceRayFlags.cs.slang
    Tests/Slang/TraceRayInline.cpp
    Tests/Slang/TraceRayInline.cs.slang
    Tests/Slang/UnboundedDescriptorArray.cpp
    Tests/Slang/UnboundedDescriptorArray.cs.slang
    Tests/Slang/WaveOps.cpp
    Tests/Slang/WaveOps.cs.slang

    Tests/Utils/Color/SampledSpectrumTests.cpp
    Tests/Utils/Color/SpectrumTests.cpp
    Tests/Utils/Color/SpectrumUtilsTests.cpp
    Tests/Utils/Color/SpectrumUtilsTests.cs.slang

    Tests/Utils/Debug/WarpProfilerTests.cpp
    Tests/Utils/Debug/WarpProfilerTests.cs.slang

    Tests/Utils/Image/BitmapTests.cpp
    Tests/Utils/Image/TextureManagerTests.cpp

    Tests/Utils/AABBTests.cpp
    Tests/Utils/AABBTests.cs.slang
    Tests/Utils/AlignedAllocatorTests.cpp
    Tests/Utils/BitonicSortTests.cpp
    Tests/Utils/BitTricksTests.cpp
    Tests/Utils/BitTricksTests.cs.slang
    Tests/Utils/BufferAllocatorTests.cpp
    Tests/Utils/ColorUtilsTests.cpp
    Tests/Utils/CryptoUtilsTests.cpp
    Tests/Utils/Float16TypesTests.cpp
    Tests/Utils/GeometryHelpersTests.cpp
    Tests/Utils/GeometryHelpersTests.cs.slang
    Tests/Utils/HalfUtilsTests.cpp
    Tests/Utils/HalfUtilsTests.cs.slang
    Tests/Utils/HashUtilsTests.cpp
    Tests/Utils/HashUtilsTests.cs.slang
    Tests/Utils/ImageProcessing.cpp
    Tests/Utils/IntersectionHelpersTests.cpp
    Tests/Utils/IntersectionHelpersTests.cs.slang
    Tests/Utils/MathHelpersTests.cpp
    Tests/Utils/MathHelpersTests.cs.slang
    Tests/Utils/MatrixTests.cpp
    Tests/Utils/PackedFormatsTests.cpp
    Tests/Utils/PackedFormatsTests.cs.slang
    Tests/Utils/ParallelReductionTests.cpp
    Tests/Utils/PathResolvingTests.cpp
    Tests/Utils/PrefixSumTests.cpp
    Tests/Utils/PropertiesTests.cpp
    Tests/Utils/QuaternionTests.cpp
    Tests/Utils/RectangleTests.cpp
    Tests/Utils/SettingsTests.cpp
    Tests/Utils/StringUtilsTests.cpp
    Tests/Utils/TextureAnalyzerTests.cpp
    Tests/Utils/UnionFindTests.cpp
    Tests/Utils/VectorTests.cpp
)


target_link_libraries(FalcorTest PRIVATE args)

target_copy_shaders(FalcorTest .)

target_source_group(FalcorTest "Tools")

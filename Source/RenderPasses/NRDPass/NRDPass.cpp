/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include <Falcor.h>

#ifdef FALCOR_D3D12
#include "NRDPass.h"
#include "RenderPasses/Shared/Denoising/NRDConstants.slang"
#include <glm/gtc/type_ptr.hpp>
#include <sstream>

#if FALCOR_ENABLE_NRD
#pragma comment(lib, "NRD.lib")
#endif

const RenderPass::Info NRDPass::kInfo { "NRD", "NRD denoiser." };

namespace
{
    const char kShaderPackRadiance[] = "RenderPasses/NRDPass/PackRadiance.cs.slang";

    // Input buffer names.
    const char kInputDiffuseRadianceHitDist[] = "diffuseRadianceHitDist";
    const char kInputSpecularRadianceHitDist[] = "specularRadianceHitDist";
    const char kInputMotionVectors[] = "mvec";
    const char kInputNormalRoughnessMaterialID[] = "normWRoughnessMaterialID";
    const char kInputViewZ[] = "viewZ";

    // Output buffer names.
    const char kOutputFilteredDiffuseRadianceHitDist[] = "filteredDiffuseRadianceHitDist";
    const char kOutputFilteredSpecularRadianceHitDist[] = "filteredSpecularRadianceHitDist";

    // Serialized parameters.

    // Common settings.
    const char kWorldSpaceMotion[] = "worldSpaceMotion";
    const char kDisocclusionThreshold[] = "disocclusionThreshold";

    // Pack radiance settings.
    const char kMaxIntensity[] = "maxIntensity";

    // ReLAX settings.
    const char kSpecularPrepassBlurRadius[] = "specularPrepassBlurRadius";
    const char kDiffusePrepassBlurRadius[] = "diffusePrepassBlurRadius";
    const char kDiffuseMaxAccumulatedFrameNum[] = "diffuseMaxAccumulatedFrameNum";
    const char kDiffuseMaxFastAccumulatedFrameNum[] = "diffuseMaxFastAccumulatedFrameNum";
    const char kSpecularMaxAccumulatedFrameNum[] = "specularMaxAccumulatedFrameNum";
    const char kSpecularMaxFastAccumulatedFrameNum[] = "specularMaxFastAccumulatedFrameNum";
    const char kSpecularVarianceBoost[] = "specularVarianceBoost";
    const char kEnableSkipReprojectionTestWithoutMotion[] = "enableSkipReprojectionTestWithoutMotion";
    const char kEnableSpecularVirtualHistoryClamping[] = "enableSpecularVirtualHistoryClamping";
    const char kEnableRoughnessBasedSpecularAccumulation[] = "enableRoughnessBasedSpecularAccumulation";
    const char kDisocclusionFixEdgeStoppingNormalPower[] = "disocclusionFixEdgeStoppingNormalPower";
    const char kDisocclusionFixMaxRadius[] = "disocclusionFixMaxRadius";
    const char kDisocclusionFixNumFramesToFix[] = "disocclusionFixNumFramesToFix";
    const char kHistoryClampingColorBoxSigmaScale[] = "historyClampingColorBoxSigmaScale";
    const char kSpatialVarianceEstimationHistoryThreshold[] = "spatialVarianceEstimationHistoryThreshold";
    const char kEnableAntiFirefly[] = "enableAntiFirefly";
    const char kAtrousIterationNum[] = "atrousIterationNum";
    const char kPhiDepth[] = "phiDepth";
    const char kPhiNormal[] = "phiNormal";
    const char kDiffusePhiLuminance[] = "diffusePhiLuminance";
    const char kSpecularPhiLuminance[] = "specularPhiLuminance";
    const char kSpecularLobeAngleFraction[] = "specularLobeAngleFraction";
    const char kSpecularLobeAngleSlack[] = "specularLobeAngleSlack";
    const char kEnableRoughnessEdgeStopping[] = "enableRoughnessEdgeStopping";
    const char kRoughnessEdgeStoppingRelaxation[] = "roughnessEdgeStoppingRelaxation";
    const char kNormalEdgeStoppingRelaxation[] = "normalEdgeStoppingRelaxation";
    const char kLuminanceEdgeStoppingRelaxation[] = "luminanceEdgeStoppingRelaxation";
}

NRDPass::SharedPtr NRDPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new NRDPass(dict));
}

NRDPass::NRDPass(const Dictionary& dict)
    : RenderPass(kInfo)
{
#if FALCOR_ENABLE_NRD
    mpPackRadiancePass = ComputePass::create(kShaderPackRadiance);

    // Override some defaults coming from the NRD SDK.
    mRelaxSettings.diffuseMaxFastAccumulatedFrameNum = 2;
    mRelaxSettings.specularMaxFastAccumulatedFrameNum = 2;
    mRelaxSettings.enableSpecularVirtualHistoryClamping = false;
    mRelaxSettings.enableRoughnessBasedSpecularAccumulation = false;
    mRelaxSettings.phiDepth = 0.02f;
    mRelaxSettings.phiNormal = 16.0f;
    mRelaxSettings.specularLobeAngleFraction = 0.5f;
    mRelaxSettings.specularLobeAngleSlack = 5.0f;

    // Deserialize pass from dictionary.
    for (const auto& [key, value] : dict)
    {
        // Common settings.
        if (key == kWorldSpaceMotion) mWorldSpaceMotion = value;
        else if (key == kDisocclusionThreshold) mDisocclusionThreshold = value;

        // Pack radiance settings.
        else if (key == kMaxIntensity) mMaxIntensity = value;

        // ReLAX settings.
        else if (key == kSpecularPrepassBlurRadius) mRelaxSettings.specularPrepassBlurRadius = value;
        else if (key == kDiffusePrepassBlurRadius) mRelaxSettings.diffusePrepassBlurRadius = value;
        else if (key == kDiffuseMaxAccumulatedFrameNum) mRelaxSettings.diffuseMaxAccumulatedFrameNum = value;
        else if (key == kDiffuseMaxFastAccumulatedFrameNum) mRelaxSettings.diffuseMaxFastAccumulatedFrameNum = value;
        else if (key == kSpecularMaxAccumulatedFrameNum) mRelaxSettings.specularMaxAccumulatedFrameNum = value;
        else if (key == kSpecularMaxFastAccumulatedFrameNum) mRelaxSettings.specularMaxFastAccumulatedFrameNum = value;
        else if (key == kSpecularVarianceBoost) mRelaxSettings.specularVarianceBoost = value;
        else if (key == kEnableSkipReprojectionTestWithoutMotion) mRelaxSettings.enableSkipReprojectionTestWithoutMotion = value;
        else if (key == kEnableSpecularVirtualHistoryClamping) mRelaxSettings.enableSpecularVirtualHistoryClamping = value;
        else if (key == kEnableRoughnessBasedSpecularAccumulation) mRelaxSettings.enableRoughnessBasedSpecularAccumulation = value;
        else if (key == kDisocclusionFixEdgeStoppingNormalPower) mRelaxSettings.disocclusionFixEdgeStoppingNormalPower = value;
        else if (key == kDisocclusionFixMaxRadius) mRelaxSettings.disocclusionFixMaxRadius = value;
        else if (key == kDisocclusionFixNumFramesToFix) mRelaxSettings.disocclusionFixNumFramesToFix = value;
        else if (key == kHistoryClampingColorBoxSigmaScale) mRelaxSettings.historyClampingColorBoxSigmaScale = value;
        else if (key == kSpatialVarianceEstimationHistoryThreshold) mRelaxSettings.spatialVarianceEstimationHistoryThreshold = value;
        else if (key == kEnableAntiFirefly) mRelaxSettings.enableAntiFirefly = value;
        else if (key == kAtrousIterationNum) mRelaxSettings.atrousIterationNum = value;
        else if (key == kPhiDepth) mRelaxSettings.phiDepth = value;
        else if (key == kPhiNormal) mRelaxSettings.phiNormal = value;
        else if (key == kDiffusePhiLuminance) mRelaxSettings.diffusePhiLuminance = value;
        else if (key == kSpecularPhiLuminance) mRelaxSettings.specularPhiLuminance = value;
        else if (key == kSpecularLobeAngleFraction) mRelaxSettings.specularLobeAngleFraction = value;
        else if (key == kSpecularLobeAngleSlack) mRelaxSettings.specularLobeAngleSlack = value;
        else if (key == kEnableRoughnessEdgeStopping) mRelaxSettings.enableRoughnessEdgeStopping = value;
        else if (key == kRoughnessEdgeStoppingRelaxation) mRelaxSettings.roughnessEdgeStoppingRelaxation = value;
        else if (key == kNormalEdgeStoppingRelaxation) mRelaxSettings.normalEdgeStoppingRelaxation = value;
        else if (key == kLuminanceEdgeStoppingRelaxation) mRelaxSettings.luminanceEdgeStoppingRelaxation = value;
        else
        {
            logWarning("Unknown field '{}' in NRD dictionary.", key);
        }
    }
#endif // FALCOR_ENABLE_NRD
}

Falcor::Dictionary NRDPass::getScriptingDictionary()
{
    Dictionary dict;

#if FALCOR_ENABLE_NRD
    // Common settings.
    dict[kWorldSpaceMotion] = mWorldSpaceMotion;
    dict[kDisocclusionThreshold] = mDisocclusionThreshold;

    // Pack radiance settings.
    dict[kMaxIntensity] = mMaxIntensity;

    // ReLAX settings.
    dict[kSpecularPrepassBlurRadius] = mRelaxSettings.specularPrepassBlurRadius;
    dict[kDiffusePrepassBlurRadius] = mRelaxSettings.diffusePrepassBlurRadius;
    dict[kDiffuseMaxAccumulatedFrameNum] = mRelaxSettings.diffuseMaxAccumulatedFrameNum;
    dict[kDiffuseMaxFastAccumulatedFrameNum] = mRelaxSettings.diffuseMaxFastAccumulatedFrameNum;
    dict[kSpecularMaxAccumulatedFrameNum] = mRelaxSettings.specularMaxAccumulatedFrameNum;
    dict[kSpecularMaxFastAccumulatedFrameNum] = mRelaxSettings.specularMaxFastAccumulatedFrameNum;
    dict[kSpecularVarianceBoost] = mRelaxSettings.specularVarianceBoost;
    dict[kEnableSkipReprojectionTestWithoutMotion] = mRelaxSettings.enableSkipReprojectionTestWithoutMotion;
    dict[kEnableSpecularVirtualHistoryClamping] = mRelaxSettings.enableSpecularVirtualHistoryClamping;
    dict[kEnableRoughnessBasedSpecularAccumulation] = mRelaxSettings.enableRoughnessBasedSpecularAccumulation;
    dict[kDisocclusionFixEdgeStoppingNormalPower] = mRelaxSettings.disocclusionFixEdgeStoppingNormalPower;
    dict[kDisocclusionFixMaxRadius] = mRelaxSettings.disocclusionFixMaxRadius;
    dict[kDisocclusionFixNumFramesToFix] = mRelaxSettings.disocclusionFixNumFramesToFix;
    dict[kHistoryClampingColorBoxSigmaScale] = mRelaxSettings.historyClampingColorBoxSigmaScale;
    dict[kSpatialVarianceEstimationHistoryThreshold] = mRelaxSettings.spatialVarianceEstimationHistoryThreshold;
    dict[kEnableAntiFirefly] = mRelaxSettings.enableAntiFirefly;
    dict[kAtrousIterationNum] = mRelaxSettings.atrousIterationNum;
    dict[kPhiDepth] = mRelaxSettings.phiDepth;
    dict[kPhiNormal] = mRelaxSettings.phiNormal;
    dict[kDiffusePhiLuminance] = mRelaxSettings.diffusePhiLuminance;
    dict[kSpecularPhiLuminance] = mRelaxSettings.specularPhiLuminance;
    dict[kSpecularLobeAngleFraction] = mRelaxSettings.specularLobeAngleFraction;
    dict[kSpecularLobeAngleSlack] = mRelaxSettings.specularLobeAngleSlack;
    dict[kEnableRoughnessEdgeStopping] = mRelaxSettings.enableRoughnessEdgeStopping;
    dict[kRoughnessEdgeStoppingRelaxation] = mRelaxSettings.roughnessEdgeStoppingRelaxation;
    dict[kNormalEdgeStoppingRelaxation] = mRelaxSettings.normalEdgeStoppingRelaxation;
    dict[kLuminanceEdgeStoppingRelaxation] = mRelaxSettings.luminanceEdgeStoppingRelaxation;
#endif // FALCOR_ENABLE_NRD

    return dict;
}

RenderPassReflection NRDPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputDiffuseRadianceHitDist, "Diffuse radiance and hit distance");
    reflector.addInput(kInputSpecularRadianceHitDist, "Specular radiance and hit distance");
    reflector.addInput(kInputViewZ, "View Z");
    reflector.addInput(kInputNormalRoughnessMaterialID, "World normal, roughness, and material ID");
    reflector.addInput(kInputMotionVectors, "Motion vectors");

    reflector.addOutput(kOutputFilteredDiffuseRadianceHitDist, "Filtered diffuse radiance and hit distance").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputFilteredSpecularRadianceHitDist, "Filtered specular radiance and hit distance").format(ResourceFormat::RGBA16Float);
    return reflector;
}

void NRDPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mScreenSize = compileData.defaultTexDims;
    mFrameIndex = 0;
#if FALCOR_ENABLE_NRD
    reinit();
#endif
}

void NRDPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

#if FALCOR_ENABLE_NRD
    executeInternal(pRenderContext, renderData);
#else
    pRenderContext->blit(renderData[kInputDiffuseRadianceHitDist]->asTexture()->getSRV(), renderData[kOutputFilteredDiffuseRadianceHitDist]->asTexture()->getRTV());
    pRenderContext->blit(renderData[kInputSpecularRadianceHitDist]->asTexture()->getSRV(), renderData[kOutputFilteredSpecularRadianceHitDist]->asTexture()->getRTV());
#endif
}

void NRDPass::renderUI(Gui::Widgets& widget)
{
#if FALCOR_ENABLE_NRD
    const nrd::LibraryDesc& nrdLibraryDesc = nrd::GetLibraryDesc();
    char name[256];
    _snprintf_s(name, 255, "NRD Library v%u.%u.%u", nrdLibraryDesc.versionMajor, nrdLibraryDesc.versionMinor, nrdLibraryDesc.versionBuild);
    widget.text(name);

    widget.text("Common:");
    widget.text(mWorldSpaceMotion ? "World space motion" : "Screen space motion");
    widget.slider("Disocclusion threshold (%)", mDisocclusionThreshold, 0.0f, 5.0f, false, "%.2f");

    widget.text("Pack radiance:");
    widget.slider("Max intensity", mMaxIntensity, 0.f, 100000.f, false, "%.0f");

    // ReLAX settings.
    {
        widget.text("ReLAX:");
        widget.text("Prepass:");
        widget.slider("Diffuse blur radius", mRelaxSettings.diffusePrepassBlurRadius, 0.0f, 100.0f, false, "%.0f");
        widget.slider("Specular blur radius", mRelaxSettings.specularPrepassBlurRadius, 0.0f, 100.0f, false, "%.0f");
        widget.text("Reprojection:");
        widget.slider("Diffuse max accumulated frames", mRelaxSettings.diffuseMaxAccumulatedFrameNum, 0u, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
        widget.slider("Diffuse responsive max accumulated frames", mRelaxSettings.diffuseMaxFastAccumulatedFrameNum, 0u, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
        widget.slider("Specular max accumulated frames", mRelaxSettings.specularMaxAccumulatedFrameNum, 0u, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
        widget.slider("Specular responsive max accumulated frames", mRelaxSettings.specularMaxFastAccumulatedFrameNum, 0u, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
        widget.slider("Specular variance boost", mRelaxSettings.specularVarianceBoost, 0.0f, 8.0f, false, "%.1f");
        widget.checkbox("Skip reprojection test without motion", mRelaxSettings.enableSkipReprojectionTestWithoutMotion);
        widget.checkbox("Enable specular virtual history clamping", mRelaxSettings.enableSpecularVirtualHistoryClamping);
        widget.checkbox("Enable roughness based specular accumulation", mRelaxSettings.enableRoughnessBasedSpecularAccumulation);
        widget.text("Disocclusion fix:");
        widget.slider("Edge stopping normal power", mRelaxSettings.disocclusionFixEdgeStoppingNormalPower, 0.0f, 128.0f, false, "%.1f");
        widget.slider("Max kernel radius", mRelaxSettings.disocclusionFixMaxRadius, 0.0f, 100.0f, false, "%.0f");
        widget.slider("Frames to fix", (uint32_t&)mRelaxSettings.disocclusionFixNumFramesToFix, 0u, 100u);
        widget.text("History clamping & antilag:");
        widget.slider("Color clamping sigma", mRelaxSettings.historyClampingColorBoxSigmaScale, 0.0f, 10.0f, false, "%.1f");
        widget.text("Spatial variance estimation:");
        widget.slider("History threshold", (uint32_t&)mRelaxSettings.spatialVarianceEstimationHistoryThreshold, 0u, 10u);
        widget.text("Firefly filter:");
        widget.checkbox("Enable firefly filter", (bool&)mRelaxSettings.enableAntiFirefly);
        widget.text("Spatial filter:");
        widget.slider("A-trous iterations", (uint32_t&)mRelaxSettings.atrousIterationNum, 2u, 8u);
        widget.slider("Depth weight (relative fraction)", mRelaxSettings.phiDepth, 0.0f, 0.05f, false, "%.2f");
        widget.slider("Normal weight (power)", mRelaxSettings.phiNormal, 1.0f, 256.0f, false, "%.0f");
        widget.slider("Diffuse luminance weight (sigma scale)", mRelaxSettings.diffusePhiLuminance, 0.0f, 10.0f, false, "%.1f");
        widget.slider("Specular normal weight (fraction of lobe)", mRelaxSettings.specularLobeAngleFraction, 0.0f, 2.0f, false, "%.0f");
        widget.slider("Specular normal weight (degrees of slack)", mRelaxSettings.specularLobeAngleSlack, 0.0f, 180.0f, false, "%.0f");
        widget.slider("Specular luminance weight (sigma scale)", mRelaxSettings.specularPhiLuminance, 0.0f, 10.0f, false, "%.1f");
        widget.checkbox("Roughness edge stopping", mRelaxSettings.enableRoughnessEdgeStopping);
        widget.slider("Roughness relaxation", mRelaxSettings.roughnessEdgeStoppingRelaxation, 0.0f, 1.0f, false, "%.2f");
        widget.slider("Normal relaxation", mRelaxSettings.normalEdgeStoppingRelaxation, 0.0f, 1.0f, false, "%.2f");
        widget.slider("Luminance relaxation", mRelaxSettings.luminanceEdgeStoppingRelaxation, 0.0f, 1.0f, false, "%.2f");
    }
#else // FALCOR_ENABLE_NRD
    widget.textWrapped("NRD is not setup and enabled in `Source/Core/FalcorConfig.h` so this pass is disabled. Please configure NRD and then recompile to use this pass.");
#endif // FALCOR_ENABLE_NRD
}

void NRDPass::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
}

#if FALCOR_ENABLE_NRD

static void* nrdAllocate(void* userArg, size_t size, size_t alignment)
{
    return malloc(size);
}

static void* nrdReallocate(void* userArg, void* memory, size_t size, size_t alignment)
{
    return realloc(memory, size);
}

static void nrdFree(void* userArg, void* memory)
{
    free(memory);
}

static Falcor::ResourceFormat getFalcorFormat(nrd::Format format)
{
    switch (format)
    {
    case nrd::Format::R8_UNORM:             return Falcor::ResourceFormat::R8Unorm;
    case nrd::Format::R8_SNORM:             return Falcor::ResourceFormat::R8Snorm;
    case nrd::Format::R8_UINT:              return Falcor::ResourceFormat::R8Uint;
    case nrd::Format::R8_SINT:              return Falcor::ResourceFormat::R8Int;
    case nrd::Format::RG8_UNORM:            return Falcor::ResourceFormat::RG8Unorm;
    case nrd::Format::RG8_SNORM:            return Falcor::ResourceFormat::RG8Snorm;
    case nrd::Format::RG8_UINT:             return Falcor::ResourceFormat::RG8Uint;
    case nrd::Format::RG8_SINT:             return Falcor::ResourceFormat::RG8Int;
    case nrd::Format::RGBA8_UNORM:          return Falcor::ResourceFormat::RGBA8Unorm;
    case nrd::Format::RGBA8_SNORM:          return Falcor::ResourceFormat::RGBA8Snorm;
    case nrd::Format::RGBA8_UINT:           return Falcor::ResourceFormat::RGBA8Uint;
    case nrd::Format::RGBA8_SINT:           return Falcor::ResourceFormat::RGBA8Int;
    case nrd::Format::RGBA8_SRGB:           return Falcor::ResourceFormat::RGBA8UnormSrgb;
    case nrd::Format::R16_UNORM:            return Falcor::ResourceFormat::R16Unorm;
    case nrd::Format::R16_SNORM:            return Falcor::ResourceFormat::R16Snorm;
    case nrd::Format::R16_UINT:             return Falcor::ResourceFormat::R16Uint;
    case nrd::Format::R16_SINT:             return Falcor::ResourceFormat::R16Int;
    case nrd::Format::R16_SFLOAT:           return Falcor::ResourceFormat::R16Float;
    case nrd::Format::RG16_UNORM:           return Falcor::ResourceFormat::RG16Unorm;
    case nrd::Format::RG16_SNORM:           return Falcor::ResourceFormat::RG16Snorm;
    case nrd::Format::RG16_UINT:            return Falcor::ResourceFormat::RG16Uint;
    case nrd::Format::RG16_SINT:            return Falcor::ResourceFormat::RG16Int;
    case nrd::Format::RG16_SFLOAT:          return Falcor::ResourceFormat::RG16Float;
    case nrd::Format::RGBA16_UNORM:         return Falcor::ResourceFormat::RGBA16Unorm;
    case nrd::Format::RGBA16_SNORM:         return Falcor::ResourceFormat::Unknown; // Not defined in Falcor
    case nrd::Format::RGBA16_UINT:          return Falcor::ResourceFormat::RGBA16Uint;
    case nrd::Format::RGBA16_SINT:          return Falcor::ResourceFormat::RGBA16Int;
    case nrd::Format::RGBA16_SFLOAT:        return Falcor::ResourceFormat::RGBA16Float;
    case nrd::Format::R32_UINT:             return Falcor::ResourceFormat::R32Uint;
    case nrd::Format::R32_SINT:             return Falcor::ResourceFormat::R32Int;
    case nrd::Format::R32_SFLOAT:           return Falcor::ResourceFormat::R32Float;
    case nrd::Format::RG32_UINT:            return Falcor::ResourceFormat::RG32Uint;
    case nrd::Format::RG32_SINT:            return Falcor::ResourceFormat::RG32Int;
    case nrd::Format::RG32_SFLOAT:          return Falcor::ResourceFormat::RG32Float;
    case nrd::Format::RGB32_UINT:           return Falcor::ResourceFormat::RGB32Uint;
    case nrd::Format::RGB32_SINT:           return Falcor::ResourceFormat::RGB32Int;
    case nrd::Format::RGB32_SFLOAT:         return Falcor::ResourceFormat::RGB32Float;
    case nrd::Format::RGBA32_UINT:          return Falcor::ResourceFormat::RGBA32Uint;
    case nrd::Format::RGBA32_SINT:          return Falcor::ResourceFormat::RGBA32Int;
    case nrd::Format::RGBA32_SFLOAT:        return Falcor::ResourceFormat::RGBA32Float;
    case nrd::Format::R10_G10_B10_A2_UNORM: return Falcor::ResourceFormat::RGB10A2Unorm;
    case nrd::Format::R10_G10_B10_A2_UINT:  return Falcor::ResourceFormat::RGB10A2Uint;
    case nrd::Format::R11_G11_B10_UFLOAT:   return Falcor::ResourceFormat::R11G11B10Float;
    case nrd::Format::R9_G9_B9_E5_UFLOAT:   return Falcor::ResourceFormat::RGB9E5Float;
    default:                                return Falcor::ResourceFormat::Unknown;
    }
}

static void copyMatrix(float* dstMatrix, const glm::mat4x4& srcMatrix)
{
    memcpy(dstMatrix, static_cast<const float*>(glm::value_ptr(srcMatrix)), sizeof(glm::mat4x4));
}

void NRDPass::reinit()
{
    // Create a new denoiser instance.
    mpDenoiser = nullptr;

    const nrd::LibraryDesc& libraryDesc = nrd::GetLibraryDesc();

    const nrd::MethodDesc methods[] =
    {
        { nrd::Method::RELAX_DIFFUSE_SPECULAR, uint16_t(mScreenSize.x), uint16_t(mScreenSize.y) }
    };

    nrd::DenoiserCreationDesc denoiserCreationDesc;
    denoiserCreationDesc.memoryAllocatorInterface.Allocate = nrdAllocate;
    denoiserCreationDesc.memoryAllocatorInterface.Reallocate = nrdReallocate;
    denoiserCreationDesc.memoryAllocatorInterface.Free = nrdFree;
    denoiserCreationDesc.requestedMethodNum = 1;
    denoiserCreationDesc.requestedMethods = methods;

    nrd::Result res = nrd::CreateDenoiser(denoiserCreationDesc, mpDenoiser);

    if (res != nrd::Result::SUCCESS) throw RuntimeError("NRDPass: Failed to create NRD denoiser");

    createResources();
    createPipelines();
}

void NRDPass::createPipelines()
{
    mpPasses.clear();
    mpCachedProgramKernels.clear();
    mpCSOs.clear();
    mCBVSRVUAVdescriptorSetLayouts.clear();
    mpRootSignatures.clear();

    // Get denoiser desc for currently initialized denoiser implementation.
    const nrd::DenoiserDesc& denoiserDesc = nrd::GetDenoiserDesc(*mpDenoiser);

    // Create samplers descriptor layout and set.
    Falcor::D3D12RootSignature::DescriptorSetLayout SamplersDescriptorSetLayout;

    for (uint32_t j = 0; j < denoiserDesc.staticSamplerNum; j++)
    {
        SamplersDescriptorSetLayout.addRange(ShaderResourceType::Sampler, denoiserDesc.staticSamplers[j].registerIndex, 1);
    }
    mpSamplersDescriptorSet = Falcor::D3D12DescriptorSet::create(gpDevice->getD3D12GpuDescriptorPool(), SamplersDescriptorSetLayout);

    // Set sampler descriptors right away.
    for (uint32_t j = 0; j < denoiserDesc.staticSamplerNum; j++)
    {
        mpSamplersDescriptorSet->setSampler(0, j, mpSamplers[j].get());
    }

    // Go over NRD passes and creating descriptor sets, root signatures and PSOs for each.
    for (uint32_t i = 0; i < denoiserDesc.pipelineNum; i++)
    {
        const nrd::PipelineDesc& nrdPipelineDesc = denoiserDesc.pipelines[i];
        const nrd::ComputeShader& nrdComputeShader = nrdPipelineDesc.computeShaderDXIL;

        // Initialize descriptor set.
        Falcor::D3D12RootSignature::DescriptorSetLayout CBVSRVUAVdescriptorSetLayout;

        // Add constant buffer to descriptor set.
        CBVSRVUAVdescriptorSetLayout.addRange(ShaderResourceType::Cbv, denoiserDesc.constantBufferDesc.registerIndex, 1);

        for (uint32_t j = 0; j < nrdPipelineDesc.descriptorRangeNum; j++)
        {
            const nrd::DescriptorRangeDesc& nrdDescriptorRange = nrdPipelineDesc.descriptorRanges[j];

            ShaderResourceType descriptorType = nrdDescriptorRange.descriptorType == nrd::DescriptorType::TEXTURE ?
                ShaderResourceType::TextureSrv :
                ShaderResourceType::TextureUav;

            CBVSRVUAVdescriptorSetLayout.addRange(descriptorType, nrdDescriptorRange.baseRegisterIndex, nrdDescriptorRange.descriptorNum);
        }

        mCBVSRVUAVdescriptorSetLayouts.push_back(CBVSRVUAVdescriptorSetLayout);

        // Create root signature for the NRD pass.
        Falcor::D3D12RootSignature::Desc rootSignatureDesc;
        rootSignatureDesc.addDescriptorSet(SamplersDescriptorSetLayout);
        rootSignatureDesc.addDescriptorSet(CBVSRVUAVdescriptorSetLayout);

        const Falcor::D3D12RootSignature::Desc& desc = rootSignatureDesc;

        Falcor::D3D12RootSignature::SharedPtr pRootSig = Falcor::D3D12RootSignature::create(desc);

        mpRootSignatures.push_back(pRootSig);

        // Create Compute PSO for the NRD pass.
        {
            std::string shaderFileName = "nrd/Shaders/" + std::string(nrdPipelineDesc.shaderFileName) + ".hlsl";

            Program::Desc programDesc;
            programDesc.addShaderLibrary(shaderFileName).csEntry(nrdPipelineDesc.shaderEntryPointName);
            programDesc.setCompilerFlags(Shader::CompilerFlags::MatrixLayoutColumnMajor);
            Program::DefineList defines;
            defines.add("COMPILER_DXC");
            defines.add("NRD_NORMAL_ENCODING", "1"); // NRD_NORMAL_ENCODING_OCT10
            defines.add("NRD_USE_MATERIAL_ID_AWARE_FILTERING", "0");
            ComputePass::SharedPtr pPass = ComputePass::create(programDesc, defines);

            ComputeProgram::SharedPtr pProgram = pPass->getProgram();
            ProgramKernels::SharedConstPtr pProgramKernels = pProgram->getActiveVersion()->getKernels(pPass->getVars().get());

            ComputeStateObject::Desc csoDesc;
            csoDesc.setProgramKernels(pProgramKernels);
            csoDesc.setD3D12RootSignatureOverride(pRootSig);

            ComputeStateObject::SharedPtr pCSO = ComputeStateObject::create(csoDesc);

            mpPasses.push_back(pPass);
            mpCachedProgramKernels.push_back(pProgramKernels);
            mpCSOs.push_back(pCSO);
        }
    }
}

void NRDPass::createResources()
{
    // Destroy previously created resources.
    mpSamplers.clear();
    mpPermanentTextures.clear();
    mpTransientTextures.clear();
    mpConstantBuffer = nullptr;

    const nrd::DenoiserDesc& denoiserDesc = nrd::GetDenoiserDesc(*mpDenoiser);
    const uint32_t poolSize = denoiserDesc.permanentPoolSize + denoiserDesc.transientPoolSize;

    // Create samplers.
    for (uint32_t i = 0; i < denoiserDesc.staticSamplerNum; i++)
    {
        const nrd::StaticSamplerDesc& nrdStaticsampler = denoiserDesc.staticSamplers[i];
        Falcor::Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Falcor::Sampler::Filter::Linear, Falcor::Sampler::Filter::Linear, Falcor::Sampler::Filter::Point);

        if (nrdStaticsampler.sampler == nrd::Sampler::NEAREST_CLAMP || nrdStaticsampler.sampler == nrd::Sampler::LINEAR_CLAMP)
        {
            samplerDesc.setAddressingMode(Falcor::Sampler::AddressMode::Clamp, Falcor::Sampler::AddressMode::Clamp, Falcor::Sampler::AddressMode::Clamp);
        }
        else
        {
            samplerDesc.setAddressingMode(Falcor::Sampler::AddressMode::Mirror, Falcor::Sampler::AddressMode::Mirror, Falcor::Sampler::AddressMode::Mirror);
        }

        if (nrdStaticsampler.sampler == nrd::Sampler::NEAREST_CLAMP || nrdStaticsampler.sampler == nrd::Sampler::NEAREST_MIRRORED_REPEAT)
        {
            samplerDesc.setFilterMode(Falcor::Sampler::Filter::Point, Falcor::Sampler::Filter::Point, Falcor::Sampler::Filter::Point);
        }
        else
        {
            samplerDesc.setFilterMode(Falcor::Sampler::Filter::Linear, Falcor::Sampler::Filter::Linear, Falcor::Sampler::Filter::Point);
        }

        mpSamplers.push_back(Falcor::Sampler::create(samplerDesc));
    }

    // Texture pool.
    for (uint32_t i = 0; i < poolSize; i++)
    {
        const bool isPermanent = (i < denoiserDesc.permanentPoolSize);

        // Get texture desc.
        const nrd::TextureDesc& nrdTextureDesc = isPermanent
            ? denoiserDesc.permanentPool[i]
            : denoiserDesc.transientPool[i - denoiserDesc.permanentPoolSize];

        // Create texture.
        Falcor::ResourceFormat textureFormat = getFalcorFormat(nrdTextureDesc.format);
        Falcor::Texture::SharedPtr pTexture = Texture::create2D(nrdTextureDesc.width, nrdTextureDesc.height, textureFormat, 1u, nrdTextureDesc.mipNum, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

        if (isPermanent)
            mpPermanentTextures.push_back(pTexture);
        else
            mpTransientTextures.push_back(pTexture);
    }

    // Constant buffer.
    mpConstantBuffer = Buffer::create(
        denoiserDesc.constantBufferDesc.maxDataSize,
        Falcor::ResourceBindFlags::Constant,
        Falcor::Buffer::CpuAccess::Write,
        nullptr);

    // Textures for classic Falcor compute pass that packs radiance.
    mpDiffuseRadianceHitDistPackedTexture = Texture::create2D(mScreenSize.x, mScreenSize.y, ResourceFormat::RGBA16Float, 1u, 1u, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    mpSpecularRadianceHitDistPackedTexture = Texture::create2D(mScreenSize.x, mScreenSize.y, ResourceFormat::RGBA16Float, 1u, 1u, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
}

void NRDPass::executeInternal(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_ASSERT(mpScene);

    // Run classic Falcor compute pass to pack radiance.
    {
        FALCOR_PROFILE("PackRadiance");
        auto perImageCB = mpPackRadiancePass["PerImageCB"];

        perImageCB["gMaxIntensity"] = mMaxIntensity;
        perImageCB["gDiffuseRadianceHitDist"] = renderData[kInputDiffuseRadianceHitDist]->asTexture();
        perImageCB["gSpecularRadianceHitDist"] = renderData[kInputSpecularRadianceHitDist]->asTexture();
        perImageCB["gOutDiffuseRadianceHitDist"] = mpDiffuseRadianceHitDistPackedTexture;
        perImageCB["gOutSpecularRadianceHitDist"] = mpSpecularRadianceHitDistPackedTexture;
        mpPackRadiancePass->execute(pRenderContext, uint3(mScreenSize.x, mScreenSize.y, 1u));
    }

    nrd::SetMethodSettings(*mpDenoiser, nrd::Method::RELAX_DIFFUSE_SPECULAR, (void*)&mRelaxSettings);

    // Initialize common settings.
    glm::mat4 viewMatrix = mpScene->getCamera()->getViewMatrix();
    glm::mat4 projMatrix = mpScene->getCamera()->getData().projMatNoJitter;
    if (mFrameIndex == 0)
    {
        mPrevViewMatrix = viewMatrix;
        mPrevProjMatrix = projMatrix;
    }

    nrd::CommonSettings commonSettings;
    copyMatrix(commonSettings.viewToClipMatrix, projMatrix);
    copyMatrix(commonSettings.viewToClipMatrixPrev, mPrevProjMatrix);
    copyMatrix(commonSettings.worldToViewMatrix, viewMatrix);
    copyMatrix(commonSettings.worldToViewMatrixPrev, mPrevViewMatrix);
    // NRD's convention for the jitter is: [-0.5; 0.5] sampleUv = pixelUv + cameraJitter
    commonSettings.cameraJitter[0] = -mpScene->getCamera()->getJitterX();
    commonSettings.cameraJitter[1] = mpScene->getCamera()->getJitterY();
    commonSettings.denoisingRange = kNRDDepthRange;
    commonSettings.disocclusionThreshold = mDisocclusionThreshold * 0.01f;
    commonSettings.frameIndex = mFrameIndex;
    commonSettings.isMotionVectorInWorldSpace = mWorldSpaceMotion;

    mPrevViewMatrix = viewMatrix;
    mPrevProjMatrix = projMatrix;
    mFrameIndex++;

    // Run NRD dispatches.
    const nrd::DispatchDesc* dispatchDescs = nullptr;
    uint32_t dispatchDescNum = 0;
    nrd::Result result = nrd::GetComputeDispatches(*mpDenoiser, commonSettings, dispatchDescs, dispatchDescNum);
    FALCOR_ASSERT(result == nrd::Result::SUCCESS);

    for (uint32_t i = 0; i < dispatchDescNum; i++)
    {
        const nrd::DispatchDesc& dispatchDesc = dispatchDescs[i];
        FALCOR_PROFILE(dispatchDesc.name);
        dispatch(pRenderContext, renderData, dispatchDesc);
    }

    // Submit the existing command list and start a new one.
    pRenderContext->flush();
}

void NRDPass::dispatch(RenderContext* pRenderContext, const RenderData& renderData, const nrd::DispatchDesc& dispatchDesc)
{
    const nrd::DenoiserDesc& denoiserDesc = nrd::GetDenoiserDesc(*mpDenoiser);
    const nrd::PipelineDesc& pipelineDesc = denoiserDesc.pipelines[dispatchDesc.pipelineIndex];

    // Set root signature.
    mpRootSignatures[dispatchDesc.pipelineIndex]->bindForCompute(pRenderContext);

    // Upload constants.
    mpConstantBuffer->setBlob(dispatchDesc.constantBufferData, 0, dispatchDesc.constantBufferDataSize);

    // Create descriptor set for the NRD pass.
    Falcor::D3D12DescriptorSet::SharedPtr CBVSRVUAVDescriptorSet = Falcor::D3D12DescriptorSet::create(gpDevice->getD3D12GpuDescriptorPool(), mCBVSRVUAVdescriptorSetLayouts[dispatchDesc.pipelineIndex]);

    // Set CBV.
    CBVSRVUAVDescriptorSet->setCbv(0 /* NB: range #0 is CBV range */, denoiserDesc.constantBufferDesc.registerIndex, mpConstantBuffer->getCBV().get());

    uint32_t resourceIndex = 0;
    for (uint32_t descriptorRangeIndex = 0; descriptorRangeIndex < pipelineDesc.descriptorRangeNum; descriptorRangeIndex++)
    {
        const nrd::DescriptorRangeDesc& nrdDescriptorRange = pipelineDesc.descriptorRanges[descriptorRangeIndex];

        for (uint32_t descriptorOffset = 0; descriptorOffset < nrdDescriptorRange.descriptorNum; descriptorOffset++)
        {
            FALCOR_ASSERT(resourceIndex < dispatchDesc.resourceNum);
            const nrd::Resource& resource = dispatchDesc.resources[resourceIndex];

            FALCOR_ASSERT(resource.stateNeeded == nrdDescriptorRange.descriptorType);

            Falcor::Texture::SharedPtr texture;

            switch (resource.type)
            {
            case nrd::ResourceType::IN_MV:
                texture = renderData[kInputMotionVectors]->asTexture();
                break;
            case nrd::ResourceType::IN_NORMAL_ROUGHNESS:
                texture = renderData[kInputNormalRoughnessMaterialID]->asTexture();
                break;
            case nrd::ResourceType::IN_VIEWZ:
                texture = renderData[kInputViewZ]->asTexture();
                break;
            case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST:
                texture = mpDiffuseRadianceHitDistPackedTexture->asTexture();
                break;
            case nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST:
                texture = mpSpecularRadianceHitDistPackedTexture->asTexture();
                break;
            case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST:
                texture = renderData[kOutputFilteredDiffuseRadianceHitDist]->asTexture();
                break;
            case nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST:
                texture = renderData[kOutputFilteredSpecularRadianceHitDist]->asTexture();
                break;
            case nrd::ResourceType::TRANSIENT_POOL:
                texture = mpTransientTextures[resource.indexInPool];
                break;
            case nrd::ResourceType::PERMANENT_POOL:
                texture = mpPermanentTextures[resource.indexInPool];
                break;
            default:
                FALCOR_ASSERT(!"Unavailable resource type");
                break;
            }

            FALCOR_ASSERT(texture);

            // Set up resource barriers.
            Falcor::Resource::State newState = resource.stateNeeded == nrd::DescriptorType::TEXTURE ? Falcor::Resource::State::ShaderResource : Falcor::Resource::State::UnorderedAccess;
            for (uint16_t mip = 0; mip < resource.mipNum; mip++)
            {
                const Falcor::ResourceViewInfo viewInfo = Falcor::ResourceViewInfo(resource.mipOffset + mip, 1, 0, 1);
                pRenderContext->resourceBarrier(texture.get(), newState, &viewInfo);
            }

            // Set the SRV and UAV descriptors.
            if (nrdDescriptorRange.descriptorType == nrd::DescriptorType::TEXTURE)
            {
                Falcor::ShaderResourceView::SharedPtr pSRV = texture->getSRV(resource.mipOffset, resource.mipNum, 0, 1);
                CBVSRVUAVDescriptorSet->setSrv(descriptorRangeIndex + 1 /* NB: range #0 is CBV range */, nrdDescriptorRange.baseRegisterIndex + descriptorOffset, pSRV.get());
            }
            else
            {
                Falcor::UnorderedAccessView::SharedPtr pUAV = texture->getUAV(resource.mipOffset, 0, 1);
                CBVSRVUAVDescriptorSet->setUav(descriptorRangeIndex + 1 /* NB: range #0 is CBV range */, nrdDescriptorRange.baseRegisterIndex + descriptorOffset, pUAV.get());
            }

            resourceIndex++;
        }
    }

    FALCOR_ASSERT(resourceIndex == dispatchDesc.resourceNum);

    // Set descriptor sets.
    mpSamplersDescriptorSet->bindForCompute(pRenderContext, mpRootSignatures[dispatchDesc.pipelineIndex].get(), 0);
    CBVSRVUAVDescriptorSet->bindForCompute(pRenderContext, mpRootSignatures[dispatchDesc.pipelineIndex].get(), 1);

    // Set pipeline state.
    ComputePass::SharedPtr pPass = mpPasses[dispatchDesc.pipelineIndex];
    ComputeProgram::SharedPtr pProgram = pPass->getProgram();
    ProgramKernels::SharedConstPtr pProgramKernels = pProgram->getActiveVersion()->getKernels(pPass->getVars().get());

    // Check if anything changed.
    bool newProgram = (pProgramKernels.get() != mpCachedProgramKernels[dispatchDesc.pipelineIndex].get());
    if (newProgram)
    {
        mpCachedProgramKernels[dispatchDesc.pipelineIndex] = pProgramKernels;

        ComputeStateObject::Desc desc;
        desc.setProgramKernels(pProgramKernels);
        desc.setD3D12RootSignatureOverride(mpRootSignatures[dispatchDesc.pipelineIndex]);

        ComputeStateObject::SharedPtr pCSO = ComputeStateObject::create(desc);
        mpCSOs[dispatchDesc.pipelineIndex] = pCSO;
    }
    pRenderContext->getLowLevelData()->getCommandList()->SetPipelineState(mpCSOs[dispatchDesc.pipelineIndex]->getApiHandle());

    // Dispatch.
    pRenderContext->getLowLevelData()->getCommandList()->Dispatch(dispatchDesc.gridWidth, dispatchDesc.gridHeight, 1);
}

#endif FALCOR_ENABLE_NRD

#endif // FALCOR_D3D12

// Don't remove this. it's required for hot-reload to function properly.
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary & lib)
{
#ifdef FALCOR_D3D12
    lib.registerPass(NRDPass::kInfo, NRDPass::create);
#endif
}

/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "CSM.h"

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

static void regCSM(pybind11::module& m)
{
    pybind11::class_<CSM, RenderPass, CSM::SharedPtr> pass(m, "CSM");
    pass.def_property("cascadeCount", &CSM::getCascadeCount, &CSM::setCascadeCount);
    pass.def_property("mapSize", &CSM::getMapSize, &CSM::setMapSize);
    pass.def_property("visibilityBitCount", &CSM::getVisibilityBufferBitsPerChannel, &CSM::setVisibilityBufferBitsPerChannel);
    pass.def_property("filter", &CSM::getFilterMode, &CSM::setFilterMode);
    pass.def_property("sdsmLatency", &CSM::getSdsmReadbackLatency, &CSM::setSdsmReadbackLatency);
    pass.def_property("partition", &CSM::getPartitionMode, &CSM::setPartitionMode);
    pass.def_property("lambda", &CSM::getPSSMLambda, &CSM::setPSSMLambda);
    pass.def_property("minDistance", &CSM::getMinDistanceRange, &CSM::setMinDistanceRange);
    pass.def_property("maxDistance", &CSM::getMaxDistanceRange, &CSM::setMaxDistanceRange);
    pass.def_property("cascadeThreshold", &CSM::getCascadeBlendThreshold, &CSM::setCascadeBlendThreshold);
    pass.def_property("depthBias", &CSM::getDepthBias, &CSM::setDepthBias);
    pass.def_property("kernelWidth", &CSM::getPcfKernelWidth, &CSM::setPcfKernelWidth);
    pass.def_property("maxAniso", &CSM::getVsmMaxAnisotropy, &CSM::setVsmMaxAnisotropy);
    pass.def_property("bleedReduction", &CSM::getVsmLightBleedReduction, &CSM::setVsmLightBleedReduction);
    pass.def_property("positiveExp", &CSM::getEvsmPositiveExponent, &CSM::setEvsmPositiveExponent);
    pass.def_property("negativeExp", &CSM::getEvsmNegativeExponent, &CSM::setEvsmNegativeExponent);

    pybind11::enum_<CSM::PartitionMode> partitionMode(m, "PartitionMode");
    partitionMode.value("Linear", CSM::PartitionMode::Linear);
    partitionMode.value("Logarithmic", CSM::PartitionMode::Logarithmic);
    partitionMode.value("PSSM", CSM::PartitionMode::PSSM);
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("CSM", "Generates a visibility map for a single light source using the CSM technique", CSM::create);
    ScriptBindings::registerBinding(regCSM);
}

const char* CSM::kDesc = "The pass generates a visibility-map using the CSM technique. The map is for a single light-source.\n"
"It supports common filtering modes, including EVSM. It also supports PSSM and SDSM";

namespace
{
    const Gui::DropdownList kFilterList = {
    { (uint32_t)CsmFilter::Point, "Point" },
    { (uint32_t)CsmFilter::HwPcf, "2x2 HW PCF" },
    { (uint32_t)CsmFilter::FixedPcf, "Fixed-Size PCF" },
    { (uint32_t)CsmFilter::Vsm, "VSM" },
    { (uint32_t)CsmFilter::Evsm2, "EVSM2" },
    { (uint32_t)CsmFilter::Evsm4, "EVSM4" },
    { (uint32_t)CsmFilter::StochasticPcf, "Stochastic Poisson PCF" }
    };

    const Gui::DropdownList kPartitionList = {
        { (uint32_t)CSM::PartitionMode::Linear, "Linear" },
        { (uint32_t)CSM::PartitionMode::Logarithmic, "Logarithmic" },
        { (uint32_t)CSM::PartitionMode::PSSM, "PSSM" }
    };

    const Gui::DropdownList kMaxAniso = {
        { (uint32_t)1, "1" },
        { (uint32_t)2, "2" },
        { (uint32_t)4, "4" },
        { (uint32_t)8, "8" },
        { (uint32_t)16, "16" }
    };

    const std::string kDepth = "depth";
    const std::string kVisibility = "visibility";
    const std::string kBlurPass = "GaussianBlur";

    const std::string kMapSize = "mapSize";
    const std::string kVisBufferSize = "visibilityBufferSize";
    const std::string kCascadeCount = "cascadeCount";
    const std::string kVisMapBitsPerChannel = "visibilityMapBitsPerChannel";
    const std::string kBlurKernelWidth = "blurWidth";
    const std::string kBlurSigma = "blurSigma";

    const std::string kDepthPassFile = "RenderPasses/CSM/DepthPass.slang";
    const std::string kShadowPassfile = "RenderPasses/CSM/ShadowPass.slang";
    const std::string kVisibilityPassFile = "RenderPasses/CSM/VisibilityPass.ps.slang";
    const std::string kSdsmReadbackLatency = "kSdsmReadbackLatency";
}

#if 0
class CsmSceneRenderer : public SceneRenderer
{
public:
    using UniquePtr = std::unique_ptr<CsmSceneRenderer>;
    static UniquePtr create(const Scene::SharedConstPtr& pScene, const ProgramReflection::BindLocation& alphaMapCbLoc, const ProgramReflection::BindLocation& alphaMapLoc, const ProgramReflection::BindLocation& alphaMapSamplerLoc)
    {
        return UniquePtr(new CsmSceneRenderer(pScene, alphaMapCbLoc, alphaMapLoc, alphaMapSamplerLoc));
    }

    void setDepthClamp(bool enable) { mDepthClamp = enable; }

    void renderScene(RenderContext* pContext, GraphicsState* pState, GraphicsVars* pVars, const Camera* pCamera) override
    {
        pState->setRasterizerState(nullptr);
        mpLastSetRs = nullptr;
        SceneRenderer::renderScene(pContext, pState, pVars, pCamera);
    }

protected:
    CsmSceneRenderer(const Scene::SharedConstPtr& pScene, const ProgramReflection::BindLocation& alphaMapCbLoc, const ProgramReflection::BindLocation& alphaMapLoc, const ProgramReflection::BindLocation& alphaMapSamplerLoc)
        : SceneRenderer(std::const_pointer_cast<Scene>(pScene))
    {
        mBindLocations.alphaCB = alphaMapCbLoc;
        mBindLocations.alphaMap = alphaMapLoc;
        mBindLocations.alphaMapSampler = alphaMapSamplerLoc;

        toggleMeshCulling(false);
        Sampler::Desc desc;
        desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        mpAlphaSampler = Sampler::create(desc);

        RasterizerState::Desc rsDesc;
        rsDesc.setDepthClamp(true);
        mpDepthClampRS = RasterizerState::create(rsDesc);
        rsDesc.setCullMode(RasterizerState::CullMode::None);
        mpDepthClampNoCullRS = RasterizerState::create(rsDesc);
        rsDesc.setDepthClamp(false);
        mpNoCullRS = RasterizerState::create(rsDesc);
    }

    bool mMaterialChanged = false;
    Sampler::SharedPtr mpAlphaSampler;

    struct
    {
        ProgramReflection::BindLocation alphaMap;
        ProgramReflection::BindLocation alphaCB;
        ProgramReflection::BindLocation alphaMapSampler;
    } mBindLocations;

    bool mDepthClamp;
    RasterizerState::SharedPtr mpDepthClampNoCullRS;
    RasterizerState::SharedPtr mpNoCullRS;
    RasterizerState::SharedPtr mpDepthClampRS;

    RasterizerState::SharedPtr mpLastSetRs;

    RasterizerState::SharedPtr getRasterizerState(const Material* pMaterial)
    {
        if (pMaterial->getAlphaMode() == AlphaModeMask)
        {
            return mDepthClamp ? mpDepthClampNoCullRS : mpNoCullRS;
        }
        else
        {
            return mDepthClamp ? mpDepthClampRS : nullptr;
        }
    }

    bool setPerMaterialData(const CurrentWorkingData& currentData, const Material* pMaterial) override
    {
        mMaterialChanged = true;
        if (currentData.pMaterial->getAlphaMode() == AlphaModeMask)
        {
            float alphaThreshold = currentData.pMaterial->getAlphaThreshold();
            auto& pDefaultBlock = currentData.pVars;
            pDefaultBlock->getParameterBlock(mBindLocations.alphaCB, 0)->setBlob(&alphaThreshold, 0u, sizeof(float));
            if (currentData.pMaterial->getBaseColorTexture())
            {
                pDefaultBlock->setSrv(mBindLocations.alphaMap, 0, currentData.pMaterial->getBaseColorTexture()->getSRV());
            }
            else
            {
                pDefaultBlock->setSrv(mBindLocations.alphaMap, 0, nullptr);
            }
            pDefaultBlock->setSampler(mBindLocations.alphaMapSampler, 0, mpAlphaSampler);
            currentData.pState->getProgram()->addDefine("TEST_ALPHA");
        }
        else
        {
            currentData.pState->getProgram()->removeDefine("TEST_ALPHA");
        }
        const auto& pRsState = getRasterizerState(currentData.pMaterial);
        if (pRsState != mpLastSetRs)
        {
            currentData.pState->setRasterizerState(pRsState);
            mpLastSetRs = pRsState;
        }
        return true;
    };
};
#endif

static void createShadowMatrix(const DirectionalLight* pLight, const float3& center, float radius, glm::mat4& shadowVP)
{
    glm::mat4 view = glm::lookAt(center, center + pLight->getWorldDirection(), float3(0, 1, 0));
    glm::mat4 proj = glm::ortho(-radius, radius, -radius, radius, -radius, radius);

    shadowVP = proj * view;
}

static void createShadowMatrix(const PointLight* pLight, const float3& center, float radius, float fboAspectRatio, glm::mat4& shadowVP)
{
    const float3 lightPos = pLight->getWorldPosition();
    const float3 lookat = pLight->getWorldDirection() + lightPos;
    float3 up(0, 1, 0);
    if (abs(glm::dot(up, pLight->getWorldDirection())) >= 0.95f)
    {
        up = float3(1, 0, 0);
    }

    glm::mat4 view = glm::lookAt(lightPos, lookat, up);
    float distFromCenter = glm::length(lightPos - center);
    float nearZ = std::max(0.1f, distFromCenter - radius);
    float maxZ = std::min(radius * 2, distFromCenter + radius);
    float angle = pLight->getOpeningAngle() * 2;
    glm::mat4 proj = glm::perspective(angle, fboAspectRatio, nearZ, maxZ);

    shadowVP = proj * view;
}

static void createShadowMatrix(const Light* pLight, const float3& center, float radius, float fboAspectRatio, glm::mat4& shadowVP)
{
    switch (pLight->getType())
    {
    case LightType::Directional:
        return createShadowMatrix((DirectionalLight*)pLight, center, radius, shadowVP);
    case LightType::Point:
        return createShadowMatrix((PointLight*)pLight, center, radius, fboAspectRatio, shadowVP);
    default:
        should_not_get_here();
    }
}

void CSM::createDepthPassResources()
{
    GraphicsProgram::Desc desc;
    desc.addShaderLibrary(kDepthPassFile);
    desc.vsEntry("vsMain").psEntry("psMain");

    Program::DefineList defines;
    defines.add("_APPLY_PROJECTION");
    defines.add("TEST_ALPHA");
    defines.add("_ALPHA_CHANNEL", "a");

    mDepthPass.pProgram = GraphicsProgram::create(desc, defines);
    mDepthPass.pState = GraphicsState::create();
    mDepthPass.pState->setProgram(mDepthPass.pProgram);
}

void CSM::createVisibilityPassResources()
{
    mVisibilityPass.pFbo = Fbo::create();
    mVisibilityPass.pPass = FullScreenPass::create(kVisibilityPassFile);
    mVisibilityPass.mPassDataOffset = mVisibilityPass.pPass->getVars()->getParameterBlock("PerFrameCB")->getVariableOffset("gPass");
}

void CSM::createShadowPassResources()
{
    mShadowPass.mapSize = mMapSize;
    const ResourceFormat depthFormat = ResourceFormat::D32Float;
    mCsmData.depthBias = 0.005f;

    Program::DefineList defines;
    defines.add("TEST_ALPHA");
    defines.add("_CASCADE_COUNT", std::to_string(mCsmData.cascadeCount));
    defines.add("_ALPHA_CHANNEL", "a");
    ResourceFormat colorFormat = ResourceFormat::Unknown;
    switch ((CsmFilter)mCsmData.filterMode)
    {
    case CsmFilter::Vsm:
        colorFormat = ResourceFormat::RG16Float;
        defines.add("_VSM");
        break;
    case CsmFilter::Evsm2:
        colorFormat = ResourceFormat::RG16Float;
        defines.add("_EVSM2");
        break;
    case CsmFilter::Evsm4:
        colorFormat = ResourceFormat::RGBA16Float;
        defines.add("_EVSM4");
        break;
    }

    Fbo::Desc fboDesc;
    fboDesc.setDepthStencilTarget(depthFormat);
    uint32_t mipLevels = 1;

    if (colorFormat != ResourceFormat::Unknown)
    {
        fboDesc.setColorTarget(0, colorFormat);
        mipLevels = Texture::kMaxPossible;
    }
    mShadowPass.pFbo = Fbo::create2D(mMapSize.x, mMapSize.y, fboDesc, mCsmData.cascadeCount, mipLevels);
    mDepthPass.pState->setFbo(Fbo::create2D(mMapSize.x, mMapSize.y, fboDesc, mCsmData.cascadeCount));

    mShadowPass.fboAspectRatio = (float)mMapSize.x / (float)mMapSize.y;

    // Create the shadows program
    GraphicsProgram::Desc desc;
    desc.addShaderLibrary(kShadowPassfile);
    desc.vsEntry("vsMain").gsEntry("gsMain").psEntry("psMain");

    mShadowPass.pProgram = GraphicsProgram::create(desc, defines);
    mShadowPass.pState = GraphicsState::create();
    mShadowPass.pState->setProgram(mShadowPass.pProgram);
    mShadowPass.pState->setDepthStencilState(nullptr);
    mShadowPass.pState->setFbo(mShadowPass.pFbo);
}

CSM::CSM()
{
    createDepthPassResources();
    createVisibilityPassResources();

    mpLightCamera = Camera::create();

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Border, Sampler::AddressMode::Border, Sampler::AddressMode::Border).setBorderColor(float4(1.0f));
    samplerDesc.setLodParams(0.f, 0.f, 0.f);
    samplerDesc.setComparisonMode(Sampler::ComparisonMode::LessEqual);
    mShadowPass.pPointCmpSampler = Sampler::create(samplerDesc);
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mShadowPass.pLinearCmpSampler = Sampler::create(samplerDesc);
    samplerDesc.setComparisonMode(Sampler::ComparisonMode::Disabled);
    samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    samplerDesc.setLodParams(-100.f, 100.f, 0.f);

    createVsmSampleState(1);;
}

CSM::SharedPtr CSM::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    auto pCSM = SharedPtr(new CSM());
    for (const auto& [key, value] : dict)
    {
        if (key == kMapSize) pCSM->mMapSize = value;
        else if (key == kVisBufferSize) pCSM->mVisibilityPassData.screenDim = value;
        else if (key == kCascadeCount) pCSM->setCascadeCount(value);
        else if (key == kVisMapBitsPerChannel) pCSM->setVisibilityBufferBitsPerChannel(value);
        else if (key == kSdsmReadbackLatency) pCSM->setSdsmReadbackLatency(value);
        else if (key == kBlurKernelWidth) pCSM->mBlurDict["kernelWidth"] = (uint32_t)value;
        else if (key == kBlurSigma) pCSM->mBlurDict["sigma"] = (float)value;
        else logWarning("Unknown field '" + key + "' in a CSM dictionary");
    }
    pCSM->createShadowPassResources();
    return pCSM;
}

Dictionary CSM::getScriptingDictionary()
{
    Dictionary dict;
    dict[kMapSize] = mMapSize;
    dict[kVisBufferSize] = mVisibilityPassData.screenDim;
    dict[kCascadeCount] = mCsmData.cascadeCount;
    dict[kVisMapBitsPerChannel] = mVisibilityPassData.mapBitsPerChannel;
    dict[kSdsmReadbackLatency] = mSdsmData.readbackLatency;

    auto blurDict = mpBlurGraph->getPass(kBlurPass)->getScriptingDictionary();
    dict[kBlurKernelWidth] = (uint32_t)blurDict["kernelWidth"];
    dict[kBlurSigma] = (float)blurDict["sigma"];
    return dict;
}

static ResourceFormat getVisBufferFormat(uint32_t bitsPerChannel, bool visualizeCascades)
{
    switch (bitsPerChannel)
    {
    case 8:
        return visualizeCascades ? ResourceFormat::RGBA8Unorm : ResourceFormat::R8Unorm;
    case 16:
        return visualizeCascades ? ResourceFormat::RGBA16Float : ResourceFormat::R16Float;
    case 32:
        return visualizeCascades ? ResourceFormat::RGBA32Float : ResourceFormat::R32Float;
    default:
        should_not_get_here();
        return ResourceFormat::Unknown;
    }
}

RenderPassReflection CSM::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addOutput(kVisibility, "Visibility map. Values are [0,1] where 0 means the pixel is completely shadowed and 1 means it's not shadowed at all")
        .format(getVisBufferFormat(mVisibilityPassData.mapBitsPerChannel, mVisibilityPassData.shouldVisualizeCascades))
        .texture2D(mVisibilityPassData.screenDim.x, mVisibilityPassData.screenDim.y);
    reflector.addInput(kDepth, "Pre-initialized scene depth buffer used for SDSM.\nIf not provided, the pass will run a depth-pass internally").flags(RenderPassReflection::Field::Flags::Optional);
    return reflector;
}

void CSM::compile(RenderContext* pContext, const CompileData& compileData)
{
    mpBlurGraph = RenderGraph::create("Gaussian Blur");
    GaussianBlur::SharedPtr pBlurPass = GaussianBlur::create(pContext, mBlurDict);
    mpBlurGraph->addPass(pBlurPass, kBlurPass);
    mpBlurGraph->markOutput(kBlurPass + ".dst");

    mVisibilityPass.pFbo->attachColorTarget(nullptr, 0);
}

void camClipSpaceToWorldSpace(const Camera* pCamera, float3 viewFrustum[8], float3& center, float& radius)
{
    float3 clipSpace[8] =
    {
        float3(-1.0f, 1.0f, 0),
        float3(1.0f, 1.0f, 0),
        float3(1.0f, -1.0f, 0),
        float3(-1.0f, -1.0f, 0),
        float3(-1.0f, 1.0f, 1.0f),
        float3(1.0f, 1.0f, 1.0f),
        float3(1.0f, -1.0f, 1.0f),
        float3(-1.0f, -1.0f, 1.0f),
    };

    glm::mat4 invViewProj = pCamera->getInvViewProjMatrix();
    center = float3(0, 0, 0);

    for (uint32_t i = 0; i < 8; i++)
    {
        float4 crd = invViewProj * float4(clipSpace[i], 1);
        viewFrustum[i] = float3(crd) / crd.w;
        center += viewFrustum[i];
    }

    center *= (1.0f / 8.0f);

    // Calculate bounding sphere radius
    radius = 0;
    for (uint32_t i = 0; i < 8; i++)
    {
        float d = glm::length(center - viewFrustum[i]);
        radius = std::max(d, radius);
    }
}

forceinline float calcPssmPartitionEnd(float nearPlane, float camDepthRange, const float2& distanceRange, float linearBlend, uint32_t cascade, uint32_t cascadeCount)
{
    // Convert to camera space
    float minDepth = nearPlane + distanceRange.x * camDepthRange;
    float maxDepth = nearPlane + distanceRange.y * camDepthRange;

    float depthRange = maxDepth - minDepth;
    float depthScale = maxDepth / minDepth;

    float cascadeScale = float(cascade + 1) / float(cascadeCount);
    float logSplit = std::pow(depthScale, cascadeScale) * minDepth;
    float uniSplit = minDepth + depthRange * cascadeScale;

    float distance = linearBlend * logSplit + (1 - linearBlend) * uniSplit;

    // Convert back to clip-space
    distance = (distance - nearPlane) / camDepthRange;
    return distance;
}

void getCascadeCropParams(const float3 crd[8], const glm::mat4& lightVP, float4& scale, float4& offset)
{
    // Transform the frustum into light clip-space and calculate min-max
    float4 maxCS(-1, -1, 0, 1);
    float4 minCS(1, 1, 1, 1);
    for (uint32_t i = 0; i < 8; i++)
    {
        float4 c = lightVP * float4(crd[i], 1.0f);
        c /= c.w;
        maxCS = glm::max(maxCS, c);
        minCS = glm::min(minCS, c);
    }

    float4 delta = maxCS - minCS;
    scale = float4(2, 2, 1, 1) / delta;

    offset.x = -0.5f * (maxCS.x + minCS.x) * scale.x;
    offset.y = -0.5f * (maxCS.y + minCS.y) * scale.y;
    offset.z = -minCS.z * scale.z;

    scale.w = 1;
    offset.w = 0;
}

void CSM::partitionCascades(const Camera* pCamera, const float2& distanceRange)
{
    struct
    {
        float3 crd[8];
        float3 center;
        float radius;
    } camFrustum;

    camClipSpaceToWorldSpace(pCamera, camFrustum.crd, camFrustum.center, camFrustum.radius);

    // Create the global shadow space
    createShadowMatrix(mpLight.get(), camFrustum.center, camFrustum.radius, mShadowPass.fboAspectRatio, mCsmData.globalMat);

    if (mCsmData.cascadeCount == 1)
    {
        mCsmData.cascadeScale[0] = float4(1);
        mCsmData.cascadeOffset[0] = float4(0);
        mCsmData.cascadeRange[0].x = 0;
        mCsmData.cascadeRange[0].y = 1;
        return;
    }

    float nearPlane = pCamera->getNearPlane();
    float farPlane = pCamera->getFarPlane();
    float depthRange = farPlane - nearPlane;

    float nextCascadeStart = distanceRange.x;

    for (uint32_t c = 0; c < mCsmData.cascadeCount; c++)
    {
        float cascadeStart = nextCascadeStart;

        switch (mControls.partitionMode)
        {
        case PartitionMode::Linear:
            nextCascadeStart = cascadeStart + (distanceRange.y - distanceRange.x) / float(mCsmData.cascadeCount);
            break;
        case PartitionMode::Logarithmic:
            nextCascadeStart = calcPssmPartitionEnd(nearPlane, depthRange, distanceRange, 1.0f, c, mCsmData.cascadeCount);
            break;
        case PartitionMode::PSSM:
            nextCascadeStart = calcPssmPartitionEnd(nearPlane, depthRange, distanceRange, mControls.pssmLambda, c, mCsmData.cascadeCount);
            break;
        default:
            should_not_get_here();
        }

        // If we blend between cascades, we need to expand the range to make sure we will not try to read off the edge of the shadow-map
        float blendCorrection = (nextCascadeStart - cascadeStart) *  (mCsmData.cascadeBlendThreshold * 0.5f);
        float cascadeEnd = nextCascadeStart + blendCorrection;
        nextCascadeStart -= blendCorrection;

        // Calculate the cascade distance in camera-clip space(Where the clip-space range is [0, farPlane])
        float camClipSpaceCascadeStart = glm::lerp(nearPlane, farPlane, cascadeStart);
        float camClipSpaceCascadeEnd = glm::lerp(nearPlane, farPlane, cascadeEnd);

        //Convert to ndc space [0, 1]
        float projTermA = farPlane / (nearPlane - farPlane);
        float projTermB = (-farPlane * nearPlane) / (farPlane - nearPlane);
        float ndcSpaceCascadeStart = (-camClipSpaceCascadeStart * projTermA + projTermB) / camClipSpaceCascadeStart;
        float ndcSpaceCascadeEnd = (-camClipSpaceCascadeEnd * projTermA + projTermB) / camClipSpaceCascadeEnd;
        mCsmData.cascadeRange[c].x = ndcSpaceCascadeStart;
        mCsmData.cascadeRange[c].y = ndcSpaceCascadeEnd - ndcSpaceCascadeStart;

        // Calculate the cascade frustum
        float3 cascadeFrust[8];
        for (uint32_t i = 0; i < 4; i++)
        {
            float3 edge = camFrustum.crd[i + 4] - camFrustum.crd[i];
            float3 start = edge * cascadeStart;
            float3 end = edge * cascadeEnd;
            cascadeFrust[i] = camFrustum.crd[i] + start;
            cascadeFrust[i + 4] = camFrustum.crd[i] + end;
        }

        getCascadeCropParams(cascadeFrust, mCsmData.globalMat, mCsmData.cascadeScale[c], mCsmData.cascadeOffset[c]);
    }
}

static bool checkOffset(size_t cbOffset, size_t cppOffset, const char* field)
{
    if (cbOffset != cppOffset)
    {
        logError("CsmData::" + std::string(field) + " CB offset mismatch. CB offset is " + std::to_string(cbOffset) + ", C++ data offset is " + std::to_string(cppOffset));
        return false;
    }
    return true;
}

#if _LOG_ENABLED
#define check_offset(_a) {static bool b = true; if(b) {assert(checkOffset(pCB["gCsmData"][#_a].getByteOffset(), offsetof(CsmData, _a), #_a));} b = false;}
#else
#define check_offset(_a)
#endif

void CSM::renderScene(RenderContext* pCtx)
{
    auto pCB = mShadowPass.pVars->getParameterBlock(mPerLightCbLoc);
    check_offset(globalMat);
    check_offset(cascadeScale);
    check_offset(cascadeOffset);
    check_offset(cascadeRange);
    check_offset(depthBias);
    check_offset(cascadeCount);
    check_offset(filterMode);
    check_offset(pcfKernelWidth);
    check_offset(lightDir);
    check_offset(lightBleedingReduction);
    check_offset(evsmExponents);
    check_offset(cascadeBlendThreshold);


    pCB->setBlob(&mCsmData, 0, sizeof(mCsmData));
    mpLightCamera->setProjectionMatrix(mCsmData.globalMat);
    mpScene->render(pCtx, mShadowPass.pState.get(), mShadowPass.pVars.get());
    //        mpCsmSceneRenderer->renderScene(pCtx, mShadowPass.pState.get(), mShadowPass.pVars.get(), mpLightCamera.get());
}

void CSM::executeDepthPass(RenderContext* pCtx, const Camera* pCamera)
{
    uint32_t width = (uint32_t)mShadowPass.mapSize.x;
    uint32_t height = (uint32_t)mShadowPass.mapSize.y;

    Fbo::SharedConstPtr pFbo = mDepthPass.pState->getFbo();
    if ((pFbo == nullptr) || (pFbo->getWidth() != width) || (pFbo->getHeight() != height))
    {
        Fbo::Desc desc;
        desc.setDepthStencilTarget(mShadowPass.pFbo->getDepthStencilTexture()->getFormat());
        mDepthPass.pState->setFbo(Fbo::create2D(width, height, desc));
    }

    pCtx->clearFbo(pFbo.get(), float4(), 1, 0, FboAttachmentType::Depth);
    //        mpCsmSceneRenderer->renderScene(pCtx, mDepthPass.pState.get(), mDepthPass.pVars.get(), pCamera);
}

void CSM::createVsmSampleState(uint32_t maxAnisotropy)
{
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    samplerDesc.setMaxAnisotropy(maxAnisotropy);
    mShadowPass.pVSMTrilinearSampler = Sampler::create(samplerDesc);
}

void CSM::reduceDepthSdsmMinMax(RenderContext* pRenderCtx, const Camera* pCamera, Texture::SharedPtr pDepthBuffer)
{
    if (pDepthBuffer == nullptr)
    {
        // Run a shadow pass
        executeDepthPass(pRenderCtx, pCamera);
        pDepthBuffer = mDepthPass.pState->getFbo()->getDepthStencilTexture();
    }

    createSdsmData(pDepthBuffer);
    float2 distanceRange = float2(mSdsmData.minMaxReduction->reduce(pRenderCtx, pDepthBuffer));

    // Convert to linear
    glm::mat4 camProj = pCamera->getProjMatrix();
    distanceRange = camProj[2][2] - distanceRange * camProj[2][3];
    distanceRange = camProj[3][2] / distanceRange;
    distanceRange = (distanceRange - pCamera->getNearPlane()) / (pCamera->getFarPlane() - pCamera->getNearPlane());
    distanceRange = glm::clamp(distanceRange, float2(0), float2(1));
    mSdsmData.sdsmResult = distanceRange;

    if (mControls.stabilizeCascades)
    {
        // Ignore minor changes that can result in swimming
        distanceRange = round(distanceRange * 16.0f) / 16.0f;
        distanceRange.y = std::max(distanceRange.y, 0.005f);
    }
}

float2 CSM::calcDistanceRange(RenderContext* pRenderCtx, const Camera* pCamera, const Texture::SharedPtr& pDepthBuffer)
{
    if (mControls.useMinMaxSdsm)
    {
        reduceDepthSdsmMinMax(pRenderCtx, pCamera, pDepthBuffer);
        return mSdsmData.sdsmResult;
    }
    else
    {
        return mControls.distanceRange;
    }
}

void CSM::setDataIntoVars(ShaderVar const& globalVars, ShaderVar const& csmDataVar)
{
    switch ((CsmFilter)mCsmData.filterMode)
    {
    case CsmFilter::Point:
        csmDataVar["shadowMap"] = mShadowPass.pFbo->getDepthStencilTexture();
        globalVars["gCsmCompareSampler"] = mShadowPass.pPointCmpSampler;
        break;
    case CsmFilter::HwPcf:
    case CsmFilter::FixedPcf:
    case CsmFilter::StochasticPcf:
        csmDataVar["shadowMap"] = mShadowPass.pFbo->getDepthStencilTexture();
        globalVars["gCsmCompareSampler"] = mShadowPass.pLinearCmpSampler;
        break;
    case CsmFilter::Vsm:
    case CsmFilter::Evsm2:
    case CsmFilter::Evsm4:
        csmDataVar["shadowMap"] = mShadowPass.pFbo->getColorTexture(0);
        csmDataVar["csmSampler"] = mShadowPass.pVSMTrilinearSampler;
        break;
    }

    mCsmData.lightDir = glm::normalize(((DirectionalLight*)mpLight.get())->getWorldDirection());
    csmDataVar.setBlob(mCsmData);
}

void CSM::setupVisibilityPassFbo(const Texture::SharedPtr& pVisBuffer)
{
    Texture::SharedPtr pTex = mVisibilityPass.pFbo->getColorTexture(0);
    bool rebind = false;

    if (pVisBuffer && (pVisBuffer != pTex))
    {
        rebind = true;
        pTex = pVisBuffer;
    }
    else if (pTex == nullptr)
    {
        rebind = true;
        ResourceFormat format = getVisBufferFormat(mVisibilityPassData.mapBitsPerChannel, mVisibilityPassData.shouldVisualizeCascades);
        pTex = Texture::create2D(mVisibilityPassData.screenDim.x, mVisibilityPassData.screenDim.y, format, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
    }

    if (rebind) mVisibilityPass.pFbo->attachColorTarget(pTex, 0);
}

void CSM::execute(RenderContext* pContext, const RenderData& renderData)
{
    if (!mpLight || !mpScene) return;

    setupVisibilityPassFbo(renderData[kVisibility]->asTexture());
    const auto& pDepth = renderData[kDepth]->asTexture();
    const auto pCamera = mpScene->getCamera().get();
    //const auto pCamera = mpCsmSceneRenderer->getScene()->getActiveCamera().get();

    const float4 clearColor(0);
    pContext->clearFbo(mShadowPass.pFbo.get(), clearColor, 1, 0, FboAttachmentType::All);

    // Calc the bounds
    float2 distanceRange = calcDistanceRange(pContext, pCamera, pDepth);

    GraphicsState::Viewport VP;
    VP.originX = 0;
    VP.originY = 0;
    VP.minDepth = 0;
    VP.maxDepth = 1;
    VP.height = mShadowPass.mapSize.x;
    VP.width = mShadowPass.mapSize.y;

    //Set shadow pass state
    mShadowPass.pState->setViewport(0, VP);
    /*mpCsmSceneRenderer->setDepthClamp(mControls.depthClamp);*/
    partitionCascades(pCamera, distanceRange);
    renderScene(pContext);

    if ((CsmFilter)mCsmData.filterMode == CsmFilter::Vsm || (CsmFilter)mCsmData.filterMode == CsmFilter::Evsm2 || (CsmFilter)mCsmData.filterMode == CsmFilter::Evsm4)
    {
        mpBlurGraph->setInput(kBlurPass + ".src", mShadowPass.pFbo->getColorTexture(0));
        mpBlurGraph->execute(pContext);
        mShadowPass.pFbo->attachColorTarget(mpBlurGraph->getOutput(kBlurPass + ".dst")->asTexture(), 0);
        mShadowPass.pFbo->getColorTexture(0)->generateMips(pContext);
    }

    // Clear visibility buffer
    pContext->clearFbo(mVisibilityPass.pFbo.get(), float4(1, 0, 0, 0), 1, 0, FboAttachmentType::All);

    // Update Vars
    mVisibilityPass.pPass["gDepth"] = pDepth ? pDepth : mDepthPass.pState->getFbo()->getDepthStencilTexture();


    auto visibilityVars = mVisibilityPass.pPass->getVars().getRootVar();
    setDataIntoVars(visibilityVars, visibilityVars["PerFrameCB"]["gCsmData"]);
    mVisibilityPassData.camInvViewProj = pCamera->getInvViewProjMatrix();
    mVisibilityPassData.screenDim = uint2(mVisibilityPass.pFbo->getWidth(), mVisibilityPass.pFbo->getHeight());
    mVisibilityPass.pPass["PerFrameCB"][mVisibilityPass.mPassDataOffset].setBlob(mVisibilityPassData);

    // Render visibility buffer
    mVisibilityPass.pPass->execute(pContext, mVisibilityPass.pFbo);
}

void CSM::setLight(const Light::SharedConstPtr& pLight)
{
    mpLight = pLight;
    if (mpLight && mpLight->getType() != LightType::Directional)
    {
        setCascadeCount(1);
    }
}

void CSM::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    // auto alphaSampler = pDefaultBlock->getResourceBinding("alphaSampler");
    // auto alphaMapCB = pDefaultBlock->getResourceBinding("AlphaMapCB");
    // auto alphaMap = pDefaultBlock->getResourceBinding("alphaMap");

    mpScene = pScene;

    setLight(mpScene && mpScene->getLightCount() ? mpScene->getLight(0) : nullptr);

    if (mpScene)
    {
        mDepthPass.pProgram->addDefines(mpScene->getSceneDefines());
        mDepthPass.pVars = GraphicsVars::create(mDepthPass.pProgram->getReflector());

        mShadowPass.pProgram->addDefines(mpScene->getSceneDefines());
        mShadowPass.pVars = GraphicsVars::create(mShadowPass.pProgram->getReflector());

        const auto& pReflector = mShadowPass.pVars->getReflection();
        const auto& pDefaultBlock = pReflector->getDefaultParameterBlock();
        mPerLightCbLoc = pDefaultBlock->getResourceBinding("PerLightCB");
    }
    else
    {
        mDepthPass.pVars = nullptr;
        mShadowPass.pVars = nullptr;
        mPerLightCbLoc = {};
    }

    //         mpCsmSceneRenderer = CsmSceneRenderer::create(pScene, alphaMapCB, alphaMap, alphaSampler);
    //         mpCsmSceneRenderer->toggleMeshCulling(mCullMeshes);
    //         setLight(pScene && pScene->getLightCount() ? pScene->getLight(0) : nullptr);
}

void CSM::renderUI(Gui::Widgets& widget)
{
    if (mpLight && mpLight->getType() == LightType::Directional)
    {
        if (widget.var("Cascade Count", (int32_t&)mCsmData.cascadeCount, 1, 8))
        {
            setCascadeCount(mCsmData.cascadeCount);
        }
    }

    // Shadow-map size
    int2 smDims = int2(mShadowPass.pFbo->getWidth(), mShadowPass.pFbo->getHeight());
    if (widget.var("Shadow-Map Size", smDims, 0, 8192)) resizeShadowMap(smDims);

    // Visibility buffer bits-per channel
    static const Gui::DropdownList visBufferBits =
    {
        {8, "8"},
        {16, "16"},
        {32, "32"}
    };
    if (widget.dropdown("Visibility Buffer Bits-Per-Channel", visBufferBits, mVisibilityPassData.mapBitsPerChannel)) setVisibilityBufferBitsPerChannel(mVisibilityPassData.mapBitsPerChannel);

    // Mesh culling
    if (widget.checkbox("Cull Meshes", mCullMeshes)) toggleMeshCulling(mCullMeshes);

    //Filter mode
    uint32_t filterIndex = static_cast<uint32_t>(mCsmData.filterMode);
    if (widget.dropdown("Filter Mode", kFilterList, filterIndex))
    {
        setFilterMode(filterIndex);
    }

    //partition mode
    uint32_t newPartitionMode = static_cast<uint32_t>(mControls.partitionMode);

    if (widget.dropdown("Partition Mode", kPartitionList, newPartitionMode))
    {
        setPartitionMode(newPartitionMode);
    }

    if (mControls.partitionMode == PartitionMode::PSSM)
    {
        widget.var("PSSM Lambda", mControls.pssmLambda, 0.f, 1.0f);
    }

    if (mControls.useMinMaxSdsm == false)
    {
        widget.var("Min Distance", mControls.distanceRange.x, 0.f, 1.f, 0.001f);
        widget.var("Max Distance", mControls.distanceRange.y, 0.f, 1.f, 0.001f);
    }

    widget.var("Cascade Blend Threshold", mCsmData.cascadeBlendThreshold, 0.f, 1.0f, 0.001f);
    widget.checkbox("Depth Clamp", mControls.depthClamp);

    widget.var("Depth Bias", mCsmData.depthBias, 0.f, FLT_MAX, 0.0001f);
    widget.checkbox("Stabilize Cascades", mControls.stabilizeCascades);

    // SDSM data
    if (auto sdsmGroup = widget.group("SDSM MinMax"))
    {
        sdsmGroup.checkbox("Enable", mControls.useMinMaxSdsm);
        if (mControls.useMinMaxSdsm)
        {
            int32_t latency = mSdsmData.readbackLatency;
            if (sdsmGroup.var("Readback Latency", latency, 0))
            {
                setSdsmReadbackLatency(latency);
            }
            std::string range = "SDSM Range=[" + std::to_string(mSdsmData.sdsmResult.x) + ", " + std::to_string(mSdsmData.sdsmResult.y) + ']';
            sdsmGroup.text(range.c_str());
        }
    }


    if ((CsmFilter)mCsmData.filterMode == CsmFilter::FixedPcf || (CsmFilter)mCsmData.filterMode == CsmFilter::StochasticPcf)
    {
        int32_t kernelWidth = mCsmData.pcfKernelWidth;
        if (widget.var("Kernel Width", kernelWidth, 1, 15, 2))
        {
            setPcfKernelWidth(kernelWidth);
        }
    }

    //VSM/ESM
    if ((CsmFilter)mCsmData.filterMode == CsmFilter::Vsm || (CsmFilter)mCsmData.filterMode == CsmFilter::Evsm2 || (CsmFilter)mCsmData.filterMode == CsmFilter::Evsm4)
    {
        if (auto vsmGroup = widget.group("VSM/EVSM"))
        {
            uint32_t newMaxAniso = mShadowPass.pVSMTrilinearSampler->getMaxAnisotropy();
            vsmGroup.dropdown("Max Aniso", kMaxAniso, newMaxAniso);
            {
                createVsmSampleState(newMaxAniso);
            }

            vsmGroup.var("Light Bleed Reduction", mCsmData.lightBleedingReduction, 0.f, 1.0f, 0.01f);

            if ((CsmFilter)mCsmData.filterMode == CsmFilter::Evsm2 || (CsmFilter)mCsmData.filterMode == CsmFilter::Evsm4)
            {
                if (auto evsmExpGroup = vsmGroup.group("EVSM Exp"))
                {
                    evsmExpGroup.var("Positive", mCsmData.evsmExponents.x, 0.0f, 5.54f, 0.01f);
                    evsmExpGroup.var("Negative", mCsmData.evsmExponents.y, 0.0f, 5.54f, 0.01f);
                }
            }

            if (auto blurGroup = vsmGroup.group("Blur Settings"))
            {
                mpBlurGraph->getPass(kBlurPass)->renderUI(blurGroup);
            }
        }
    }
}

void CSM::onResize(uint32_t width, uint32_t height)
{
    mVisibilityPassData.screenDim = { width, height };
    mVisibilityPass.pFbo->attachColorTarget(nullptr, 0);
}

void CSM::setSdsmReadbackLatency(uint32_t latency)
{
    if (mSdsmData.readbackLatency != latency)
    {
        mSdsmData.readbackLatency = latency;
        mSdsmData.minMaxReduction = nullptr;
    }
}

void CSM::createSdsmData(Texture::SharedPtr pTexture)
{
    assert(pTexture);
    // Only create a new technique if it doesn't exist or the dimensions changed
    if (mSdsmData.minMaxReduction)
    {
        if (mSdsmData.width == pTexture->getWidth() && mSdsmData.height == pTexture->getHeight() && mSdsmData.sampleCount == pTexture->getSampleCount())
        {
            return;
        }
    }
    mSdsmData.width = pTexture->getWidth();
    mSdsmData.height = pTexture->getHeight();
    mSdsmData.sampleCount = pTexture->getSampleCount();
    mSdsmData.minMaxReduction = ParallelReduction::create(ParallelReduction::Type::MinMax, mSdsmData.readbackLatency, mSdsmData.width, mSdsmData.height, mSdsmData.sampleCount);
}

void CSM::setCascadeCount(uint32_t cascadeCount)
{
    if (mpLight && (mpLight->getType() != LightType::Directional) && (cascadeCount != 1))
    {
        logWarning("CSM::setCascadeCount() - cascadeCount for a non-directional light must be 1");
        cascadeCount = 1;
    }

    if (mCsmData.cascadeCount != cascadeCount)
    {
        mCsmData.cascadeCount = cascadeCount;
        createShadowPassResources();
    }
}

void CSM::setFilterMode(uint32_t newFilterMode)
{
    mCsmData.filterMode = newFilterMode;
    createShadowPassResources();
}

void CSM::setVisibilityBufferBitsPerChannel(uint32_t bitsPerChannel)
{
    if (bitsPerChannel != 8 && bitsPerChannel != 16 && bitsPerChannel != 32)
    {
        logWarning("CSM::setVisibilityBufferBitsPerChannel() - bitsPerChannel must by 8, 16 or 32");
        return;
    }
    mVisibilityPassData.mapBitsPerChannel = bitsPerChannel;
    mVisibilityPass.pFbo->attachColorTarget(nullptr, 0);
    mPassChangedCB();
}

void CSM::resizeShadowMap(const uint2& smDims)
{
    mMapSize = smDims;
    createShadowPassResources();
}

void CSM::toggleMeshCulling(bool enabled)
{
    mCullMeshes = enabled;
}

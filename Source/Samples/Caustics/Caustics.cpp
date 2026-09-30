/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Caustics.h"
#include "Utils/UI/TextRenderer.h"
#include "Utils/Math/FalcorMath.h"
#include <fstream>

FALCOR_EXPORT_D3D12_AGILITY_SDK

uint32_t mSampleGuiWidth = 250u;
uint32_t mSampleGuiHeight = 200u;
uint32_t mSampleGuiPositionX = 20u;
uint32_t mSampleGuiPositionY = 40u;

static const float4 kClearColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
static const std::string kDefaultScene = "Caustics/Data/ring.pyscene";

const char kPhotonTraceShaderFilename[] = "Samples/Caustics/PhotonTrace.rt.hlsl";
const char kPhotonScatter3dShaderFilename[] = "Samples/Caustics/PhotonScatter.3d.hlsl";

const char kCausticsRtShaderFilename[] = "Samples/Caustics/CausticsRT.rt.hlsl";
const char kCaustics3dShaderFilename[] = "Samples/Caustics/CausticsRaster.ps.hlsl";

const char kGBuffer3dShaderFilename[] = "Samples/Caustics/GBufferRaster.ps.hlsl";

const char kCompositeRtShaderFilename[] = "Samples/Caustics/CompositeRT.rt.hlsl";
const char kComposite3dShaderFilename[] = "Samples/Caustics/CompositeRaster.ps.hlsl";

std::string to_string(const float3& v)
{
    std::string s;
    s += "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
    return s;
}

void Caustics::onLoad(RenderContext* pRenderContext)
{
    if (getDevice()->isFeatureSupported(Device::SupportedFeatures::Raytracing) == false) throw RuntimeError("Device does not support raytracing!");

    loadScene(kDefaultScene, getTargetFbo().get());
    loadSceneSetting("Samples/Caustics/Data/init.ini");
    loadShader();
}

void Caustics::onShutdown() {}

Caustics::Raytracer Caustics::getPhotonTraceShader()
{
    // Get shader modules and type conformances for types used by the scene.
    // These need to be set on the program in order to fully use Falcor's material system.
    const auto& shaderModules = mpScene->getShaderModules();
    const auto& typeConformances = mpScene->getTypeConformances(); //const auto& globaltypeConformances = mpScene->getMaterialSystem().getTypeConformances();
    // Get scene defines. These need to be set on any program using the scene.
    const auto& sceneDefines = mpScene->getSceneDefines();

    uint flag = photonMacroToFlags();
    auto pIter = mPhotonTraceShaderList.find(flag);
    if (pIter == mPhotonTraceShaderList.end())
    {
        ProgramDesc rtProgDesc;
        rtProgDesc.addShaderModules(shaderModules);
        rtProgDesc.addShaderLibrary(kPhotonTraceShaderFilename);
        /*rtProgDesc.addRayGen("rayGen");
        rtProgDesc.addHitGroup("primaryClosestHit");
        rtProgDesc.addMiss("primaryMiss");*/
        rtProgDesc.addTypeConformances(typeConformances);

        DefineList rtProgDefineList;
        
        switch (mPhotonTraceMacro)
        {
        case Caustics::RAY_DIFFERENTIAL:
            rtProgDefineList.add("RAY_DIFFERENTIAL", "1");
            break;
        case Caustics::RAY_CONE:
            rtProgDefineList.add("RAY_CONE", "1");
            break;
        case Caustics::RAY_NONE:
            rtProgDefineList.add("RAY_NONE", "1");
            break;
        default:
            break;
        }
        switch (mTraceType)
        {
        case Caustics::TRACE_FIXED:
            rtProgDefineList.add("TRACE_FIXED", "1");
            break;
        case Caustics::TRACE_ADAPTIVE:
            rtProgDefineList.add("TRACE_ADAPTIVE", "1");
            break;
        case Caustics::TRACE_NONE:
            rtProgDefineList.add("TRACE_NONE", "1");
            break;
        case Caustics::TRACE_ADAPTIVE_RAY_MIP_MAP:
            rtProgDefineList.add("TRACE_ADAPTIVE_RAY_MIP_MAP", "1");
            break;
        default:
            break;
        }
        if (mFastPhotonPath)
        {
            rtProgDefineList.add("FAST_PHOTON_PATH", "1");
        }
        if (mShrinkColorPayload)
        {
            rtProgDefineList.add("SMALL_COLOR", "1");
        }
        if (mShrinkRayDiffPayload)
        {
            rtProgDefineList.add("SMALL_RAY_DIFFERENTIAL", "1");
        }
        if (mUpdatePhoton)
        {
            rtProgDefineList.add("UPDATE_PHOTON", "1");
        }
        rtProgDefineList.add(sceneDefines);

        uint payLoadSize = 80U;
        if (mShrinkColorPayload) payLoadSize -= 12U;
        if (mShrinkRayDiffPayload) payLoadSize -= 24U;

        rtProgDesc.maxPayloadSize = payLoadSize;
        rtProgDesc.maxTraceRecursionDepth = 1u;

        mPhotonTraceShaderList[flag].pBindingTable = RtBindingTable::create(2u, 2u, mpScene->getGeometryCount());
        mPhotonTraceShaderList[flag].pBindingTable->setRayGen(rtProgDesc.addRayGen("rayGen"));
        mPhotonTraceShaderList[flag].pBindingTable->setHitGroup(0u, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), rtProgDesc.addHitGroup("primaryClosestHit"));
        mPhotonTraceShaderList[flag].pBindingTable->setMiss(0u, rtProgDesc.addMiss("primaryMiss"));
        //mPhotonTraceShaderList[flag].pBindingTable->setHitGroup(1u, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), rtProgDesc.addHitGroup("", "shadowAnyHit"));
        //mPhotonTraceShaderList[flag].pBindingTable->setMiss(1u, rtProgDesc.addMiss("shadowMiss"));

        mPhotonTraceShaderList[flag].pProgram = Program::create(getDevice(), rtProgDesc, rtProgDefineList); //auto pPhotonTraceProgram = Program::create(getDevice(), rtProgDesc, payLoadSize, 8U);
        mPhotonTraceShaderList[flag].pProgramVars = RtProgramVars::create(getDevice(), mPhotonTraceShaderList[flag].pProgram, mPhotonTraceShaderList[flag].pBindingTable);
    }
    return mPhotonTraceShaderList[flag];
}

void Caustics::onResize(uint32_t width, uint32_t height)
{
    float h = (float)height;
    float w = (float)width;

    if (mpCamera)
    {
        mpCamera->setFocalLength(18);
        float aspectRatio = (w / h);
        mpCamera->setAspectRatio(aspectRatio);
    }

    ref<Program> photonTraceProgram = mPhotonTraceShaderList.begin()->second.pProgram;
    auto photonTraceVars = mPhotonTraceShaderList.begin()->second.pProgramVars->getRootVar();

    auto analyseVar = mpAnalyseVars->getRootVar();
    mpRayTaskBuffer = getDevice()->createStructuredBuffer(analyseVar["gRayTask"], MAX_PHOTON_COUNT, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    auto updateRayDensityVar = mpUpdateRayDensityVars->getRootVar();
    mpPixelInfoBuffer = getDevice()->createStructuredBuffer(updateRayDensityVar["gPixelInfo"], MAX_CAUSTICS_MAP_SIZE * MAX_CAUSTICS_MAP_SIZE,
        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    // mpPixelInfoBufferDisplay = getDevice()->createStructuredBuffer(mpUpdateRayDensityProgram.get(), std::string("gPixelInfo"),
    // CAUSTICS_MAP_SIZE * CAUSTICS_MAP_SIZE,
    // ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    mpPhotonBuffer = getDevice()->createStructuredBuffer(photonTraceVars["gPhotonBuffer"], MAX_PHOTON_COUNT,
        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    mpPhotonBuffer2 = getDevice()->createStructuredBuffer(photonTraceVars["gPhotonBuffer"], MAX_PHOTON_COUNT,
        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    auto drawArgVar = mpDrawArgumentVars->getRootVar();
    mpDrawArgumentBuffer = getDevice()->createStructuredBuffer(drawArgVar["gDrawArgument"], 1u, 
        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::IndirectArg | ResourceBindFlags::ShaderResource);

    mpRayArgumentBuffer = getDevice()->createStructuredBuffer(drawArgVar["gRayArgument"], 1, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::IndirectArg);

    auto genRayCountVar = mpGenerateRayCountVars->getRootVar();
    mpRayCountQuadTree = getDevice()->createStructuredBuffer(genRayCountVar["gRayCountQuadTree"], MAX_CAUSTICS_MAP_SIZE * MAX_CAUSTICS_MAP_SIZE * 2,
        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    mpRtOut = getDevice()->createTexture2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    int2 tileDim((mpRtOut->getWidth() + mTileSize.x - 1) / mTileSize.x, (mpRtOut->getHeight() + mTileSize.y - 1) / mTileSize.y);
    int avgTileIDCount = 63356; //?

    auto allocTileVar = mpAllocateTileVars[0]->getRootVar();

    mpTileIDInfoBuffer = getDevice()->createStructuredBuffer(allocTileVar["gTileInfo"], tileDim.x * tileDim.y,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    mpIDBuffer = getDevice()->createBuffer(tileDim.x * tileDim.y * avgTileIDCount * sizeof(uint32_t),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal);

    mpIDCounterBuffer = getDevice()->createBuffer(sizeof(uint32_t), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal);

    createCausticsMap();

    mpRayDensityTex = getDevice()->createTexture2D(MAX_CAUSTICS_MAP_SIZE, MAX_CAUSTICS_MAP_SIZE, ResourceFormat::RGBA16Float, 1, 1, nullptr,
        ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    mpPhotonCountTex = getDevice()->createTexture1D(width, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    createGBuffer(width, height, mGBuffer[0]);
    createGBuffer(width, height, mGBuffer[1]);
}

void Caustics::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    pRenderContext->clearFbo(pTargetFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);

    if (mpScene)
    {
        mCamController->update();
        if (mRayTrace)
            renderRT(pRenderContext, pTargetFbo);
        else
            renderRaster(pRenderContext, pTargetFbo);
    }

    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
}

const FileDialogFilterVec settingFilter = {{"ini", "Scene Setting File"}};

void Caustics::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Caustics", {300, 400}, {10, 80});
    renderGlobalUI(pGui);
    w.text("Hello from Caustics");

    w.checkbox("Ray Trace", mRayTrace);

    if (w.button("Load Scene"))
    {
        std::filesystem::path filePath;
        if (openFileDialog({}, filePath))
        {
            loadScene(filePath.generic_string(), getTargetFbo().get());
            loadShader();
        }
    }

    if (w.button("Load Scene Settings"))
    {
        // std::string filename;
        std::filesystem::path filePath;
        if (openFileDialog(settingFilter, filePath))
        {
            loadSceneSetting(filePath.generic_string());
        }
    }
    if (w.button("Save Scene Settings"))
    {
        // std::string filename;
        std::filesystem::path filePath;
        if (saveFileDialog(settingFilter, filePath))
        {
            saveSceneSetting(filePath.generic_string());
        }
    }
    if (w.button("Update Shader"))
    {
        loadShader();
    }

    // if (w.group("Display", true))
    {
        auto g = w.group("Display", true);
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({0, "Rasterize"});
            debugModeList.push_back({1, "Depth"});
            debugModeList.push_back({2, "Normal"});
            debugModeList.push_back({3, "Diffuse"});
            debugModeList.push_back({4, "Specular"});
            debugModeList.push_back({5, "Photon"});
            debugModeList.push_back({6, "World"});
            debugModeList.push_back({7, "Roughness"});
            debugModeList.push_back({8, "Ray Info"});
            debugModeList.push_back({9, "Raytrace"});
            debugModeList.push_back({10, "Avg. Screen Area"});
            debugModeList.push_back({11, "Screen Area Std. Variance"});
            debugModeList.push_back({12, "Photon Count"});
            debugModeList.push_back({13, "Photon Total Count"});
            debugModeList.push_back({14, "Ray count Mipmap"});
            debugModeList.push_back({15, "Photon Density"});
            debugModeList.push_back({16, "Small Photon Color"});
            debugModeList.push_back({17, "Small Photon Count"});
            g.dropdown("Composite mode", debugModeList, (uint32_t&)mDebugMode);
        }
        g.var("Max Pixel Value", mMaxPixelArea, 0.0f, 1000000000.f, 5.f);
        g.var("Max Photon Count", mMaxPhotonCount, 0.f, 1000000000.f, 5.f);
        g.var("Ray Count Mipmap", mRayCountMipIdx, 0, 11);
        // pGui->addFloatVar("Max Pixel Value", mMaxPixelArea, 0, 1000000000, 5.f);
        // pGui->addFloatVar("Max Photon Count", mMaxPhotonCount, 0, 1000000000, 5.f);
        // pGui->addIntVar("Ray Count Mipmap", mRayCountMipIdx, 0, 11);
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({1, "x1"});
            debugModeList.push_back({2, "x2"});
            debugModeList.push_back({4, "x4"});
            debugModeList.push_back({8, "x8"});
            debugModeList.push_back({16, "x16"});
            g.dropdown("Ray Tex Scale", debugModeList, (uint32_t&)mRayTexScaleFactor);
        }
        // pGui->endGroup();
    }

    // if (pGui->beginGroup("Photon Trace", true))
    {
        auto g = w.group("Photon Trace", true);
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({0, "Fixed Resolution"});
            debugModeList.push_back({1, "Adaptive Resolution"});
            debugModeList.push_back({3, "Fast Adaptive Resolution"});
            debugModeList.push_back({2, "None"});
            g.dropdown("Trace Type", debugModeList, (uint32_t&)mTraceType);
        }
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({0, "Ray Differential"});
            debugModeList.push_back({1, "Ray Cone"});
            debugModeList.push_back({2, "None"});
            g.dropdown("Ray Type", debugModeList, (uint32_t&)mPhotonTraceMacro);
        }
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({64, "64"});
            debugModeList.push_back({128, "128"});
            debugModeList.push_back({256, "256"});
            debugModeList.push_back({512, "512"});
            debugModeList.push_back({1024, "1024"});
            debugModeList.push_back({2048, "2048"});
            g.dropdown("Dispatch Size", debugModeList, (uint32_t&)mDispatchSize);
        }
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({0, "Avg Square"});
            debugModeList.push_back({1, "Avg Length"});
            debugModeList.push_back({2, "Max Square"});
            debugModeList.push_back({3, "Exact Area"});
            g.dropdown("Area Type", debugModeList, (uint32_t&)mAreaType);
        }
        g.var("Intensity", mIntensity, 0.f, 10.f, 0.1f);
        g.var("Emit size", mEmitSize, 0.f, 1000.f, 1.f);
        g.var("Rough Threshold", mRoughThreshold, 0.f, 1.f, 0.01f);
        g.var("Max Trace Depth", mMaxTraceDepth, 0, 30);
        g.var("IOR Override", mIOROveride, 0.f, 3.f, 0.01f);
        g.checkbox("ID As Color", mColorPhoton);
        g.var("Photon ID Scale", mPhotonIDScale);
        g.var("Min Trace Luminance", mTraceColorThreshold, 0.f, 10.f, 0.005f);
        g.var("Min Cull Luminance", mCullColorThreshold, 0.f, 10000.f, 0.01f);
        g.checkbox("Fast Photon Path", mFastPhotonPath);
        g.var("Max Pixel Radius", mMaxPhotonPixelRadius, 0.f, 5000.f, 1.f);
        g.var("Fast Pixel Radius", mFastPhotonPixelRadius, 0.f, 5000.f, 1.f);
        g.var("Fast Draw Count", mFastPhotonDrawCount, 0.f, 50000.f, 0.1f);
        g.var("Color Compress Scale", mSmallPhotonCompressScale, 0.f, 5000.f, 1.f);
        g.checkbox("Shrink Color Payload", mShrinkColorPayload);
        g.checkbox("Shrink Ray Diff Payload", mShrinkRayDiffPayload);
        g.checkbox("Update Photon", mUpdatePhoton);
        // pGui->endGroup();
    }

    // if (pGui->beginGroup("Adaptive Resolution", true))
    {
        auto g = w.group("Adaptive Resolution", true);
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({0, "Random"});
            debugModeList.push_back({1, "Grid"});
            g.dropdown("Sample Placement", debugModeList, (uint32_t&)mSamplePlacement);
        }
        g.var("Luminance Threshold", mPixelLuminanceThreshold, 0.01f, 10.0f, 0.01f);
        g.var("Photon Size Threshold", mMinPhotonPixelSize, 1.f, 1000.0f, 0.1f);
        g.var("Smooth Weight", mSmoothWeight, 0.f, 10.0f, 0.001f);
        g.var("Proportional Gain", mUpdateSpeed, 0.f, 1.f, 0.01f);
        g.var("Variance Gain", mVarianceGain, 0.f, 10.f, 0.0001f);
        g.var("Derivative Gain", mDerivativeGain, -10.f, 10.f, 0.1f);
        g.var("Max Task Per Pixel", mMaxTaskCountPerPixel, 1.0f, 1000000.f, 5.f);
        // pGui->endGroup();
    }

    // if (pGui->beginGroup("Smooth Photon", false))
    {
        auto g = w.group("Smooth Photon", true);
        g.checkbox("Remove Isolated Photon", mRemoveIsolatedPhoton);
        g.checkbox("Enable Median Filter", mMedianFilter);
        g.var("Normal Threshold", mNormalThreshold, 0.01f, 1.0f, 0.01f);
        g.var("Distance Threshold", mDistanceThreshold, 0.1f, 100.0f, 0.1f);
        g.var("Planar Threshold", mPlanarThreshold, 0.01f, 10.0f, 0.1f);
        g.var("Trim Direction Threshold", trimDirectionThreshold, 0.f, 1.f);
        g.var("Min Neighbour Count", mMinNeighbourCount, 0, 8);
        // pGui->endGroup();
    }

    // if (pGui->beginGroup("Photon Splatting", true))
    {
        auto g = w.group("Photon Splatting", true);
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({0, "Scatter"});
            debugModeList.push_back({1, "Gather"});
            debugModeList.push_back({2, "None"});
            g.dropdown("Density Estimation", debugModeList, (uint32_t&)mScatterOrGather);
        }
        g.var("Splat size", mSplatSize, 0.f, 100.f, 0.01f);
        g.var("Kernel Power", mKernelPower, 0.01f, 10.f, 0.01f);

        // if(pGui->beginGroup("Scatter Parameters", false))
        {
            auto g2 = w.group("Scatter Parameters", true);
            g2.var("Z Tolerance", mZTolerance, 0.001f, 1.f, 0.001f);
            g2.var("Scatter Normal Threshold", mScatterNormalThreshold, 0.01f, 1.0f, 0.01f);
            g2.var("Scatter Distance Threshold", mScatterDistanceThreshold, 0.1f, 10.0f, 0.1f);
            g2.var("Scatter Planar Threshold", mScatterPlanarThreshold, 0.01f, 10.0f, 0.1f);
            g2.var("Max Anisotropy", mMaxAnisotropy, 1.f, 100.f, 0.1f);
            {
                Gui::DropdownList debugModeList;
                debugModeList.push_back({0, "Quad"});
                debugModeList.push_back({1, "Sphere"});
                g2.dropdown("Photon Geometry", debugModeList, (uint32_t&)mScatterGeometry);
            }
            {
                Gui::DropdownList debugModeList;
                debugModeList.push_back({0, "Kernel"});
                debugModeList.push_back({1, "Solid"});
                debugModeList.push_back({2, "Shaded"});
                g2.dropdown("Photon Display Mode", debugModeList, (uint32_t&)mPhotonDisplayMode);
            }

            {
                Gui::DropdownList debugModeList;
                debugModeList.push_back({0, "Anisotropic"});
                debugModeList.push_back({1, "Isotropic"});
                debugModeList.push_back({2, "Photon Mesh"});
                debugModeList.push_back({3, "Screen Dot"});
                debugModeList.push_back({4, "Screen Dot With Color"});
                g2.dropdown("Photon mode", debugModeList, (uint32_t&)mPhotonMode);
            }
            // pGui->endGroup();
        }

        // if(pGui->beginGroup("Gather Parameters", false))
        {
            auto g2 = w.group("Gather Parameters", true);
            g2.var("Gather Depth Radius", mDepthRadius, 0.f, 10.f, 0.01f);
            g2.var("Gather Min Color", mMinGatherColor, 0.f, 2.f, 0.001f);
            g2.checkbox("Gather Show Tile Count", mShowTileCount);
            g2.var("Gather Tile Count Scale", mTileCountScale, 0, 1000);
            // pGui->endGroup();
        }

        // pGui->endGroup();
    }

    // if (pGui->beginGroup("Temporal Filter", true))
    {
        auto g = w.group("Temporal Filter", true);
        g.checkbox("Enable Temporal Filter", mTemporalFilter);
        g.var("Filter Weight", mFilterWeight, 0.0f, 1.0f, 0.001f);
        g.var("Jitter", mJitter, 0.f, 10.f, 0.01f);
        g.var("Jitter Power", mJitterPower, 0.f, 200.f, 0.01f);
        g.var("Temporal Normal Strength", mTemporalNormalKernel, 0.0001f, 1000.f, 0.01f);
        g.var("Temporal Depth Strength", mTemporalDepthKernel, 0.0001f, 1000.f, 0.01f);
        g.var("Temporal Color Strength", mTemporalColorKernel, 0.0001f, 1000.f, 0.01f);
        // pGui->endGroup();
    }

    // if (pGui->beginGroup("Spacial Filter", true))
    {
        auto g = w.group("Spacial Filter", true);
        g.checkbox("Enable Spatial Filter", mSpacialFilter);
        g.var("A trous Pass", mSpacialPasses, 0, 10);
        g.var("Spacial Normal Strength", mSpacialNormalKernel, 0.0001f, 100.f, 0.01f);
        g.var("Spacial Depth Strength", mSpacialDepthKernel, 0.0001f, 100.f, 0.01f);
        g.var("Spacial Color Strength", mSpacialColorKernel, 0.0001f, 100.f, 0.01f);
        g.var("Spacial Screen Kernel", mSpacialScreenKernel, 0.0001f, 100.f, 0.01f);
        // pGui->endGroup();
    }

    // if (pGui->beginGroup("Composite", true))
    {
        auto g = w.group("Composite", true);
        {
            int oldResRatio = mCausticsMapResRatio;
            Gui::DropdownList debugModeList;
            debugModeList.push_back({1, "x 1"});
            debugModeList.push_back({2, "x 1/2"});
            debugModeList.push_back({4, "x 1/4"});
            debugModeList.push_back({8, "x 1/8"});
            g.dropdown("Caustics Resolution", debugModeList, (uint32_t&)mCausticsMapResRatio);
            if (oldResRatio != mCausticsMapResRatio)
            {
                createCausticsMap();
            }
        }
        g.checkbox("Filter Caustics Map", mFilterCausticsMap);
        g.var("UV Kernel", mUVKernel, 0.0f, 1000.f, 0.1f);
        g.var("Depth Kernel", mZKernel, 0.0f, 1000.f, 0.1f);
        g.var("Normal Kernel", mNormalKernel, 0.0f, 1000.f, 0.1f);
        // pGui->endGroup();
    }

    mLightDirection = float3(cos(mLightAngle.x) * sin(mLightAngle.y), cos(mLightAngle.y), sin(mLightAngle.x) * sin(mLightAngle.y));
    // if (pGui->beginGroup("Light", true))
    {
        auto g = w.group("Light", true);
        g.var("Light Angle", mLightAngle, -FLT_MAX, FLT_MAX, 0.01f);
        if (mpScene)
        {
            if (mpScene->getLightCount() != 0u)
            {
                auto light0 = dynamic_cast<DirectionalLight*>(mpScene->getLight(0).get());
                light0->setWorldDirection(mLightDirection);
            }
        }
        g.var("Light Angle Speed", mLightAngleSpeed, -FLT_MAX, FLT_MAX, 0.001f);
        mLightAngle += mLightAngleSpeed * 0.01f;
        // pGui->endGroup();
    }

    // if (pGui->beginGroup("Camera"))
    {
        auto g = w.group("Camera", true);
        mpCamera->renderUI(w);
        // pGui->endGroup();
    }
}

bool Caustics::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (mCamController->onKeyEvent(keyEvent))
    {
        return true;
    }
    if (keyEvent.key == Input::Key::Space && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        mRayTrace = !mRayTrace;
        return true;
    }
    return false;
}

bool Caustics::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mCamController->onMouseEvent(mouseEvent);
}

void Caustics::onHotReload(HotReloadFlags reloaded)
{
    //
}

void Caustics::setPerFrameVars(RenderContext* pRenderContext, const Fbo* pTargetFbo)
{
    FALCOR_PROFILE(pRenderContext, "setPerFrameVars");
    {
        auto var = mpCausticsTracer.pProgramVars->getRootVar();
        auto pCB = var["PerFrameCB"];
        pCB["invView"] = inverse(mpCamera->getViewMatrix());
        pCB["viewportDims"] = float2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
        float fovY = focalLengthToFovY(mpCamera->getFocalLength(), Camera::kDefaultFrameHeight);
        pCB["tanHalfFovY"] = std::tanf(fovY * 0.5f);
        pCB["sampleIndex"] = mSampleIndex;
        pCB["useDOF"] = mUseDOF;
    }
    // setCommonVars(mpCausticsTracer.pProgramVars->getGlobalVars().get(), pTargetFbo);

    mSampleIndex++;
}

void Caustics::loadScene(const std::string& filename, const Fbo* pTargetFbo)
{
    // mpScene = RtScene::loadFromFile(filename, RtBuildFlags::None, Model::LoadFlags::None);
    mpScene = Scene::create(getDevice(), filename);
    FALCOR_ASSERT(mpScene);
    const uint32_t geometryCount = mpScene->getGeometryCount();

    mpCamera = mpScene->getCamera();
    assert(mpCamera);

    mpScene->setCameraController(Scene::CameraControllerType::FirstPerson);
    mCamController = std::make_unique<FirstPersonCameraController>(mpCamera);

    // Update the controllers
    float radius = mpScene->getSceneBounds().radius();
    mpScene->setCameraSpeed(radius * 0.25f);
    float nearZ = std::max(0.1f, radius / 750.0f);
    float farZ = radius * 10;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)pTargetFbo->getWidth() / (float)pTargetFbo->getHeight());

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    ref<Sampler> pSampler = getDevice()->createSampler(samplerDesc);

    mpScene->setDefaultTextureSampler(pSampler);

    mpQuad = Scene::create(getDevice(), "Caustics/quad.obj");
    mpSphere = Scene::create(getDevice(), "Caustics/sphere.obj");
    mpGaussianKernel = Texture::createFromFile(getDevice(), "D:/MyDocuments/VSPrograms/Direct3D/Falcor/media/Caustics/gaussian.png", true, false);
    mpUniformNoise = Texture::createFromFile(getDevice(), "D:/MyDocuments/VSPrograms/Direct3D/Falcor/media/Caustics/uniform.png", true, false);
}

void Caustics::loadSceneSetting(std::string path)
{
    std::ifstream file(path, std::ios::in);
    if (!file) return;

    file >> mLightAngle.x >> mLightAngle.y;

    float3 camOri, camTarget;
    file >> camOri.x >> camOri.y >> camOri.z;
    file >> camTarget.x >> camTarget.y >> camTarget.z;
    mpCamera->setPosition(camOri);
    mpCamera->setTarget(camTarget);
}

void Caustics::loadShader()
{
    const auto& shaderModules = mpScene->getShaderModules();
    const auto& typeConformances = mpScene->getTypeConformances();
    const auto& sceneDefines = mpScene->getSceneDefines();

#if defined(DEBUG) || defined(_DEBUG)
    printf("Scene loaded with %llu material conformances.\n", typeConformances.size());
    for (auto& c : typeConformances)
    {
        printf("  Found implementation: %s\n", c.first.typeName.c_str());
    }
#endif
    // caustics rt
    {
        ProgramDesc rtProgDesc;
        rtProgDesc.addShaderModules(shaderModules);
        rtProgDesc.addShaderLibrary(kCausticsRtShaderFilename);

        rtProgDesc.setMaxTraceRecursionDepth(3u);
        rtProgDesc.addTypeConformances(typeConformances);
        rtProgDesc.setMaxPayloadSize(24u);

        mpCausticsTracer.pBindingTable = RtBindingTable::create(2u, 2u, mpScene->getGeometryCount());
        mpCausticsTracer.pBindingTable->setRayGen(rtProgDesc.addRayGen("rayGen"));
        mpCausticsTracer.pBindingTable->setMiss(0u, rtProgDesc.addMiss("primaryMiss"));
        mpCausticsTracer.pBindingTable->setMiss(1u, rtProgDesc.addMiss("shadowMiss"));
        mpCausticsTracer.pBindingTable->setHitGroup(0u, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), rtProgDesc.addHitGroup("primaryClosestHit", ""));
        mpCausticsTracer.pBindingTable->setHitGroup(1u, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), rtProgDesc.addHitGroup("", "shadowAnyHit"));

        mpCausticsTracer.pProgram = Program::create(getDevice(), rtProgDesc, sceneDefines);
        mpCausticsTracer.pProgramVars = RtProgramVars::create(getDevice(), mpCausticsTracer.pProgram, mpCausticsTracer.pBindingTable);
    }

    // clear draw argument program
    mpDrawArgumentProgram = Program::createCompute(getDevice(), "Samples/Caustics/ResetDrawArgument.cs.hlsl", "main");
    mpDrawArgumentState = ComputeState::create(getDevice());
    mpDrawArgumentState->setProgram(mpDrawArgumentProgram);
    mpDrawArgumentVars = ProgramVars::create(getDevice(), mpDrawArgumentProgram.get());

    // photon trace
    mPhotonTraceShaderList.clear();
    getPhotonTraceShader();

    // composite rt
    {
        ProgramDesc rtProgDesc;
        rtProgDesc.addShaderModules(shaderModules);
        rtProgDesc.addShaderLibrary("Samples/Caustics/CompositeRT.rt.hlsl");
        rtProgDesc.setMaxTraceRecursionDepth(3u);
        rtProgDesc.addTypeConformances(typeConformances);
        rtProgDesc.setMaxPayloadSize(48u);

        mpCompositeTracer.pBindingTable = RtBindingTable::create(2u, 2u, mpScene->getGeometryCount());
        mpCompositeTracer.pBindingTable->setRayGen(rtProgDesc.addRayGen("rayGen"));
        mpCompositeTracer.pBindingTable->setHitGroup(0u, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), rtProgDesc.addHitGroup("primaryClosestHit", ""));
        mpCompositeTracer.pBindingTable->setHitGroup(1u, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), rtProgDesc.addHitGroup("", "shadowAnyHit"));
        mpCompositeTracer.pBindingTable->setMiss(0u, rtProgDesc.addMiss("primaryMiss"));
        mpCompositeTracer.pBindingTable->setMiss(1u, rtProgDesc.addMiss("shadowMiss"));

        mpCompositeTracer.pProgram = Program::create(getDevice(), rtProgDesc, sceneDefines);
        mpCompositeTracer.pProgramVars = RtProgramVars::create(getDevice(), mpCompositeTracer.pProgram, mpCompositeTracer.pBindingTable);
    }

    // update ray density texture
    mpUpdateRayDensityProgram = Program::createCompute(getDevice(), "Samples/Caustics/UpdateRayDensity.cs.hlsl", "updateRayDensityTex");
    mpUpdateRayDensityState = ComputeState::create(getDevice());
    mpUpdateRayDensityState->setProgram(mpUpdateRayDensityProgram);
    mpUpdateRayDensityVars = ProgramVars::create(getDevice(), mpUpdateRayDensityProgram.get());

    // analyse trace result
    mpAnalyseProgram = Program::createCompute(getDevice(), "Samples/Caustics/AnalyseTraceResult.cs.hlsl", "addPhotonTaskFromTexture");
    mpAnalyseState = ComputeState::create(getDevice());
    mpAnalyseState->setProgram(mpAnalyseProgram);
    mpAnalyseVars = ProgramVars::create(getDevice(), mpAnalyseProgram.get());

    // generate ray count tex
    mpGenerateRayCountProgram = Program::createCompute(getDevice(), "Samples/Caustics/GenerateRayCountMipmap.cs.hlsl", "generateMip0");
    mpGenerateRayCountState = ComputeState::create(getDevice());
    mpGenerateRayCountState->setProgram(mpGenerateRayCountProgram);
    mpGenerateRayCountVars = ProgramVars::create(getDevice(), mpGenerateRayCountProgram.get());

    // generate ray count mip tex
    mpGenerateRayCountMipProgram = Program::createCompute(getDevice(), "Samples/Caustics/GenerateRayCountMipmap.cs.hlsl", "generateMipLevel");
    mpGenerateRayCountMipState = ComputeState::create(getDevice());
    mpGenerateRayCountMipState->setProgram(mpGenerateRayCountMipProgram);
    mpGenerateRayCountMipVars = ProgramVars::create(getDevice(), mpGenerateRayCountMipProgram.get());

    // smooth photon
    mpSmoothProgram = Program::createCompute(getDevice(), "Samples/Caustics/SmoothPhoton.cs.hlsl", "main");
    mpSmoothState = ComputeState::create(getDevice());
    mpSmoothState->setProgram(mpSmoothProgram);
    mpSmoothVars = ProgramVars::create(getDevice(), mpSmoothProgram.get());

    // allocate tile
    const char* shaderEntries[] = {"OrthogonalizePhoton", "CountTilePhoton", "AllocateMemory", "StoreTilePhoton"};
    for (int i = 0; i < GATHER_PROCESSING_SHADER_COUNT; i++)
    {
        mpAllocateTileProgram[i] = Program::createCompute(getDevice(), "Samples/Caustics/AllocateTilePhoton.cs.hlsl", shaderEntries[i]);
        mpAllocateTileState[i] = ComputeState::create(getDevice());
        mpAllocateTileState[i]->setProgram(mpAllocateTileProgram[i]);
        mpAllocateTileVars[i] = ProgramVars::create(getDevice(), mpAllocateTileProgram[i].get());
    }

    // photon gather
    mpPhotonGatherProgram = Program::createCompute(getDevice(), "Samples/Caustics/PhotonGather.cs.hlsl", "main");
    mpPhotonGatherState = ComputeState::create(getDevice());
    mpPhotonGatherState->setProgram(mpPhotonGatherProgram);
    mpPhotonGatherVars = ProgramVars::create(getDevice(), mpPhotonGatherProgram.get());

    // photon scatter
    {
        mpPhotonScatterProgram = Program::createGraphics(getDevice(), kPhotonScatter3dShaderFilename, "photonScatterVS", "photonScatterPS", sceneDefines);

        BlendState::Desc blendDesc;
        blendDesc.setRtBlend(0u, true);
        blendDesc.setRtParams(0u,
                              BlendState::BlendOp::Add,
                              BlendState::BlendOp::Add,
                              BlendState::BlendFunc::One,
                              BlendState::BlendFunc::One,
                              BlendState::BlendFunc::One,
                              BlendState::BlendFunc::One);
        ref<BlendState> scatterBlendState = BlendState::create(blendDesc);

        BlendState::Desc noBlendDesc;
        noBlendDesc.setRtBlend(0u, false);
        ref<BlendState> noBlendState = BlendState::create(noBlendDesc);

        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthEnabled(false);
        dsDesc.setDepthWriteMask(false);
        auto depthStencilState = DepthStencilState::create(dsDesc);

        RasterizerState::Desc rasterDesc;
        rasterDesc.setCullMode(RasterizerState::CullMode::None);
        static int32_t depthBias = -8;
        static float slopeBias = -16.0f;
        rasterDesc.setDepthBias(depthBias, slopeBias);
        auto rasterState = RasterizerState::create(rasterDesc);

        mpPhotonScatterBlendState = GraphicsState::create(getDevice());
        mpPhotonScatterBlendState->setProgram(mpPhotonScatterProgram);
        mpPhotonScatterBlendState->setBlendState(scatterBlendState);
        mpPhotonScatterBlendState->setDepthStencilState(depthStencilState);
        mpPhotonScatterBlendState->setRasterizerState(rasterState);

        mpPhotonScatterNoBlendState = GraphicsState::create(getDevice());
        mpPhotonScatterNoBlendState->setProgram(mpPhotonScatterProgram);
        mpPhotonScatterNoBlendState->setRasterizerState(rasterState);
        mpPhotonScatterNoBlendState->setBlendState(noBlendState);
        mpPhotonScatterNoBlendState->setDepthStencilState(depthStencilState);
        mpPhotonScatterVars = ProgramVars::create(getDevice(), mpPhotonScatterProgram->getReflector());
    }

    // temporal filter
    mpFilterProgram = Program::createCompute(getDevice(), "Samples/Caustics/TemporalFilter.cs.hlsl", "main");
    mpFilterState = ComputeState::create(getDevice());
    mpFilterState->setProgram(mpFilterProgram);
    mpFilterVars = ProgramVars::create(getDevice(), mpFilterProgram.get());

    // spacial filter
    mpSpacialFilterProgram = Program::createCompute(getDevice(), "Samples/Caustics/SpacialFilter.cs.hlsl", "main");
    mpSpacialFilterState = ComputeState::create(getDevice());
    mpSpacialFilterState->setProgram(mpSpacialFilterProgram);
    mpSpacialFilterVars = ProgramVars::create(getDevice(), mpSpacialFilterProgram.get());

    // caustics raster
    {
        ProgramDesc rasterDesc;
        rasterDesc.addShaderModules(shaderModules);
        rasterDesc.addShaderLibrary(kCaustics3dShaderFilename)/*.vsEntry("vsMain")*/.psEntry("psMain");
        rasterDesc.addTypeConformances(typeConformances);

        mpRasterPass = RasterPass::create(getDevice(), rasterDesc, sceneDefines);
    }

    // GBuffer raster
    {
        ProgramDesc rasterDesc;
        rasterDesc.addShaderModules(shaderModules);
        rasterDesc.addShaderLibrary(kGBuffer3dShaderFilename)/*.vsEntry("vsMain")*/.psEntry("psMain");
        rasterDesc.addTypeConformances(typeConformances);

        mpGPass = RasterPass::create(getDevice(), rasterDesc, sceneDefines);
    }

    // composite raster
    {
        ProgramDesc rasterDesc;
        rasterDesc.addShaderModules(shaderModules);
        rasterDesc.addShaderLibrary(kComposite3dShaderFilename)/*.vsEntry("vsMain")*/.psEntry("psMain");
        rasterDesc.addTypeConformances(typeConformances);

        mpCompositePass = FullScreenPass::create(getDevice(), rasterDesc, sceneDefines);
    }

    // Samplers
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    samplerDesc.setAddressingMode(TextureAddressingMode::Border, TextureAddressingMode::Border, TextureAddressingMode::Border);
    mpLinearSampler = getDevice()->createSampler(samplerDesc);

    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
    mpPointSampler = getDevice()->createSampler(samplerDesc);
}

void Caustics::setCommonVars(ProgramVars* pVars, const Fbo* pTargetFbo)
{
    //auto pCB = pVars->getRootVar()["PerFrameCB"];
    //pCB["invView"] = inverse(mpCamera->getViewMatrix());
    //pCB["viewportDims"] = float2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
    //pCB["emitSize"] = mEmitSize;
    //float fovY = focalLengthToFovY(mpCamera->getFocalLength(), Camera::kDefaultFrameHeight);
    //pCB["tanHalfFovY"] = tanf(fovY * 0.5f);
    //pCB["sampleIndex"] = mSampleIndex;
    //pCB["useDOF"] = false;// mUseDOF;
}

void Caustics::setPhotonTracingCommonVariable(Raytracer& photonTracer)
{

}

void Caustics::saveSceneSetting(std::string path)
{
    if (path.find(".ini") == std::string::npos)
    {
        path += ".ini";
    }
    std::ofstream file(path, std::ios::out);
    if (!file)
    {
        return;
    }

    file << mLightAngle.x << " " << mLightAngle.y << std::endl;
    float3 camOri = mpCamera->getPosition();
    float3 camTarget = mpCamera->getTarget();
    file << camOri.x << " " << camOri.y << " " << camOri.z << std::endl;
    file << camTarget.x << " " << camTarget.y << " " << camTarget.z << std::endl;
}

void Caustics::createCausticsMap()
{
    uint32_t width = mpRtOut->getWidth();
    uint32_t height = mpRtOut->getHeight();
    uint2 dim(width / mCausticsMapResRatio, height / mCausticsMapResRatio);

    mpSmallPhotonTex = getDevice()->createTexture2D(dim.x, dim.y, ResourceFormat::R32Uint, 1u, 1u, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    auto pPhotonMapTex = getDevice()->createTexture2D(dim.x, dim.y, ResourceFormat::RGBA16Float, 1u, 1u, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    auto depthTex = getDevice()->createTexture2D(dim.x, dim.y, ResourceFormat::D32Float, 1u, 1u, nullptr, ResourceBindFlags::DepthStencil);
    mpCausticsFbo[0] = Fbo::create(getDevice(), {pPhotonMapTex}, depthTex);

    pPhotonMapTex = getDevice()->createTexture2D(dim.x, dim.y, ResourceFormat::RGBA16Float, 1u, 1u, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    mpCausticsFbo[1] = Fbo::create(getDevice(), {pPhotonMapTex}, depthTex);
}

void Caustics::createGBuffer(int width, int height, GBuffer& gbuffer)
{
    gbuffer.mpDepthTex = getDevice()->createTexture2D(width, height, ResourceFormat::D32Float, 1u, 1u, nullptr, ResourceBindFlags::DepthStencil | ResourceBindFlags::ShaderResource);
    gbuffer.mpNormalTex = getDevice()->createTexture2D(width, height, ResourceFormat::RGBA16Float, 1u, 1u, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);
    gbuffer.mpDiffuseTex = getDevice()->createTexture2D(width, height, ResourceFormat::RGBA16Float, 1u, 1u, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);
    gbuffer.mpSpecularTex = getDevice()->createTexture2D(width, height, ResourceFormat::RGBA16Float, 1u, 1u, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);
    gbuffer.mpGPassFbo = Fbo::create(getDevice(), {gbuffer.mpNormalTex, gbuffer.mpDiffuseTex, gbuffer.mpSpecularTex}, gbuffer.mpDepthTex);
}

int2 Caustics::getTileDim() const
{
    int2 tileDim;
    tileDim.x = (mpRtOut->getWidth() / mCausticsMapResRatio + mTileSize.x - 1) / mTileSize.x;
    tileDim.y = (mpRtOut->getHeight() / mCausticsMapResRatio + mTileSize.y - 1) / mTileSize.y;
    return tileDim;
}

float Caustics::resolutionFactor()
{
    float2 res(mpRtOut->getWidth(), mpRtOut->getHeight());
    float2 refRes(1920.0f, 1080.0f);
    return length(res) / length(refRes);
}

float2 getRandomPoint(int i)
{
    const double g = 1.32471795724474602596;
    const double a1 = 1.0 / g;
    const double a2 = 1.0 / (g * g);
    double x = 0.5 + a1 * (i + 1);
    double y = 0.5 + a2 * (i + 1);
    float xF = float(x - floor(x));
    float yF = float(y - floor(y));
    return float2(xF, yF);
}

void Caustics::renderRT(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    FALCOR_ASSERT(mpScene);
    FALCOR_PROFILE(pRenderContext, "renderRT");
    // setPerFrameVars(pTargetFbo.get());

    // reset data
    uint32_t statisticsOffset = uint32_t(mFrameCounter % mpPhotonCountTex->getWidth());
    int thisIdx = mFrameCounter % 2;
    int lastIdx = 1 - thisIdx;
    GBuffer* gBuffer = mGBuffer + thisIdx;
    GBuffer* gBufferLast = mGBuffer + lastIdx;
    ref<Fbo> causticsFbo = mpCausticsFbo[thisIdx];
    ref<Fbo> causticsFboLast = mpCausticsFbo[lastIdx];

    if (mUpdatePhoton)
    {
        auto var = mpDrawArgumentVars->getRootVar();
        var["PerFrameCB"]["initRayCount"] = uint(mDispatchSize * mDispatchSize);
        var["PerFrameCB"]["coarseDim"] = uint2(mDispatchSize, mDispatchSize);
        var["PerFrameCB"]["textureOffset"] = statisticsOffset;
        var["PerFrameCB"]["scatterGeoIdxCount"] = mScatterGeometry == SCATTER_GEOMETRY_QUAD ? 6U : 12U;
        mpDrawArgumentVars->setBuffer("gDrawArgument", mpDrawArgumentBuffer); // setStructuredBuffer
        mpDrawArgumentVars->setBuffer("gRayArgument", mpRayArgumentBuffer);   // setStructuredBuffer
        // mpDrawArgumentVars->setBuffer("gPhotonBuffer", mpPhotonBuffer);    // setStructuredBuffer
        mpDrawArgumentVars->setBuffer("gPixelInfo", mpPixelInfoBuffer);       // setStructuredBuffer
        mpDrawArgumentVars->setTexture("gPhotonCountTexture", mpPhotonCountTex);
        pRenderContext->dispatch(mpDrawArgumentState.get(), mpDrawArgumentVars.get(), uint3(mDispatchSize / 16, mDispatchSize / 16, 1));
    }

    // gpass
    {
        auto& GPassState = mpGPass->getState();
        GPassState->setFbo(gBuffer->mpGPassFbo);

        pRenderContext->clearFbo(gBuffer->mpGPassFbo.get(), float4(0.0f, 0.0f, 0.0f, 1.0f), 1.0f, 0);
        //mpGPass->getState()->setFbo(gBuffer->mpGPassFbo); /*mpGPass->renderScene(pContext, gBuffer->mpGPassFbo);*/
        mpScene->rasterize(pRenderContext, GPassState.get(), mpGPass->getVars().get());
    }

    // photon tracing
    if (mTraceType != TRACE_NONE)
    {
        pRenderContext->clearUAV(mpSmallPhotonTex->getUAV().get(), uint4(0u, 0u, 0u, 0u));
        auto photonTracer = getPhotonTraceShader();
        // setPhotonTracingCommonVariable(photonTracer);
        //GraphicsVars* pVars = photonTraceShader.mpPhotonTraceVars->getGlobalVars().get();
        auto var = photonTracer.pProgramVars->getRootVar();
        float2 r = getRandomPoint(mFrameCounter) * 2.0f - 1.0f;
        float2 sign(r.x > 0 ? 1 : -1, r.y > 0 ? 1 : -1);
        float2 randomOffset = sign * float2(pow(abs(r.x), mJitterPower), pow(abs(r.y), mJitterPower)) * mJitter;
        var["PerFrameCB"]["invView"] = inverse(mpCamera->getViewMatrix());
        var["PerFrameCB"]["viewportDims"] = float2(mpRtOut->getWidth(), mpRtOut->getHeight());
        var["PerFrameCB"]["emitSize"] = mEmitSize;
        var["PerFrameCB"]["roughThreshold"] = mRoughThreshold;
        var["PerFrameCB"]["randomOffset"] = mTemporalFilter ? randomOffset : float2(0, 0);
        var["PerFrameCB"]["rayTaskOffset"] = mDispatchSize * mDispatchSize;
        var["PerFrameCB"]["coarseDim"] = uint2(mDispatchSize, mDispatchSize);
        var["PerFrameCB"]["maxDepth"] = mMaxTraceDepth;
        var["PerFrameCB"]["iorOverride"] = mIOROveride;
        var["PerFrameCB"]["colorPhotonID"] = (uint32_t)mColorPhoton;
        var["PerFrameCB"]["photonIDScale"] = mPhotonIDScale;
        var["PerFrameCB"]["traceColorThreshold"] = mTraceColorThreshold * (512 * 512) / (mDispatchSize * mDispatchSize);
        var["PerFrameCB"]["cullColorThreshold"] = mCullColorThreshold / 255;
        var["PerFrameCB"]["gAreaType"] = (uint32_t)mAreaType;
        var["PerFrameCB"]["gIntensity"] = mIntensity / 1000;
        var["PerFrameCB"]["gSplatSize"] = mSplatSize;
        var["PerFrameCB"]["updatePhoton"] = (uint32_t)mUpdatePhoton;
        var["PerFrameCB"]["gMinDrawCount"] = mFastPhotonDrawCount;
        var["PerFrameCB"]["gMinScreenRadius"] = mFastPhotonPixelRadius * resolutionFactor();
        var["PerFrameCB"]["gMaxScreenRadius"] = mMaxPhotonPixelRadius * resolutionFactor();
        var["PerFrameCB"]["gMipmap"] = int(log(mDispatchSize) / log(2));
        var["PerFrameCB"]["gSmallPhotonColorScale"] = mSmallPhotonCompressScale;
        var["PerFrameCB"]["cameraPos"] = mpCamera->getPosition();

        var["gPhotonBuffer"].setBuffer(mpPhotonBuffer);
        var["gRayTask"].setBuffer(mpRayTaskBuffer);
        var["gRayArgument"].setBuffer(mpRayArgumentBuffer);
        var["gPixelInfo"].setBuffer(mpPixelInfoBuffer);
        var["gDrawArgument"].setBuffer(mpDrawArgumentBuffer);
        var["gRayCountQuadTree"].setBuffer(mpRayCountQuadTree);
        var["gUniformNoise"].setTexture(mpUniformNoise);

        var["gRayDensityTex"].setTexture(mpRayDensityTex);
        var["gSmallPhotonBuffer"].setTexture(mpSmallPhotonTex);
        var["gPhotonTexture"].setTexture(causticsFboLast->getColorTexture(0));

        var["gPixelInfo"].setBuffer(mpPixelInfoBuffer);
        var["gPhotonBuffer"].setBuffer(mpPhotonBuffer);
        var["gDrawArgument"].setBuffer(mpDrawArgumentBuffer);
        var["gRayTask"].setBuffer(mpRayTaskBuffer);

        uint3 resolution = mTraceType == TRACE_FIXED ? uint3(mDispatchSize, mDispatchSize, 1u) : uint3(2048u, 4096u, 1u);

        mpScene->raytrace(pRenderContext, photonTracer.pProgram.get(), photonTracer.pProgramVars, resolution);
    }

    // analysis output
    if (mUpdatePhoton)
    {
        if (mTraceType == TRACE_ADAPTIVE || mTraceType == TRACE_ADAPTIVE_RAY_MIP_MAP)
        {
            auto var = mpUpdateRayDensityVars->getRootVar();
            var["PerFrameCB"]["coarseDim"] = int2(mDispatchSize, mDispatchSize);
            var["PerFrameCB"]["minPhotonPixelSize"] = mMinPhotonPixelSize * resolutionFactor();
            var["PerFrameCB"]["smoothWeight"] = mSmoothWeight;
            var["PerFrameCB"]["maxTaskPerPixel"] = (int)mMaxTaskCountPerPixel;
            var["PerFrameCB"]["updateSpeed"] = mUpdateSpeed;
            var["PerFrameCB"]["varianceGain"] = mVarianceGain;
            var["PerFrameCB"]["derivativeGain"] = mDerivativeGain;
            mpUpdateRayDensityVars->setBuffer("gPixelInfo", mpPixelInfoBuffer); //setStructuredBuffer
            mpUpdateRayDensityVars->setBuffer("gRayArgument", mpRayArgumentBuffer); //setStructuredBuffer
            mpUpdateRayDensityVars->setTexture("gRayDensityTex", mpRayDensityTex);
            static int groupSize = 16;
            pRenderContext->dispatch(
                mpUpdateRayDensityState.get(), mpUpdateRayDensityVars.get(), uint3(mDispatchSize / groupSize, mDispatchSize / groupSize, 1));
        }

        if (mTraceType == TRACE_ADAPTIVE)
        {
            auto var = mpAnalyseVars->getRootVar();
            float4x4 wvp = mul(mpCamera->getProjMatrix(), mpCamera->getViewMatrix());
            var["PerFrameCB"]["viewProjMat"] = wvp; // mpCamera->getViewProjMatrix();
            var["PerFrameCB"]["taskDim"] = int2(mDispatchSize, mDispatchSize);
            var["PerFrameCB"]["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
            var["PerFrameCB"]["normalThreshold"] = mNormalThreshold;
            var["PerFrameCB"]["distanceThreshold"] = mDistanceThreshold;
            var["PerFrameCB"]["planarThreshold"] = mPlanarThreshold;
            var["PerFrameCB"]["samplePlacement"] = (uint32_t)mSamplePlacement;
            var["PerFrameCB"]["pixelLuminanceThreshold"] = mPixelLuminanceThreshold;
            var["PerFrameCB"]["minPhotonPixelSize"] = mMinPhotonPixelSize * resolutionFactor();
            static float2 offset(0.5f, 0.5f);
            static float speed = 0.0f;
            var["PerFrameCB"]["randomOffset"] = offset;
            offset += speed;
            mpAnalyseVars->setBuffer("gPhotonBuffer", mpPhotonBuffer);        //setStructuredBuffer
            mpAnalyseVars->setBuffer("gRayArgument", mpRayArgumentBuffer);    //setStructuredBuffer
            mpAnalyseVars->setBuffer("gRayTask", mpRayTaskBuffer);            //setStructuredBuffer
            mpAnalyseVars->setBuffer("gPixelInfo", mpPixelInfoBuffer);        //setStructuredBuffer
            mpAnalyseVars->setTexture("gDepthTex", gBuffer->mpGPassFbo->getDepthStencilTexture());
            mpAnalyseVars->setTexture("gRayDensityTex", mpRayDensityTex);
            int2 groupSize(32, 16);
            pRenderContext->dispatch(
                mpAnalyseState.get(), mpAnalyseVars.get(), uint3(mDispatchSize / groupSize.x, mDispatchSize / groupSize.y, 1)
            );
        }
        else if (mTraceType == TRACE_ADAPTIVE_RAY_MIP_MAP)
        {
            int startMipLevel = int(log(mDispatchSize) / log(2)) - 1;
            {
                auto var = mpGenerateRayCountVars->getRootVar();
                var["PerFrameCB"]["taskDim"] = int2(mDispatchSize, mDispatchSize);
                var["PerFrameCB"]["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
                var["PerFrameCB"]["mipLevel"] = startMipLevel;
                mpGenerateRayCountVars->setBuffer("gRayArgument", mpRayArgumentBuffer); // setStructuredBuffer
                mpGenerateRayCountVars->setTexture("gRayDensityTex", mpRayDensityTex);
                mpGenerateRayCountVars->setBuffer("gRayCountQuadTree", mpRayCountQuadTree); // setStructuredBuffer
                int2 groupSize(8, 8);
                uint3 blockCount(mDispatchSize / groupSize.x / 2, mDispatchSize / groupSize.y / 2, 1);
                pRenderContext->dispatch(mpGenerateRayCountState.get(), mpGenerateRayCountVars.get(), blockCount);
            }

            for (int mipLevel = startMipLevel - 1, dispatchSize = mDispatchSize / 4; mipLevel >= 0; mipLevel--, dispatchSize >>= 1)
            {
                auto var = mpGenerateRayCountMipVars->getRootVar();
                var["PerFrameCB"]["taskDim"] = int2(mDispatchSize, mDispatchSize);
                var["PerFrameCB"]["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
                var["PerFrameCB"]["mipLevel"] = mipLevel;
                mpGenerateRayCountMipVars->setBuffer("gRayArgument", mpRayArgumentBuffer); //setStructuredBuffer
                mpGenerateRayCountMipVars->setTexture("gRayDensityTex", mpRayDensityTex);
                mpGenerateRayCountMipVars->setBuffer("gRayCountQuadTree", mpRayCountQuadTree); // setStructuredBuffer
                int2 groupSize(8, 8);
                uint3 blockCount((dispatchSize + groupSize.x - 1) / groupSize.x, (dispatchSize + groupSize.y - 1) / groupSize.y, 1);
                pRenderContext->dispatch(mpGenerateRayCountMipState.get(), mpGenerateRayCountMipVars.get(), blockCount);
            }
        }
    }

    // smooth photon
    ref<Buffer> photonBuffer = mpPhotonBuffer; //StructuredBuffer
    if (mRemoveIsolatedPhoton || mMedianFilter)
    {
        auto var = mpSmoothVars->getRootVar();
        float4x4 wvp = mul(mpCamera->getProjMatrix(), mpCamera->getViewMatrix());
        var["PerFrameCB"]["viewProjMat"] = wvp; // mpCamera->getViewProjMatrix();
        var["PerFrameCB"]["taskDim"] = int2(mDispatchSize, mDispatchSize);
        var["PerFrameCB"]["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
        var["PerFrameCB"]["normalThreshold"] = mNormalThreshold;
        var["PerFrameCB"]["distanceThreshold"] = mDistanceThreshold;
        var["PerFrameCB"]["planarThreshold"] = mPlanarThreshold;
        var["PerFrameCB"]["pixelLuminanceThreshold"] = mPixelLuminanceThreshold;
        var["PerFrameCB"]["minPhotonPixelSize"] = mMinPhotonPixelSize * resolutionFactor();
        var["PerFrameCB"]["trimDirectionThreshold"] = trimDirectionThreshold;
        var["PerFrameCB"]["enableMedianFilter"] = uint32_t(mMedianFilter);
        var["PerFrameCB"]["removeIsolatedPhoton"] = uint32_t(mRemoveIsolatedPhoton);
        var["PerFrameCB"]["minNeighbourCount"] = mMinNeighbourCount;
        mpSmoothVars->setBuffer("gSrcPhotonBuffer", mpPhotonBuffer);  //setStructuredBuffer
        mpSmoothVars->setBuffer("gDstPhotonBuffer", mpPhotonBuffer2); //setStructuredBuffer
        mpSmoothVars->setBuffer("gRayArgument", mpRayArgumentBuffer); //setStructuredBuffer
        mpSmoothVars->setBuffer("gRayTask", mpPixelInfoBuffer);       //setStructuredBuffer
        mpSmoothVars->setTexture("gDepthTex", gBuffer->mpGPassFbo->getDepthStencilTexture());
        static int groupSize = 16;
        pRenderContext->dispatch(mpSmoothState.get(), mpSmoothVars.get(), uint3(mDispatchSize / groupSize, mDispatchSize / groupSize, 1));
        photonBuffer = mpPhotonBuffer2;
    }

    // photon scattering
    if (mScatterOrGather == DENSITY_ESTIMATION_SCATTER)
    {
        pRenderContext->clearRtv(causticsFbo->getColorTexture(0)->getRTV().get(), float4(0.0f, 0.0f, 0.0f, 0.0f));
        float4x4 wvp = mul(mpCamera->getProjMatrix(), mpCamera->getViewMatrix());
        float4x4 invP = inverse(mpCamera->getViewMatrix());
        auto var = mpPhotonScatterVars->getRootVar();
        var["PerFrameCB"]["gWorldMat"] = float4x4();
        var["PerFrameCB"]["gWvpMat"] = wvp;
        var["PerFrameCB"]["gInvProjMat"] = invP;
        var["PerFrameCB"]["gEyePosW"] = mpCamera->getPosition();
        var["PerFrameCB"]["gSplatSize"] = mSplatSize;
        var["PerFrameCB"]["gPhotonMode"] = (uint)mPhotonMode;
        var["PerFrameCB"]["gKernelPower"] = mKernelPower;
        var["PerFrameCB"]["gShowPhoton"] = uint32_t(mPhotonDisplayMode);
        var["PerFrameCB"]["gLightDir"] = mLightDirection;
        var["PerFrameCB"]["taskDim"] = int2(mDispatchSize, mDispatchSize);
        var["PerFrameCB"]["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
        var["PerFrameCB"]["normalThreshold"] = mScatterNormalThreshold;
        var["PerFrameCB"]["distanceThreshold"] = mScatterDistanceThreshold;
        var["PerFrameCB"]["planarThreshold"] = mScatterPlanarThreshold;
        var["PerFrameCB"]["gMaxAnisotropy"] = mMaxAnisotropy;
        var["PerFrameCB"]["gCameraPos"] = mpCamera->getPosition();
        var["PerFrameCB"]["gZTolerance"] = mZTolerance;
        var["PerFrameCB"]["gResRatio"] = mCausticsMapResRatio;
        mpPhotonScatterVars->setSampler("gLinearSampler", mpLinearSampler);
        mpPhotonScatterVars->setBuffer("gPhotonBuffer", photonBuffer); //setStructuredBuffer
        mpPhotonScatterVars->setBuffer("gRayTask", mpPixelInfoBuffer); //setStructuredBuffer
        mpPhotonScatterVars->setTexture("gDepthTex", gBuffer->mpGPassFbo->getDepthStencilTexture());
        mpPhotonScatterVars->setTexture("gNormalTex", gBuffer->mpGPassFbo->getColorTexture(0));
        mpPhotonScatterVars->setTexture("gDiffuseTex", gBuffer->mpGPassFbo->getColorTexture(1));
        mpPhotonScatterVars->setTexture("gSpecularTex", gBuffer->mpGPassFbo->getColorTexture(2));
        mpPhotonScatterVars->setTexture("gGaussianTex", mpGaussianKernel);
        int instanceCount = mDispatchSize * mDispatchSize;

        ref<GraphicsState> scatterState = (mPhotonDisplayMode == 2) ? mpPhotonScatterNoBlendState : mpPhotonScatterBlendState;
        FALCOR_ASSERT(causticsFbo);
        FALCOR_ASSERT(mpQuad);
        FALCOR_ASSERT(mpSphere);

        if (mScatterGeometry == SCATTER_GEOMETRY_QUAD)
            scatterState->setVao(mpQuad->getMeshVao());
        else
            scatterState->setVao(mpSphere->getMeshVao());

        scatterState->setFbo(causticsFbo);

        if (mPhotonMode == PHOTON_MODE_PHOTON_MESH)
        {
            pRenderContext->drawIndexedInstanced(scatterState.get(), mpPhotonScatterVars.get(), 6u, mDispatchSize * mDispatchSize, 0u, 0, 0u);
        }
        else
        {
            pRenderContext->drawIndexedIndirect(scatterState.get(), mpPhotonScatterVars.get(), 1u, mpDrawArgumentBuffer.get(), (uint64_t)0, nullptr, (uint64_t)0);
        }
    }
    else if (mScatterOrGather == DENSITY_ESTIMATION_GATHER)
    {
        int2 tileDim = getTileDim();
        int dimX, dimY;
        if (mTraceType == TRACE_FIXED)
        {
            dimX = mDispatchSize;
            dimY = mDispatchSize;
        }
        else
        {
            int photonCount = MAX_PHOTON_COUNT;
            int sqrtCount = int(sqrt(photonCount));
            dimX = dimY = sqrtCount;
        }
        int blockSize = 32;
        uint3 dispatchDim[] = {
            uint3((dimX + blockSize - 1) / blockSize, (dimY + blockSize - 1) / blockSize, 1),
            uint3((dimX + blockSize - 1) / blockSize, (dimY + blockSize - 1) / blockSize, 1),
            uint3((tileDim.x + mTileSize.x - 1) / mTileSize.x, (tileDim.y + mTileSize.y - 1) / mTileSize.y, 1),
            uint3((dimX + blockSize - 1) / blockSize, (dimY + blockSize - 1) / blockSize, 1)
        };
        // build tile data
        int2 screenSize(mpRtOut->getWidth() / mCausticsMapResRatio, mpRtOut->getHeight() / mCausticsMapResRatio);
        for (int i = 0; i < GATHER_PROCESSING_SHADER_COUNT; i++)
        {
            auto vars = mpAllocateTileVars[i];
            auto states = mpAllocateTileState[i];
            auto var = vars->getRootVar();
            float4x4 wvp = mul(mpCamera->getProjMatrix(), mpCamera->getViewMatrix());
            var["PerFrameCB"]["gViewProjMat"] = wvp;
            var["PerFrameCB"]["screenDim"] = screenSize;
            var["PerFrameCB"]["tileDim"] = tileDim;
            var["PerFrameCB"]["gSplatSize"] = mSplatSize;
            var["PerFrameCB"]["minColor"] = mMinGatherColor;
            var["PerFrameCB"]["blockCount"] = int2(dispatchDim[i].x, dispatchDim[i].y);
            vars->setBuffer("gDrawArgument", mpDrawArgumentBuffer); //setStructuredBuffer
            vars->setBuffer("gPhotonBuffer", photonBuffer);         //setStructuredBuffer
            vars->setBuffer("gTileInfo", mpTileIDInfoBuffer);       //setStructuredBuffer
            vars->setBuffer("gIDBuffer", mpIDBuffer);               //setRawBuffer
            vars->setBuffer("gIDCounter", mpIDCounterBuffer);       //setRawBuffer
            pRenderContext->dispatch(states.get(), vars.get(), dispatchDim[i]);
        }
        // gathering
        auto var = mpPhotonGatherVars->getRootVar();
        float4x4 wvp = mul(mpCamera->getProjMatrix(), mpCamera->getViewMatrix());
        var["PerFrameCB"]["gInvViewProjMat"] = mpCamera->getInvViewProjMatrix();
        var["PerFrameCB"]["screenDim"] = screenSize;
        var["PerFrameCB"]["tileDim"] = tileDim;
        var["PerFrameCB"]["gSplatSize"] = mSplatSize;
        var["PerFrameCB"]["gDepthRadius"] = mDepthRadius;
        var["PerFrameCB"]["gShowTileCount"] = int(mShowTileCount);
        var["PerFrameCB"]["gTileCountScale"] = int(mTileCountScale);
        var["PerFrameCB"]["gKernelPower"] = mKernelPower;
        var["PerFrameCB"]["causticsMapResRatio"] = mCausticsMapResRatio;
        mpPhotonGatherVars->setBuffer("gPhotonBuffer", photonBuffer);     //setStructuredBuffer
        mpPhotonGatherVars->setBuffer("gTileInfo", mpTileIDInfoBuffer);   //setStructuredBuffer
        mpPhotonGatherVars->setBuffer("gIDBuffer", mpIDBuffer); //setRawBuffer
        mpPhotonGatherVars->setTexture("gDepthTex", gBuffer->mpGPassFbo->getDepthStencilTexture());
        mpPhotonGatherVars->setTexture("gNormalTex", gBuffer->mpGPassFbo->getColorTexture(0));
        mpPhotonGatherVars->setTexture("gPhotonTex", causticsFbo->getColorTexture(0));
        uint3 dispatchSize((screenSize.x + mTileSize.x - 1) / mTileSize.x, (screenSize.y + mTileSize.y - 1) / mTileSize.y, 1);
        pRenderContext->dispatch(mpPhotonGatherState.get(), mpPhotonGatherVars.get(), dispatchSize);
    }

    // Temporal filter
    if (mTemporalFilter)
    {
        static float4x4 lastViewProj;
        static float4x4 lastProj;
        float4x4 thisViewProj = mpCamera->getViewProjMatrix();
        float4x4 thisProj = mpCamera->getProjMatrix();
        float4x4 reproj = mul(lastViewProj, inverse(thisViewProj));
        int2 causticsDim(causticsFbo->getWidth(), causticsFbo->getHeight());
        auto var = mpFilterVars->getRootVar();
        var["PerFrameCB"]["causticsDim"] = causticsDim;
        var["PerFrameCB"]["gBufferDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
        var["PerFrameCB"]["blendWeight"] = mFilterWeight;
        var["PerFrameCB"]["reprojMatrix"] = reproj;
        var["PerFrameCB"]["invProjMatThis"] = inverse(thisProj);
        var["PerFrameCB"]["invProjMatLast"] = inverse(lastProj);
        var["PerFrameCB"]["normalKernel"] = mTemporalNormalKernel;
        var["PerFrameCB"]["depthKernel"] = mTemporalDepthKernel;
        var["PerFrameCB"]["colorKernel"] = mTemporalColorKernel;
        mpFilterVars->setTexture("causticsTexThis", causticsFbo->getColorTexture(0));
        mpFilterVars->setTexture("causticsTexLast", causticsFboLast->getColorTexture(0));
        mpFilterVars->setTexture("depthTexThis", gBuffer->mpDepthTex);
        mpFilterVars->setTexture("depthTexLast", gBufferLast->mpDepthTex); // ensure that gBufferLast->mpDepthTex is D32Float
        mpFilterVars->setTexture("normalTexThis", gBuffer->mpNormalTex);
        mpFilterVars->setTexture("normalTexLast", gBufferLast->mpNormalTex);
        static int groupSize = 16;
        uint3 dim((causticsDim.x + groupSize - 1) / groupSize, (causticsDim.y + groupSize - 1) / groupSize, 1);
        pRenderContext->dispatch(mpFilterState.get(), mpFilterVars.get(), dim);
        lastViewProj = thisViewProj;
        lastProj = thisProj;
    }

    // Spacial filter
    if (mSpacialFilter)
    {
        for (int i = 0; i < mSpacialPasses; i++)
        {
            int2 causticsDim(causticsFbo->getWidth(), causticsFbo->getHeight());
            auto var = mpSpacialFilterVars->getRootVar();
            var["PerFrameCB"]["causticsDim"] = causticsDim;
            var["PerFrameCB"]["gBufferDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
            var["PerFrameCB"]["normalKernel"] = mSpacialNormalKernel;
            var["PerFrameCB"]["depthKernel"] = mSpacialDepthKernel;
            var["PerFrameCB"]["colorKernel"] = mSpacialColorKernel;
            var["PerFrameCB"]["screenKernel"] = mSpacialScreenKernel;
            var["PerFrameCB"]["passID"] = i;
            var["PerFrameCB"]["gSmallPhotonColorScale"] = mSmallPhotonCompressScale;
            mpSpacialFilterVars->setTexture("causticsTexThis", causticsFbo->getColorTexture(0));
            mpSpacialFilterVars->setTexture("depthTexThis", gBuffer->mpDepthTex);
            mpSpacialFilterVars->setTexture("normalTexThis", gBuffer->mpNormalTex);
            mpSpacialFilterVars->setTexture("smallPhotonTex", mpSmallPhotonTex);
            static int groupSize = 16;
            uint3 dim((causticsDim.x + groupSize - 1) / groupSize, (causticsDim.y + groupSize - 1) / groupSize, 1);
            pRenderContext->dispatch(mpSpacialFilterState.get(), mpSpacialFilterVars.get(), dim);
        }
    }

    // Render output
    if (mDebugMode == ShowRayTracing ||
        mDebugMode == ShowAvgScreenArea ||
        mDebugMode == ShowAvgScreenAreaVariance ||
        mDebugMode == ShowCount ||
        mDebugMode == ShowTotalPhoton ||
        mDebugMode == ShowRayTex ||
        mDebugMode == ShowRayCountMipmap ||
        mDebugMode == ShowPhotonDensity ||
        mDebugMode == ShowSmallPhoton ||
        mDebugMode == ShowSmallPhotonCount)
    {
        pRenderContext->clearUAV(mpRtOut->getUAV().get(), kClearColor);
        auto var = mpCompositeTracer.pProgramVars->getRootVar();
        var["PerFrameCB"]["invView"] = inverse(mpCamera->getViewMatrix());
        var["PerFrameCB"]["invProj"] = inverse(mpCamera->getProjMatrix());
        var["PerFrameCB"]["viewportDims"] = float2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
        float fovY = focalLengthToFovY(mpCamera->getFocalLength(), Camera::kDefaultFrameHeight);
        var["PerFrameCB"]["tanHalfFovY"] = tanf(fovY * 0.5f);
        var["PerFrameCB"]["sampleIndex"] = mSampleIndex++;
        var["PerFrameCB"]["useDOF"] = mUseDOF;
        var["PerFrameCB"]["roughThreshold"] = mRoughThreshold;
        var["PerFrameCB"]["maxDepth"] = mMaxTraceDepth;
        var["PerFrameCB"]["iorOverride"] = mIOROveride;
        var["PerFrameCB"]["causticsResRatio"] = mCausticsMapResRatio;
        var["PerFrameCB"]["gPosKernel"] = mFilterCausticsMap ? mUVKernel : 0.0f;
        var["PerFrameCB"]["gZKernel"] = mFilterCausticsMap ? mZKernel : 0.0f;
        var["PerFrameCB"]["gNormalKernel"] = mFilterCausticsMap ? mNormalKernel : 0.0f;

        var["gCausticsTex"].setTexture(causticsFbo->getColorTexture(0));
        var["gNormalTex"].setTexture(gBuffer->mpGPassFbo->getColorTexture(0));
        var["gDepthTex"].setTexture(gBuffer->mpGPassFbo->getDepthStencilTexture());
        var["gLinearSampler"].setSampler(mpLinearSampler);
        var["gPointSampler"].setSampler(mpPointSampler);

        var["gOutput"].setTexture(mpRtOut);

        mpScene->raytrace(pRenderContext, mpCompositeTracer.pProgram.get(), mpCompositeTracer.pProgramVars, uint3(pTargetFbo->getWidth(), pTargetFbo->getHeight(), 1u));
    }

    {
        auto var = mpCompositePass->getRootVar();
        var["gDepthTex"] = gBuffer->mpGPassFbo->getDepthStencilTexture();
        var["gNormalTex"] = gBuffer->mpGPassFbo->getColorTexture(0);
        var["gDiffuseTex"] = gBuffer->mpGPassFbo->getColorTexture(1);
        var["gSpecularTex"] = gBuffer->mpGPassFbo->getColorTexture(2);
        var["gPhotonTex"] = causticsFbo->getColorTexture(0);
        var["gRayCountQuadTree"] = mpRayCountQuadTree;
        var["gRaytracingTex"] = mpRtOut;
        var["gRayTex"] = mpRayDensityTex;
        var["gStatisticsTex"] = mpPhotonCountTex;
        var["gSmallPhotonTex"] = mpSmallPhotonTex;
        var["gPointSampler"] = mpPointSampler;

        var["PerImageCB"]["gNumLights"] = mpScene->getLightCount();
        var["PerImageCB"]["gDebugMode"] = (uint32_t)mDebugMode;
        var["PerImageCB"]["gInvWvpMat"] = mpCamera->getInvViewProjMatrix();
        var["PerImageCB"]["gInvPMat"] = inverse(mpCamera->getProjMatrix());
        var["PerImageCB"]["gCameraPos"] = mpCamera->getPosition();
        var["PerImageCB"]["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
        var["PerImageCB"]["dispatchSize"] = int2(mDispatchSize, mDispatchSize);
        var["PerImageCB"]["gMaxPixelArea"] = mMaxPixelArea;
        var["PerImageCB"]["gMaxPhotonCount"] = mMaxPhotonCount;
        var["PerImageCB"]["gRayTexScale"] = mRayTexScaleFactor;
        var["PerImageCB"]["gStatisticsOffset"] = statisticsOffset;
        var["PerImageCB"]["gRayCountMip"] = mRayCountMipIdx;
        var["PerImageCB"]["gSmallPhotonColorScale"] = mSmallPhotonCompressScale;
        // old/explicit way to set variables
        mpCompositePass->getVars()->setBuffer("gPixelInfo", mpPixelInfoBuffer); //setStructuredBuffer

        mpCompositePass->execute(pRenderContext, pTargetFbo);
    }
    mFrameCounter++;
}

void Caustics::renderRaster(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    FALCOR_ASSERT(mpScene);
    FALCOR_PROFILE(pRenderContext, "renderRaster");

    mpRasterPass->getState()->setFbo(pTargetFbo);
    mpScene->rasterize(pRenderContext, mpRasterPass->getState().get(), mpRasterPass->getVars().get());
}

uint Caustics::photonMacroToFlags() const
{
    uint flags = 0;
    flags |= (1 << mPhotonTraceMacro); // 3 bits
    flags |= ((1 << mTraceType) << 3); // 4 bits
    if (mFastPhotonPath)
        flags |= (1 << 7); // 1 bits
    if (mShrinkColorPayload)
        flags |= (1 << 8); // 1 bits
    if (mShrinkRayDiffPayload)
        flags |= (1 << 9); // 1 bits
    if (mUpdatePhoton)
        flags |= (1 << 10); // 1 bits
    return flags;
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "Caustics";
    config.windowDesc.resizableWindow = true;

    Caustics pRenderer(config);
    return pRenderer.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}

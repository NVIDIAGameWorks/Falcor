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
#pragma once

#include "Falcor.h"
#include "Core/SampleApp.h"
#include <Core/Pass/RasterPass.h>
#include <Core/Pass/FullScreenPass.h>

#define MAX_CAUSTICS_MAP_SIZE 2048
#define MAX_PHOTON_COUNT 2048 * 2048

using namespace Falcor;

class Caustics : public SampleApp
{
public:
    Caustics(const SampleAppConfig& config) : SampleApp(config) {}
    ~Caustics() noexcept = default;

    void onLoad(RenderContext* pRenderContext) override;
    void onShutdown() override;
    void onResize(uint32_t width, uint32_t height) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
    void onGuiRender(Gui* pGui) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;
    void onHotReload(HotReloadFlags reloaded) override;

private:
    // Photon trace
    enum TraceType
    {
        TRACE_FIXED = 0,
        TRACE_ADAPTIVE = 1,
        TRACE_NONE = 2,
        TRACE_ADAPTIVE_RAY_MIP_MAP = 3,
    };
    TraceType mTraceType = TRACE_ADAPTIVE_RAY_MIP_MAP;
    int mDispatchSize = 64;
    int mMaxTraceDepth = 10;
    float mEmitSize = 30.0;
    float mIntensity = 2.0f;
    float mRoughThreshold = 0.1f;

    enum AreaType
    {
        AREA_AVG_SQUARE = 0,
        AREA_AVG_LENGTH = 1,
        AREA_MAX_SQUARE = 2,
        AREA_EXACT = 3
    } mAreaType = AREA_AVG_SQUARE;

    float mIOROveride = 1.5f;
    int mColorPhoton = 0;
    int mPhotonIDScale = 50;
    float mTraceColorThreshold = 0.0005f;
    float mCullColorThreshold = 1.0f;
    bool mUpdatePhoton = true;
    float mMaxPhotonPixelRadius = 90.0f;
    float mSmallPhotonCompressScale = 1.0f;
    float mFastPhotonPixelRadius = 19.0f;
    float mFastPhotonDrawCount = 0.f;
    bool mFastPhotonPath = false;
    bool mShrinkColorPayload = true;
    bool mShrinkRayDiffPayload = true;

    // Adaptive photon refine
    float mNormalThreshold = 0.2f;
    float mDistanceThreshold = 10.0f;
    float mPlanarThreshold = 2.0f;
    float mMinPhotonPixelSize = 15.0f;
    float mSmoothWeight = 0.15f;
    float mMaxTaskCountPerPixel = 8192;
    float mUpdateSpeed = 0.2f;
    float mVarianceGain = 0.0f;
    float mDerivativeGain = 0.0f;

    enum SamplePlacement
    {
        SAMPLE_PLACEMENT_RANDOM = 0,
        SAMPLE_PLACEMENT_GRID = 1
    } mSamplePlacement = SAMPLE_PLACEMENT_GRID;

    // smooth photon
    bool mMedianFilter = false;
    int mMinNeighbourCount = 2;
    bool mRemoveIsolatedPhoton = false;
    float mPixelLuminanceThreshold = 0.5f;
    float trimDirectionThreshold = 0.5f;

    // Photon Scatter
    enum ScatterGeometry
    {
        SCATTER_GEOMETRY_QUAD = 0,
        SCATTER_GEOMETRY_SPHERE = 1,
    };
    ScatterGeometry mScatterGeometry = SCATTER_GEOMETRY_QUAD;
    int mCausticsMapResRatio = 1;
    enum DensityEstimation
    {
        DENSITY_ESTIMATION_SCATTER = 0,
        DENSITY_ESTIMATION_GATHER = 1,
        DENSITY_ESTIMATION_NONE = 2
    } mScatterOrGather = DENSITY_ESTIMATION_GATHER;
    float mSplatSize = 1.1f;
    float mKernelPower = 1.0f;
    enum PhotonDisplayMode
    {
        PHOTON_DISPLAY_MODE_KERNEL = 0,
        PHOTON_DISPLAY_MODE_SOLID = 1,
        PHOTON_DISPLAY_MODE_SHADED = 2,
    } mPhotonDisplayMode = PHOTON_DISPLAY_MODE_KERNEL;
    enum PhotonMode
    {
        PHOTON_MODE_ANISOTROPIC = 0,
        PHOTON_MODE_ISOTROPIC = 1,
        PHOTON_MODE_PHOTON_MESH = 2,
        PHOTON_MODE_SCREEN_DOT = 3,
        PHOTON_MODE_SCREEN_DOT_WITH_COLOR = 4,
    } mPhotonMode = PHOTON_MODE_ANISOTROPIC;
    float mScatterNormalThreshold = 0.2f;
    float mScatterDistanceThreshold = 10.0f;
    float mScatterPlanarThreshold = 2.0f;
    float mMaxAnisotropy = 20.0f;
    float mZTolerance = 0.2f;

    // Photon Gather
    int mTileCountScale = 10;
    uint2 mTileSize = uint2(32, 32);
    bool mShowTileCount = false;
    float mDepthRadius = 0.1f;
    float mMinGatherColor = 0.001f;

    // Temporal Filter
    bool mTemporalFilter = true;
    float mFilterWeight = 0.85f;
    float mJitter = 0.6f;
    float mJitterPower = 1.0f;
    float mTemporalNormalKernel = 0.7f;
    float mTemporalDepthKernel = 3.0f;
    float mTemporalColorKernel = 10.0f;

    // Spacial Filter
    bool mSpacialFilter = false;
    int mSpacialPasses = 1;
    float mSpacialNormalKernel = 0.7f;
    float mSpacialDepthKernel = 3.0f;
    float mSpacialColorKernel = 0.5f;
    float mSpacialScreenKernel = 1.0f;

    // Composite
    bool mRayTrace = true;
    enum Display
    {
        ShowRasterization = 0,
        ShowDepth = 1,
        ShowNormal = 2,
        ShowDiffuse = 3,
        ShowSpecular = 4,
        ShowPhoton = 5,
        ShowWorld = 6,
        ShowRoughness = 7,
        ShowRayTex = 8,
        ShowRayTracing = 9,
        ShowAvgScreenArea = 10,
        ShowAvgScreenAreaVariance = 11,
        ShowCount = 12,
        ShowTotalPhoton = 13,
        ShowRayCountMipmap = 14,
        ShowPhotonDensity = 15,
        ShowSmallPhoton = 16,
        ShowSmallPhotonCount = 17,
    };
    Display mDebugMode = ShowRayTracing;
    float mMaxPixelArea = 100.0f;
    float mMaxPhotonCount = 1000000.0f;
    int mRayCountMipIdx = 5;
    int mRayTexScaleFactor = 4;
    float mUVKernel = 0.7f;
    float mZKernel = 4.5f;
    float mNormalKernel = 4.0f;
    bool mFilterCausticsMap = false;

    // Others
    int mFrameCounter = 0;
    bool mUseDOF = false;
    uint32_t mSampleIndex = 0xdeadbeef;
    float2 mLightAngle{3.01f, 2.f};
    float3 mLightDirection;
    float2 mLightAngleSpeed{0, 0};

    ref<Scene> mpQuad;
    ref<Scene> mpSphere;
    ref<Scene> mpScene;
    ref<Camera> mpCamera;
    std::unique_ptr<FirstPersonCameraController> mCamController;

    // forward shading pass
    ref<RasterPass> mpRasterPass;

    // Clear draw argument
    ref<Program> mpDrawArgumentProgram;     // ComputeProgram
    ref<ProgramVars> mpDrawArgumentVars;    // ComputeVars
    ref<ComputeState> mpDrawArgumentState;
    ref<Buffer> mpDrawArgumentBuffer;       //StructuredBuffer

    // g-pass
    ref<RasterPass> mpGPass;
    struct GBuffer
    {
        ref<Texture> mpNormalTex;
        ref<Texture> mpDiffuseTex;
        ref<Texture> mpSpecularTex;
        ref<Texture> mpDepthTex;
        ref<Fbo>     mpGPassFbo;
    };
    GBuffer mGBuffer[2];
    ref<Texture> mpSmallPhotonTex;

    struct Raytracer
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pProgramVars;
    };

    // photon trace
    Raytracer mpPhotonTracer;
    enum PhotonTraceMacro
    {
        RAY_DIFFERENTIAL = 0,
        RAY_CONE = 1,
        RAY_NONE = 2
    };
    PhotonTraceMacro mPhotonTraceMacro = RAY_DIFFERENTIAL;
    std::unordered_map<uint32_t, Raytracer> mPhotonTraceShaderList;
    ref<Texture> mpUniformNoise;

    // update ray density result
    ref<Program> mpUpdateRayDensityProgram;  // ComputeProgram
    ref<ProgramVars> mpUpdateRayDensityVars; // ComputeVars
    ref<ComputeState> mpUpdateRayDensityState;

    // analyse trace result
    ref<Program> mpAnalyseProgram;      // ComputeProgram
    ref<ProgramVars> mpAnalyseVars;     // ComputeVars
    ref<ComputeState> mpAnalyseState;
    ref<Buffer> mpRayArgumentBuffer;    // StructuredBuffer

    // generate ray count
    ref<Program> mpGenerateRayCountProgram;     // ComputeProgram
    ref<ProgramVars> mpGenerateRayCountVars;    // ComputeVars
    ref<ComputeState> mpGenerateRayCountState;
    ref<Buffer> mpRayCountQuadTree;             // StructuredBuffer

    // generate ray count mipmap
    ref<Program> mpGenerateRayCountMipProgram;  // ComputeProgram
    ref<ProgramVars> mpGenerateRayCountMipVars; // ComputeVars
    ref<ComputeState> mpGenerateRayCountMipState;

    // smooth photon
    ref<Program> mpSmoothProgram;    // ComputeProgram
    ref<ProgramVars> mpSmoothVars;   // ComputeVars
    ref<ComputeState> mpSmoothState;

    // photon scatter
    ref<Program>mpPhotonScatterProgram;             // GraphicsProgram
    ref<ProgramVars> mpPhotonScatterVars;           // GraphicsVars
    ref<GraphicsState> mpPhotonScatterBlendState;
    ref<GraphicsState> mpPhotonScatterNoBlendState;
    ref<Fbo> mpCausticsFbo[2];
    ref<Texture> mpGaussianKernel;
    ref<Sampler> mpLinearSampler;

    // photon gather
#define GATHER_PROCESSING_SHADER_COUNT 4
    ref<Program> mpAllocateTileProgram[GATHER_PROCESSING_SHADER_COUNT];     // ComputeProgram
    ref<ProgramVars> mpAllocateTileVars[GATHER_PROCESSING_SHADER_COUNT];    // ComputeVars
    ref<ComputeState> mpAllocateTileState[GATHER_PROCESSING_SHADER_COUNT];

    ref<Program> mpPhotonGatherProgram;   // ComputeProgram
    ref<ProgramVars> mpPhotonGatherVars;  // ComputeVars
    ref<ComputeState> mpPhotonGatherState;

    ref<Buffer> mpTileIDInfoBuffer;       // StructuredBuffer
    ref<Buffer> mpIDBuffer;               // Buffer
    ref<Buffer> mpIDCounterBuffer;        // Buffer

    // temporal filter
    ref<Program> mpFilterProgram;         // ComputeProgram
    ref<ProgramVars> mpFilterVars;        // ComputeVars
    ref<ComputeState> mpFilterState;

    // spacial filter
    ref<Program> mpSpacialFilterProgram;  // ComputeProgram
    ref<ProgramVars> mpSpacialFilterVars; // ComputeVars
    ref<ComputeState> mpSpacialFilterState;

    // raytrace
    Raytracer mpCausticsTracer;
    ref<Texture> mpRtOut;

    // composite pass
    ref<Sampler> mpPointSampler;
    ref<FullScreenPass> mpCompositePass;
    ref<Texture> mpPhotonCountTex;

    // RT composite pass
    Raytracer mpCompositeTracer;

    // Caustics map
    ref<Buffer> mpPhotonBuffer;     // StructuredBuffer
    ref<Buffer> mpPhotonBuffer2;    // StructuredBuffer
    ref<Buffer> mpRayTaskBuffer;    // StructuredBuffer
    ref<Buffer> mpPixelInfoBuffer;  // StructuredBuffer
    ref<Texture> mpRayDensityTex;

private:
    // Falcor default
    void loadScene(const std::string& filename, const Fbo* pTargetFbo);
    void setPerFrameVars(RenderContext* pRenderContext, const Fbo* pTargetFbo);
    void renderRT(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void renderRaster(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);

    void loadShader();
    Raytracer getPhotonTraceShader();

    void setCommonVars(ProgramVars* pVars, const Fbo* pTargetFbo); //GraphicsVars
    void setPhotonTracingCommonVariable(Raytracer& photonTracer);

    void loadSceneSetting(std::string path);
    void saveSceneSetting(std::string path);

    void createCausticsMap();
    void createGBuffer(int width, int height, GBuffer& gbuffer);

    int2 getTileDim() const;
    float resolutionFactor();
    uint photonMacroToFlags() const;
};

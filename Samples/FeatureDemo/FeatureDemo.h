/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#pragma once
#include "Falcor.h"
#include "SampleTest.h"
#include "FeatureDemoSceneRenderer.h"

using namespace Falcor;

class FeatureDemo : public SampleTest
{
public:
    void onLoad() override;
    void onFrameRender() override;
    void onResizeSwapChain() override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;
    void onGuiRender() override;

private:
    Fbo::SharedPtr mpMainFbo;
    Fbo::SharedPtr mpDepthPassFbo;
    Fbo::SharedPtr mpResolveFbo;
    Fbo::SharedPtr mpPostProcessFbo;

    struct ShadowPass
    {
        bool updateShadowMap = true;
        CascadedShadowMaps::UniquePtr pCsm;
        glm::mat4 camVpAtLastCsmUpdate = glm::mat4();
    };
    ShadowPass mShadowPass;

    //  SkyBox Pass.
    struct
    {
        SkyBox::UniquePtr pEffect;
        DepthStencilState::SharedPtr pDS;
        Sampler::SharedPtr pSampler;
    } mSkyBox;

    //  Lighting Pass.
    struct
    {
        GraphicsVars::SharedPtr pVars;
        GraphicsProgram::SharedPtr pProgram;
        DepthStencilState::SharedPtr pDsState;
        RasterizerState::SharedPtr pNoCullRS;
        BlendState::SharedPtr pAlphaBlendBS;
    } mLightingPass;

    struct
    {
        GraphicsVars::SharedPtr pVars;
        GraphicsProgram::SharedPtr pProgram;
    } mDepthPass;


    //  The Temporal Anti-Aliasing Pass.
    class
    {
    public:
        TemporalAA::UniquePtr pTAA;
        Fbo::SharedPtr getActiveFbo() { return pTAAFbos[activeFboIndex]; }
        Fbo::SharedPtr getInactiveFbo()  { return pTAAFbos[1 - activeFboIndex]; }
        void createFbos(uint32_t width, uint32_t height, const Fbo::Desc & fboDesc)
        {
            pTAAFbos[0] = FboHelper::create2D(width, height, fboDesc);
            pTAAFbos[1] = FboHelper::create2D(width, height, fboDesc);
        }

        void switchFbos() { activeFboIndex = 1 - activeFboIndex; }
        void resetFbos()
        {
            activeFboIndex = 0;
            pTAAFbos[0] = nullptr;
            pTAAFbos[1] = nullptr;
        }

        void resetFboActiveIndex() { activeFboIndex = 0;}

    private:
        Fbo::SharedPtr pTAAFbos[2];
        uint32_t activeFboIndex = 0;
    } mTAA;


    ToneMapping::UniquePtr mpToneMapper;

    struct
    {
        SSAO::UniquePtr pSSAO;
        FullScreenPass::UniquePtr pApplySSAOPass;
        GraphicsVars::SharedPtr pVars;
    } mSSAO;

    void beginFrame();
    void endFrame();
    void depthPass();
    void shadowPass();
    void renderSkyBox();
    void lightingPass();
    void antiAliasing();
    void resolveMSAA();
    void runTAA();
    void postProcess();
    void ambientOcclusion();


    void renderOpaqueObjects();
    void renderTransparentObjects();


    void initSkyBox(const std::string& name);
    void initPostProcess();
    void initLightingPass();
    void initDepthPass();
    void initShadowPass();
    void initSSAO();
    void initEnvMap(const std::string& name);
    void initTAA();

    void initControls();

    GraphicsState::SharedPtr mpState;
    FeatureDemoSceneRenderer::SharedPtr mpSceneRenderer;
    void loadModel(const std::string& filename, bool showProgressBar);
    void loadScene(const std::string& filename, bool showProgressBar);
    void initScene(Scene::SharedPtr pScene);
    void applyCustomSceneVars(const Scene* pScene, const std::string& filename);
    void resetScene();

    void setActiveCameraAspectRatio();
    void setSceneSampler(uint32_t maxAniso);

    Texture::SharedPtr mpEnvMap;
    Sampler::SharedPtr mpSceneSampler;

    struct ProgramControl
    {
        bool enabled;
        bool unsetOnEnabled;
        std::string define;
        std::string value;
    };

    enum ControlID
    {
        SuperSampling,
        EnableSpecAA,
        EnableShadows,
        EnableReflections,
        EnableSSAO,
        EnableHashedAlpha,
        EnableTransparency,
        VisualizeCascades,
        Count
    };


    enum class SamplePattern : uint32_t
    {
        Halton,
        DX11
    };

    enum class AAMode
    {
        MSAA,
        TAA
    };

    float mEnvMapFactorScale = 0.25f;
    float mOpacityScale = 0.5f;
    AAMode mAAMode = AAMode::TAA;
    uint32_t mMSAASampleCount = 4;
    SamplePattern mTAASamplePattern = SamplePattern::Halton;
    void applyAaMode();
    std::vector<ProgramControl> mControls;
    void applyLightingProgramControl(ControlID controlID);

    bool mUseCameraPath = true;
    void applyCameraPathState();
    bool mPerMaterialShader = false;
    bool mEnableDepthPass = true;

    // Testing 
    void onInitializeTesting() override;
    void onBeginTestFrame() override;
};

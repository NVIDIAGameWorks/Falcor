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
#include "FeatureDemo.h"

//  Halton Sampler Pattern.
static const float kHaltonSamplePattern[8][2] = { { 1.0f / 2.0f - 0.5f, 1.0f / 3.0f - 0.5f },
{ 1.0f / 4.0f - 0.5f, 2.0f / 3.0f - 0.5f },
{ 3.0f / 4.0f - 0.5f, 1.0f / 9.0f - 0.5f },
{ 1.0f / 8.0f - 0.5f, 4.0f / 9.0f - 0.5f },
{ 5.0f / 8.0f - 0.5f, 7.0f / 9.0f - 0.5f },
{ 3.0f / 8.0f - 0.5f, 2.0f / 9.0f - 0.5f },
{ 7.0f / 8.0f - 0.5f, 5.0f / 9.0f - 0.5f },
{ 0.5f / 8.0f - 0.5f, 8.0f / 9.0f - 0.5f } };

//  DirectX 11 Sample Pattern.
static const float kDX11SamplePattern[8][2] = { { 1.0f / 16.0f, -3.0f / 16.0f },
{ -1.0f / 16.0f, 3.0f / 16.0f },
{ 5.0f / 16.0f, 1.0f / 16.0f },
{ -3.0f / 16.0f, -5.0f / 16.0f },
{ -5.0f / 16.0f, 5.0f / 16.0f },
{ -7.0f / 16.0f, -1.0f / 16.0f },
{ 3.0f / 16.0f, 7.0f / 16.0f },
{ 7.0f / 16.0f, -7.0f / 16.0f } };

void FeatureDemo::initDepthPass()
{
    mDepthPass.pProgram = GraphicsProgram::createFromFile("", "DepthPass.ps.slang");
    mDepthPass.pVars = GraphicsVars::create(mDepthPass.pProgram->getActiveVersion()->getReflector());
}

void FeatureDemo::initLightingPass()
{
    mLightingPass.pProgram = GraphicsProgram::createFromFile("FeatureDemo.vs.slang", "FeatureDemo.ps.slang");
    mLightingPass.pProgram->addDefine("_LIGHT_COUNT", std::to_string(mpSceneRenderer->getScene()->getLightCount()));
    initControls();
    mLightingPass.pVars = GraphicsVars::create(mLightingPass.pProgram->getActiveVersion()->getReflector());
    
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthTest(true).setStencilTest(false).setDepthWriteMask(false).setDepthFunc(DepthStencilState::Func::LessEqual);
    mLightingPass.pDsState = DepthStencilState::create(dsDesc);

    RasterizerState::Desc rsDesc;
    rsDesc.setCullMode(RasterizerState::CullMode::None);
    mLightingPass.pNoCullRS = RasterizerState::create(rsDesc);

    BlendState::Desc bsDesc;
    bsDesc.setRtBlend(0, true).setRtParams(0, BlendState::BlendOp::Add, BlendState::BlendOp::Add, BlendState::BlendFunc::SrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha, BlendState::BlendFunc::One, BlendState::BlendFunc::Zero);
    mLightingPass.pAlphaBlendBS = BlendState::create(bsDesc);
}

void FeatureDemo::initShadowPass()
{
    mShadowPass.pCsm = CascadedShadowMaps::create(2048, 2048, mpSceneRenderer->getScene()->getLight(0), mpSceneRenderer->getScene()->shared_from_this(), 4);
    mShadowPass.pCsm->setFilterMode(CsmFilterEvsm2);
    mShadowPass.pCsm->setVsmLightBleedReduction(0.3f);
    mShadowPass.pCsm->setVsmMaxAnisotropy(4);
    mShadowPass.pCsm->setEvsmBlur(7, 3);
}

void FeatureDemo::initSSAO()
{
    mSSAO.pSSAO = SSAO::create(uvec2(1024));
    mSSAO.pApplySSAOPass = FullScreenPass::create("ApplyAO.ps.slang");
    mSSAO.pVars = GraphicsVars::create(mSSAO.pApplySSAOPass->getProgram()->getActiveVersion()->getReflector());

    Sampler::Desc desc;
    desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mSSAO.pVars->setSampler("gSampler", Sampler::create(desc));
}

void FeatureDemo::setSceneSampler(uint32_t maxAniso)
{
    Scene* pScene = const_cast<Scene*>(mpSceneRenderer->getScene().get());
    Sampler::Desc samplerDesc;
    samplerDesc.setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap).setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear).setMaxAnisotropy(maxAniso);
    mpSceneSampler = Sampler::create(samplerDesc);
    pScene->bindSampler(mpSceneSampler);
}

void FeatureDemo::applyCustomSceneVars(const Scene* pScene, const std::string& filename)
{
    std::string folder = getDirectoryFromFile(filename);

    Scene::UserVariable var = pScene->getUserVariable("sky_box");
    if (var.type == Scene::UserVariable::Type::String) initSkyBox(folder + '/' + var.str);

    var = pScene->getUserVariable("opacity_scale");
    if (var.type == Scene::UserVariable::Type::Double) mOpacityScale = (float)var.d64;
}

void FeatureDemo::initScene(SampleCallbacks* pSample, Scene::SharedPtr pScene)
{
    if (pScene->getCameraCount() == 0)
    {
        // Place the camera above the center, looking slightly downwards
        const Model* pModel = pScene->getModel(0).get();
        Camera::SharedPtr pCamera = Camera::create();

        vec3 position = pModel->getCenter();
        float radius = pModel->getRadius();
        position.y += 0.1f * radius;
        pScene->setCameraSpeed(radius * 0.03f);

        pCamera->setPosition(position);
        pCamera->setTarget(position + vec3(0, -0.3f, -radius));
        pCamera->setDepthRange(0.1f, radius * 10);

        pScene->addCamera(pCamera);
    }

    if (pScene->getLightCount() == 0)
    {
        // Create a directional light
        DirectionalLight::SharedPtr pDirLight = DirectionalLight::create();
        pDirLight->setWorldDirection(vec3(-0.189f, -0.861f, -0.471f));
        pDirLight->setIntensity(vec3(1, 1, 0.985f) * 10.0f);
        pDirLight->setName("DirLight");
        pScene->addLight(pDirLight);
    }

    mpSceneRenderer = FeatureDemoSceneRenderer::create(pScene);
    mpSceneRenderer->setCameraControllerType(SceneRenderer::CameraControllerType::FirstPerson);
    mpSceneRenderer->toggleStaticMaterialCompilation(mPerMaterialShader);
    setSceneSampler(mpSceneSampler ? mpSceneSampler->getMaxAnisotropy() : 4);
    setActiveCameraAspectRatio(pSample->getCurrentFbo()->getWidth(), pSample->getCurrentFbo()->getHeight());
    initDepthPass();
    initLightingPass();
    initShadowPass();
    initSSAO();
    initTAA(pSample);

    mControls[EnableReflections].enabled = pScene->getLightProbeCount() > 0;
    applyLightingProgramControl(ControlID::EnableReflections);
    
    pSample->setCurrentTime(0);
}

void FeatureDemo::resetScene()
{
    mpSceneRenderer = nullptr;
    mSkyBox.pEffect = nullptr;
}

void FeatureDemo::loadModel(SampleCallbacks* pSample, const std::string& filename, bool showProgressBar)
{
    Mesh::resetGlobalIdCounter();
    resetScene();

    ProgressBar::SharedPtr pBar;
    if (showProgressBar)
    {
        pBar = ProgressBar::create("Loading Model");
    }

    Model::SharedPtr pModel = Model::createFromFile(filename.c_str());
    if (!pModel) return;
    Scene::SharedPtr pScene = Scene::create();
    pScene->addModelInstance(pModel, "instance");

    initScene(pSample, pScene);
}

void FeatureDemo::loadScene(SampleCallbacks* pSample, const std::string& filename, bool showProgressBar)
{
    Mesh::resetGlobalIdCounter();
    resetScene();

    ProgressBar::SharedPtr pBar;
    if (showProgressBar)
    {
        pBar = ProgressBar::create("Loading Scene", 100);
    }

    Scene::SharedPtr pScene = Scene::loadFromFile(filename);

    if (pScene != nullptr)
    {
        initScene(pSample, pScene);
        applyCustomSceneVars(pScene.get(), filename);
    }
}

void FeatureDemo::initSkyBox(const std::string& name)
{
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mSkyBox.pSampler = Sampler::create(samplerDesc);
    mSkyBox.pEffect = SkyBox::createFromTexture(name, true, mSkyBox.pSampler);
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(DepthStencilState::Func::Always);
    mSkyBox.pDS = DepthStencilState::create(dsDesc);
}

void FeatureDemo::initLightProbe(const std::string& name)
{
    Scene::SharedPtr pScene = mpSceneRenderer->getScene();

    // Remove existing light probes
    while (pScene->getLightProbeCount() > 0)
    {
        pScene->deleteLightProbe(0);
    }

    // Create new light probe from file
    LightProbe::SharedPtr pLightProbe = LightProbe::create(name, true, true, ResourceFormat::RGBA16Float);
    pLightProbe->setSampler(mpSceneSampler);
    pScene->addLightProbe(pLightProbe);

    mControls[EnableReflections].enabled = true;
    applyLightingProgramControl(ControlID::EnableReflections);
}

void FeatureDemo::initTAA(SampleCallbacks* pSample)
{
    mTAA.pTAA = TemporalAA::create();
    applyAaMode(pSample);
}

void FeatureDemo::initPostProcess()
{
    mpToneMapper = ToneMapping::create(ToneMapping::Operator::Aces);
}

void FeatureDemo::onLoad(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext)
{
    mpState = GraphicsState::create();

    initPostProcess();
//    initializeTesting();
}

void FeatureDemo::renderSkyBox(RenderContext* pContext)
{
    if (mSkyBox.pEffect)
    {
        PROFILE(skyBox);
        mpState->setDepthStencilState(mSkyBox.pDS);
        mSkyBox.pEffect->render(pContext, mpSceneRenderer->getScene()->getActiveCamera().get());
        mpState->setDepthStencilState(nullptr);
    }
}

void FeatureDemo::beginFrame(RenderContext* pContext, Fbo* pCurrentFbo, uint32_t frameId)
{
    pContext->pushGraphicsState(mpState);
    pContext->clearFbo(mpMainFbo.get(), glm::vec4(0.7f, 0.7f, 0.7f, 1.0f), 1, 0, FboAttachmentType::All);
    pContext->clearFbo(mpPostProcessFbo.get(), glm::vec4(), 1, 0, FboAttachmentType::Color);

    if (mAAMode == AAMode::TAA)
    {
        glm::vec2 targetResolution = glm::vec2(pCurrentFbo->getWidth(), pCurrentFbo->getHeight());
        pContext->clearRtv(mpMainFbo->getColorTexture(2)->getRTV().get(), vec4(0));
        pContext->clearFbo(mpResolveFbo.get(), glm::vec4(), 1, 0, FboAttachmentType::Color);

        //  Select the sample pattern and set the camera jitter
        const auto& samplePattern = (mTAASamplePattern == SamplePattern::Halton) ? kHaltonSamplePattern : kDX11SamplePattern;
        static_assert(arraysize(kHaltonSamplePattern) == arraysize(kDX11SamplePattern), "Mismatch in the array size of the sample patterns");
        uint32_t patternIndex = frameId % arraysize(kHaltonSamplePattern);
        mpSceneRenderer->getScene()->getActiveCamera()->setJitter(samplePattern[patternIndex][0] / targetResolution.x, samplePattern[patternIndex][1] / targetResolution.y);
    }
}

void FeatureDemo::endFrame(RenderContext* pContext)
{
    pContext->popGraphicsState();
}

void FeatureDemo::postProcess(RenderContext* pContext, Fbo::SharedPtr pCurrentFbo)
{
    PROFILE(postProcess);
    mpToneMapper->execute(pContext, mpResolveFbo, mControls[EnableSSAO].enabled ? mpPostProcessFbo : pCurrentFbo);
}

void FeatureDemo::depthPass(RenderContext* pContext)
{
    PROFILE(depthPass);
    if (mEnableDepthPass == false) 
    {
        return;
    }

    mpState->setFbo(mpDepthPassFbo);
    mpState->setProgram(mDepthPass.pProgram);
    pContext->setGraphicsVars(mDepthPass.pVars);
    
    auto renderMode = mControls[EnableTransparency].enabled ? FeatureDemoSceneRenderer::Mode::Opaque : FeatureDemoSceneRenderer::Mode::All;
    mpSceneRenderer->setRenderMode(renderMode);
    mpSceneRenderer->renderScene(pContext);
}

void FeatureDemo::lightingPass(RenderContext* pContext, Fbo* pCurrentFbo)
{
    PROFILE(lightingPass);
    mpState->setProgram(mLightingPass.pProgram);
    mpState->setDepthStencilState(mEnableDepthPass ? mLightingPass.pDsState : nullptr);
    pContext->setGraphicsVars(mLightingPass.pVars);
    ConstantBuffer::SharedPtr pCB = mLightingPass.pVars->getConstantBuffer("PerFrameCB");
    pCB["gOpacityScale"] = mOpacityScale;

    if (mControls[ControlID::EnableShadows].enabled)
    {
        pCB["camVpAtLastCsmUpdate"] = mShadowPass.camVpAtLastCsmUpdate;
        mShadowPass.pCsm->setDataIntoGraphicsVars(mLightingPass.pVars, "gCsmData");
    }

    if (mAAMode == AAMode::TAA)
    {
        pContext->clearFbo(mTAA.getActiveFbo().get(), vec4(0.0, 0.0, 0.0, 0.0), 1, 0, FboAttachmentType::Color);
        pCB["gRenderTargetDim"] = glm::vec2(pCurrentFbo->getWidth(), pCurrentFbo->getHeight());
    }

    if(mControls[EnableTransparency].enabled)
    {
        renderOpaqueObjects(pContext);
        renderTransparentObjects(pContext);
    }
    else
    {
        mpSceneRenderer->setRenderMode(FeatureDemoSceneRenderer::Mode::All);
        mpSceneRenderer->renderScene(pContext);
    }
    pContext->flush();
    mpState->setDepthStencilState(nullptr);
}

void FeatureDemo::renderOpaqueObjects(RenderContext* pContext)
{
    mpSceneRenderer->setRenderMode(FeatureDemoSceneRenderer::Mode::Opaque);
    mpSceneRenderer->renderScene(pContext);
}

void FeatureDemo::renderTransparentObjects(RenderContext* pContext)
{
    mpSceneRenderer->setRenderMode(FeatureDemoSceneRenderer::Mode::Transparent);
    mpState->setBlendState(mLightingPass.pAlphaBlendBS);
    mpState->setRasterizerState(mLightingPass.pNoCullRS);
    mpSceneRenderer->renderScene(pContext);
    mpState->setBlendState(nullptr);
    mpState->setRasterizerState(nullptr);
}

void FeatureDemo::resolveMSAA(RenderContext* pContext)
{
    pContext->blit(mpMainFbo->getColorTexture(0)->getSRV(), mpResolveFbo->getRenderTargetView(0));
    pContext->blit(mpMainFbo->getColorTexture(1)->getSRV(), mpResolveFbo->getRenderTargetView(1));
    pContext->blit(mpMainFbo->getDepthStencilTexture()->getSRV(), mpResolveFbo->getRenderTargetView(2));
}

void FeatureDemo::shadowPass(RenderContext* pContext)
{
    PROFILE(shadowPass);
    if (mControls[EnableShadows].enabled && mShadowPass.updateShadowMap)
    {
        mShadowPass.camVpAtLastCsmUpdate = mpSceneRenderer->getScene()->getActiveCamera()->getViewProjMatrix();
        mShadowPass.pCsm->setup(pContext, mpSceneRenderer->getScene()->getActiveCamera().get(), mEnableDepthPass ? mpDepthPassFbo->getDepthStencilTexture() : nullptr);
        pContext->flush();
    }
}

void FeatureDemo::antiAliasing(RenderContext* pContext)
{
    PROFILE(resolveMSAA);
    switch (mAAMode)
    {
    case AAMode::MSAA:
        return resolveMSAA(pContext);
    case AAMode::TAA:
        return runTAA(pContext);
    default:
        should_not_get_here();
    }
}

void FeatureDemo::runTAA(RenderContext* pContext)
{
    //  Get the Current Color and Motion Vectors
    const Texture::SharedPtr pCurColor = mpMainFbo->getColorTexture(0);
    const Texture::SharedPtr pMotionVec = mpMainFbo->getColorTexture(2);

    //  Get the Previous Color
    const Texture::SharedPtr pPrevColor = mTAA.getInactiveFbo()->getColorTexture(0);

    //  Execute the Temporal Anti-Aliasing
    pContext->getGraphicsState()->pushFbo(mTAA.getActiveFbo());
    mTAA.pTAA->execute(pContext, pCurColor, pPrevColor, pMotionVec);
    pContext->getGraphicsState()->popFbo();

    //  Copy over the Anti-Aliased Color Texture
    pContext->blit(mTAA.getActiveFbo()->getColorTexture(0)->getSRV(0, 1), mpResolveFbo->getColorTexture(0)->getRTV());

    //  Copy over the Remaining Texture Data
    pContext->blit(mpMainFbo->getColorTexture(1)->getSRV(), mpResolveFbo->getRenderTargetView(1));
    pContext->blit(mpMainFbo->getDepthStencilTexture()->getSRV(), mpResolveFbo->getRenderTargetView(2));

    //  Swap the Fbos
    mTAA.switchFbos();
}

void FeatureDemo::ambientOcclusion(RenderContext* pContext, Fbo::SharedPtr pCurrentFbo)
{
    PROFILE(ssao);
    if (mControls[EnableSSAO].enabled)
    {
        Texture::SharedPtr pAOMap = mSSAO.pSSAO->generateAOMap(pContext, mpSceneRenderer->getScene()->getActiveCamera().get(), mpResolveFbo->getColorTexture(2), mpResolveFbo->getColorTexture(1));
        mSSAO.pVars->setTexture("gColor", mpPostProcessFbo->getColorTexture(0));
        mSSAO.pVars->setTexture("gAOMap", pAOMap);

        pContext->getGraphicsState()->setFbo(pCurrentFbo);
        pContext->setGraphicsVars(mSSAO.pVars);

        mSSAO.pApplySSAOPass->execute(pContext);
    }
}

// void FeatureDemo::onBeginTestFrame()
// {
//     //  Already exisitng. Is this a problem?
//     if (mCurrentTriggerType == SampleTest::TriggerType::None)
//     {
//        SampleTest::TaskType taskType = (mCurrentTriggerType == SampleTest::TriggerType::Frame) ? mFrameTasks[mCurrentFrameTaskIndex]->mTaskType : mTimeTasks[mCurrentTimeTaskIndex]->mTaskType;
// 
//         mShadowPass.pCsm->setSdsmReadbackLatency(taskType == SampleTest::TaskType::ScreenCaptureTask ? 0 : 1);
//     }
// }

void FeatureDemo::onFrameRender(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext, Fbo::SharedPtr pCurrentFbo)
{
    if (mpSceneRenderer)
    {
        beginFrame(pRenderContext.get(), pCurrentFbo.get(), pSample->getCurrentFrameId());

        {
            PROFILE(updateScene);
            mpSceneRenderer->update(pSample->getCurrentTime());
        }

        depthPass(pRenderContext.get());
        shadowPass(pRenderContext.get());
        mpState->setFbo(mpMainFbo);
        renderSkyBox(pRenderContext.get());
        lightingPass(pRenderContext.get(), pCurrentFbo.get());
        antiAliasing(pRenderContext.get());
        postProcess(pRenderContext.get(), pCurrentFbo);
        ambientOcclusion(pRenderContext.get(), pCurrentFbo);
        endFrame(pRenderContext.get());
    }
    else
    {
        pRenderContext->clearFbo(pCurrentFbo.get(), vec4(0.2f, 0.4f, 0.5f, 1), 1, 0);
    }
}

void FeatureDemo::applyCameraPathState()
{
    const Scene* pScene = mpSceneRenderer->getScene().get();
    if(pScene->getPathCount())
    {
        mUseCameraPath = mUseCameraPath;
        if (mUseCameraPath)
        {
            pScene->getPath(0)->attachObject(pScene->getActiveCamera());
        }
        else
        {
            pScene->getPath(0)->detachObject(pScene->getActiveCamera());
        }
    }
}

bool FeatureDemo::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    if (mpSceneRenderer && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        switch (keyEvent.key)
        {
        case KeyboardEvent::Key::Minus:
            mUseCameraPath = !mUseCameraPath;
            applyCameraPathState();
            return true;
        case KeyboardEvent::Key::O:
            mPerMaterialShader = !mPerMaterialShader;
            mpSceneRenderer->toggleStaticMaterialCompilation(mPerMaterialShader);
            return true;
        }
    }

    return mpSceneRenderer ? mpSceneRenderer->onKeyEvent(keyEvent) : false;
}

void FeatureDemo::onDroppedFile(SampleCallbacks* pSample, const std::string& filename)
{
    if (hasSuffix(filename, ".fscene", false) == false)
    {
        msgBox("You can only drop a scene file into the window");
        return;
    }
    loadScene(pSample, filename, true);
}

bool FeatureDemo::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return mpSceneRenderer ? mpSceneRenderer->onMouseEvent(mouseEvent) : true;
}

void FeatureDemo::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    // Create the post-process FBO and AA resolve Fbo
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA8UnormSrgb);
    mpPostProcessFbo = FboHelper::create2D(width, height, fboDesc);
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setColorTarget(1, ResourceFormat::RGBA8Unorm).setColorTarget(2, ResourceFormat::R32Float);
    mpResolveFbo = FboHelper::create2D(width, height, fboDesc);

    applyAaMode(pSample);
    
    if(mpSceneRenderer)
    {
        setActiveCameraAspectRatio(width, height);
    }
}

void FeatureDemo::setActiveCameraAspectRatio(uint32_t w, uint32_t h)
{
    mpSceneRenderer->getScene()->getActiveCamera()->setAspectRatio((float)w / (float)h);
}

// void FeatureDemo::onInitializeTesting()
// {
//     std::vector<ArgList::Arg> model = mArgList.getValues("loadmodel");
//     if (!model.empty())
//     {
//         loadModel(model[0].asString(), false);
//     }
// 
//     std::vector<ArgList::Arg> scene = mArgList.getValues("loadscene");
//     if (!scene.empty())
//     {
//         loadScene(scene[0].asString(), false);
//     }
// 
//     std::vector<ArgList::Arg> cameraPos = mArgList.getValues("camerapos");
//     if (!cameraPos.empty())
//     {
//         mpSceneRenderer->getScene()->getActiveCamera()->setPosition(glm::vec3(cameraPos[0].asFloat(), cameraPos[1].asFloat(), cameraPos[2].asFloat()));
//     }
// 
//     std::vector<ArgList::Arg> cameraTarget = mArgList.getValues("cameratarget");
//     if (!cameraTarget.empty())
//     {
//         mpSceneRenderer->getScene()->getActiveCamera()->setTarget(glm::vec3(cameraTarget[0].asFloat(), cameraTarget[1].asFloat(), cameraTarget[2].asFloat()));
//     }
// }

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    FeatureDemo::UniquePtr pRenderer = std::make_unique<FeatureDemo>();
    SampleConfig config;
    config.windowDesc.title = "Falcor Feature Demo";
    config.windowDesc.resizableWindow = false;
#ifdef _WIN32
    Sample::run(config, pRenderer);
#else
    Sample::run(config, pRenderer, (uint32_t)argc, argv);
#endif
    return 0;
}

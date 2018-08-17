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
#include "StereoRendering.h"

const std::string StereoRendering::skDefaultScene = "Arcade/Arcade.fscene";

static const glm::vec4 kClearColor(0.38f, 0.52f, 0.10f, 1);


void StereoRendering::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (pGui->addButton("Load Scene"))
    {
        loadScene();
    }

    if(VRSystem::instance())
    {
        pGui->addCheckBox("Display VR FBO", mShowStereoViews);
    }

    if (pGui->addDropdown("Submission Mode", mSubmitModeList, (uint32_t&)mRenderMode))
    {
        setRenderMode();
    }
}

bool displaySpsWarning()
{
#ifdef FALCOR_D3D12
    logWarning("The sample will run faster if you use NVIDIA's Single Pass Stereo\nIf you have an NVIDIA GPU, download the NVAPI SDK and enable NVAPI support in FalcorConfig.h.\nCheck the readme for instructions");
#endif
    return false;
}

void StereoRendering::initVR(Fbo* pTargetFbo)
{
    mSubmitModeList.clear();
    mSubmitModeList.push_back({ (int)RenderMode::Mono, "Render to Screen" });

    if (VRSystem::instance())
    {
        // Create the FBOs
        Fbo::Desc vrFboDesc;

        vrFboDesc.setColorTarget(0, pTargetFbo->getColorTexture(0)->getFormat());
        vrFboDesc.setDepthStencilTarget(pTargetFbo->getDepthStencilTexture()->getFormat());

        mpVrFbo = VrFbo::create(vrFboDesc);

        mSubmitModeList.push_back({ (int)RenderMode::Stereo, "Stereo" });

        if (mSPSSupported)
        {
            mSubmitModeList.push_back({ (int)RenderMode::SinglePassStereo, "Single Pass Stereo" });
        }
        else
        {
            displaySpsWarning();
        }
    }
    else
    {
        msgBox("Can't initialize the VR system. Make sure that your HMD is connected properly");
    }
}

void StereoRendering::submitStereo(RenderContext* pContext, Fbo::SharedPtr pTargetFbo, bool singlePassStereo)
{
    PROFILE(STEREO);
    VRSystem::instance()->refresh();

    // Clear the FBO
    pContext->clearFbo(mpVrFbo->getFbo().get(), kClearColor, 1.0f, 0, FboAttachmentType::All);
    
    // update state
    if (singlePassStereo)
    {
        mpGraphicsState->setProgram(mpMonoSPSProgram);
        pContext->setGraphicsVars(mpMonoSPSVars);
    }
    else
    {
        mpGraphicsState->setProgram(mpStereoProgram);
        pContext->setGraphicsVars(mpStereoVars);
    }
    mpGraphicsState->setFbo(mpVrFbo->getFbo());
    pContext->pushGraphicsState(mpGraphicsState);

    // Render
    mpSceneRenderer->renderScene(pContext);

    // Restore the state
    pContext->popGraphicsState();

    // Submit the views and display them
    mpVrFbo->submitToHmd(pContext);
    blitTexture(pContext, pTargetFbo.get(), mpVrFbo->getEyeResourceView(VRDisplay::Eye::Left), 0);
    blitTexture(pContext, pTargetFbo.get(), mpVrFbo->getEyeResourceView(VRDisplay::Eye::Right), pTargetFbo->getWidth() / 2);
}

void StereoRendering::submitToScreen(RenderContext* pContext, Fbo::SharedPtr pTargetFbo)
{
    mpGraphicsState->setProgram(mpMonoSPSProgram);
    mpGraphicsState->setFbo(pTargetFbo);
    pContext->setGraphicsState(mpGraphicsState);
    pContext->setGraphicsVars(mpMonoSPSVars);
    mpSceneRenderer->renderScene(pContext);
}

void StereoRendering::setRenderMode()
{
    if(mpScene)
    {
        mpMonoSPSProgram->removeDefine("_SINGLE_PASS_STEREO");

        mpGraphicsState->toggleSinglePassStereo(false);
        switch(mRenderMode)
        {
        case RenderMode::SinglePassStereo:
            mpMonoSPSProgram->addDefine("_SINGLE_PASS_STEREO");
            mpGraphicsState->toggleSinglePassStereo(true);
            mpSceneRenderer->setCameraControllerType(SceneRenderer::CameraControllerType::Hmd);
            break;
        case RenderMode::Stereo:
            mpSceneRenderer->setCameraControllerType(SceneRenderer::CameraControllerType::Hmd);
            break;
        case RenderMode::Mono:
            mpSceneRenderer->setCameraControllerType(SceneRenderer::CameraControllerType::SixDof);
            break;
        }
    }
}

void StereoRendering::loadScene()
{
    std::string filename;
    if(openFileDialog("Scene files\0*.fscene\0\0", filename))
    {
        loadScene(filename);
    }
}

void StereoRendering::loadScene(const std::string& filename)
{
    mpScene = Scene::loadFromFile(filename);
    mpSceneRenderer = SceneRenderer::create(mpScene);
    mpMonoSPSProgram = GraphicsProgram::createFromFile("StereoRendering.ps.hlsl", "", "main");
    GraphicsProgram::Desc progDesc;
    progDesc.addShaderLibrary("StereoRendering.vs.hlsl").vsEntry("main").addShaderLibrary("StereoRendering.ps.hlsl").psEntry("main").addShaderLibrary("StereoRendering.gs.hlsl").gsEntry("main");
    mpStereoProgram = GraphicsProgram::create(progDesc);

    setRenderMode();
    mpMonoSPSVars = GraphicsVars::create(mpMonoSPSProgram->getReflector());
    mpStereoVars = GraphicsVars::create(mpStereoProgram->getReflector());

    for (uint32_t m = 0; m < mpScene->getModelCount(); m++)
    {
        mpScene->getModel(m)->bindSamplerToMaterials(mpTriLinearSampler);
    }
}

void StereoRendering::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mSPSSupported = gpDevice->isExtensionSupported("VK_NVX_multiview_per_view_attributes");

    initVR(pSample->getCurrentFbo().get());

    mpGraphicsState = GraphicsState::create();
    setRenderMode();

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpTriLinearSampler = Sampler::create(samplerDesc);

    loadScene(skDefaultScene);
}

void StereoRendering::blitTexture(RenderContext* pContext, Fbo* pTargetFbo, Texture::SharedPtr pTexture, uint32_t xStart)
{
    if(mShowStereoViews)
    {
        uvec4 dstRect;
        dstRect.x = xStart;
        dstRect.y = 0;
        dstRect.z = xStart + (pTargetFbo->getWidth() / 2);
        dstRect.w = pTargetFbo->getHeight();
        pContext->blit(pTexture->getSRV(0, 1, 0, 1), pTargetFbo->getRenderTargetView(0), uvec4(-1), dstRect);
    }
}

void StereoRendering::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    static uint32_t frameCount = 0u;

    pRenderContext->clearFbo(pTargetFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);

    if(mpSceneRenderer)
    {      
        mpSceneRenderer->update(pSample->getCurrentTime());

        switch(mRenderMode)
        {
        case RenderMode::Mono:
            submitToScreen(pRenderContext.get(), pTargetFbo);
            break;
        case RenderMode::SinglePassStereo:
            submitStereo(pRenderContext.get(), pTargetFbo, true);
            break;
        case RenderMode::Stereo:
            submitStereo(pRenderContext.get(), pTargetFbo, false);
            break;
        default:
            should_not_get_here();
        }
    }

    std::string message = pSample->getFpsMsg();
    message += "\nFrame counter: " + std::to_string(frameCount);

    pSample->renderText(message, glm::vec2(10, 10));

    frameCount++;
}

bool StereoRendering::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    if(keyEvent.key == KeyboardEvent::Key::Space && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        if (VRSystem::instance())
        {
            // Cycle through modes
            uint32_t nextMode = (uint32_t)mRenderMode + 1;
            mRenderMode = (RenderMode)(nextMode % (mSPSSupported ? 3 : 2));
            setRenderMode();
            return true;
        }
    }
    return mpSceneRenderer ? mpSceneRenderer->onKeyEvent(keyEvent) : false;
}

bool StereoRendering::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return mpSceneRenderer ? mpSceneRenderer->onMouseEvent(mouseEvent) : false;
}

void StereoRendering::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    initVR(pSample->getCurrentFbo().get());
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    StereoRendering::UniquePtr pRenderer = std::make_unique<StereoRendering>();
    SampleConfig config;
    config.windowDesc.title = "Stereo Rendering";
    config.windowDesc.height = 1024;
    config.windowDesc.width = 1600;
    config.windowDesc.resizableWindow = true;
    config.deviceDesc.enableVR = true;
#ifdef FALCOR_VK
    config.deviceDesc.enableDebugLayer = false; // OpenVR requires an extension that the debug layer doesn't recognize. It causes the application to crash
#endif

#ifdef _WIN32
    Sample::run(config, pRenderer);
#else
    config.argc = (uint32_t)argc;
    config.argv = argv;
    Sample::run(config, pRenderer);
#endif
    return 0;
}

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

static const glm::vec4 kClearColor(0.38f, 0.52f, 0.10f, 1);


void StereoRendering::onGuiRender()
{
    if (mpGui->addButton("Load Scene"))
    {
        loadScene();
    }

    if(VRSystem::instance())
    {
        mpGui->addCheckBox("Display VR FBO", mShowStereoViews);
    }

    if (mpGui->addDropdown("Submission Mode", mSubmitModeList, (uint32_t&)mRenderMode))
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

void StereoRendering::initVR()
{
    mSubmitModeList.clear();
    mSubmitModeList.push_back({ (int)RenderMode::Mono, "Render to Screen" });

    if (VRSystem::instance())
    {
        // Create the FBOs
        Fbo::Desc vrFboDesc;

        vrFboDesc.setColorTarget(0, mpDefaultFBO->getColorTexture(0)->getFormat());
        vrFboDesc.setDepthStencilTarget(mpDefaultFBO->getDepthStencilTexture()->getFormat());

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

void StereoRendering::submitStereo(bool singlePassStereo)
{
    PROFILE(STEREO);
    VRSystem::instance()->refresh();

    // Clear the FBO
    mpRenderContext->clearFbo(mpVrFbo->getFbo().get(), kClearColor, 1.0f, 0, FboAttachmentType::All);
    
    // update state
    if (singlePassStereo)
    {
        mpGraphicsState->setProgram(mpMonoSPSProgram);
        mpRenderContext->setGraphicsVars(mpMonoSPSVars);
    }
    else
    {
        mpGraphicsState->setProgram(mpStereoProgram);
        mpRenderContext->setGraphicsVars(mpStereoVars);
    }
    mpGraphicsState->setFbo(mpVrFbo->getFbo());
    mpRenderContext->pushGraphicsState(mpGraphicsState);

    // Render
    mpSceneRenderer->renderScene(mpRenderContext.get());

    // Restore the state
    mpRenderContext->popGraphicsState();

    // Submit the views and display them
    mpVrFbo->submitToHmd(mpRenderContext.get());
    blitTexture(mpVrFbo->getEyeResourceView(VRDisplay::Eye::Left), 0);
    blitTexture(mpVrFbo->getEyeResourceView(VRDisplay::Eye::Right), mpDefaultFBO->getWidth() / 2);
}

void StereoRendering::submitToScreen()
{
    mpGraphicsState->setProgram(mpMonoSPSProgram);
    mpGraphicsState->setFbo(mpDefaultFBO);
    mpRenderContext->setGraphicsState(mpGraphicsState);
    mpRenderContext->setGraphicsVars(mpMonoSPSVars);
    mpSceneRenderer->renderScene(mpRenderContext.get());
}

void StereoRendering::setRenderMode()
{
    if(mpScene)
    {
        std::string lights;
        getSceneLightString(mpScene.get(), lights);
        mpMonoSPSProgram->addDefine("_LIGHT_SOURCES", lights);
        mpStereoProgram->addDefine("_LIGHT_SOURCES", lights);
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
    mpMonoSPSProgram = GraphicsProgram::createFromFile("", appendShaderExtension("StereoRendering.ps"));
    mpStereoProgram = GraphicsProgram::createFromFile(appendShaderExtension("StereoRendering.vs"), appendShaderExtension("StereoRendering.ps"), appendShaderExtension("StereoRendering.gs"), "", "");

    setRenderMode();
    mpMonoSPSVars = GraphicsVars::create(mpMonoSPSProgram->getActiveVersion()->getReflector());
    mpStereoVars = GraphicsVars::create(mpStereoProgram->getActiveVersion()->getReflector());

    for (uint32_t m = 0; m < mpScene->getModelCount(); m++)
    {
        mpScene->getModel(m)->bindSamplerToMaterials(mpTriLinearSampler);
    }
}

void StereoRendering::onLoad()
{
    mSPSSupported = gpDevice->isExtensionSupported("VK_NVX_multiview_per_view_attributes");

    initVR();

    mpGraphicsState = GraphicsState::create();
    setRenderMode();

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpTriLinearSampler = Sampler::create(samplerDesc);
}

void StereoRendering::blitTexture(Texture::SharedPtr pTexture, uint32_t xStart)
{
    if(mShowStereoViews)
    {
        uvec4 dstRect;
        dstRect.x = xStart;
        dstRect.y = 0;
        dstRect.z = xStart + (mpDefaultFBO->getWidth() / 2);
        dstRect.w = mpDefaultFBO->getHeight();
        mpRenderContext->blit(pTexture->getSRV(0, 1, 0, 1), mpDefaultFBO->getRenderTargetView(0), uvec4(-1), dstRect);
    }
}

void StereoRendering::onFrameRender()
{
    static uint32_t frameCount = 0u;

    mpRenderContext->clearFbo(mpDefaultFBO.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);

    if(mpSceneRenderer)
    {      
        mpSceneRenderer->update(mCurrentTime);

        switch(mRenderMode)
        {
        case RenderMode::Mono:
            submitToScreen();
            break;
        case RenderMode::SinglePassStereo:
            submitStereo(true);
            break;
        case RenderMode::Stereo:
            submitStereo(false);
            break;
        default:
            should_not_get_here();
        }
    }

    std::string message = getFpsMsg();
    message += "\nFrame counter: " + std::to_string(frameCount);

    renderText(message, glm::vec2(10, 10));

    frameCount++;
}

bool StereoRendering::onKeyEvent(const KeyboardEvent& keyEvent)
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

bool StereoRendering::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpSceneRenderer ? mpSceneRenderer->onMouseEvent(mouseEvent) : false;
}

void StereoRendering::onDataReload()
{

}

void StereoRendering::onResizeSwapChain()
{
    initVR();
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    StereoRendering sample;
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
    sample.run(config);
#else
    sample.run(config, (uint32_t)argc, argv);
#endif
    return 0;
}

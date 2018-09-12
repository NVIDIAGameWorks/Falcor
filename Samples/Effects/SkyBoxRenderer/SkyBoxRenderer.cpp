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
#include "SkyBoxRenderer.h"

const std::string SkyBoxRenderer::skDefaultSkyBoxTexture = "Cubemaps/Sorsele3/Sorsele3.dds";

void SkyBoxRenderer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (pGui->addButton("Load TexCube"))
    {
        loadTexture();
    }
    float s = mpSkybox->getScale();
    if (pGui->addFloatVar("Cubemap Scale", s, 0.01f, FLT_MAX, 0.01f))
    {
        mpSkybox->setScale(s);
    }
}

void SkyBoxRenderer::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mpCamera = Camera::create();
    mpCameraController = SixDoFCameraController::SharedPtr(new SixDoFCameraController);
    mpCameraController->attachCamera(mpCamera);
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpTriLinearSampler = Sampler::create(samplerDesc);

    mpSkybox = SkyBox::create(skDefaultSkyBoxTexture, true, mpTriLinearSampler);
}

void SkyBoxRenderer::loadTexture()
{
    std::string filename;
    if(openFileDialog("DDS files\0*.dds\0\0", filename))
    {
        mpSkybox = SkyBox::create(filename, true, mpTriLinearSampler);
    }
}

void SkyBoxRenderer::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if(mpSkybox)
    {
        mpCameraController->update();
        mpSkybox->render(pRenderContext.get(), mpCamera.get());
    }
}

bool SkyBoxRenderer::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    return mpCameraController->onKeyEvent(keyEvent);
}

bool SkyBoxRenderer::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return mpCameraController->onMouseEvent(mouseEvent);
}

void SkyBoxRenderer::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    mpCamera->setFocalLength(60.0f);
    mpCamera->setAspectRatio((float)width / (float)height);
    mpCamera->setDepthRange(0.01f, 1000);
}

 void SkyBoxRenderer::onInitializeTesting(SampleCallbacks* pSample)
 {
     auto argList = pSample->getArgList();
     std::vector<ArgList::Arg> viewFrames = argList.getValues("changeView");
     if (!viewFrames.empty())
     {
         mChangeViewFrames.resize(viewFrames.size());
         for (uint32_t i = 0; i < viewFrames.size(); ++i)
         {
             mChangeViewFrames[i] = viewFrames[i].asUint();
         }
     }
 
     mChangeViewIt = mChangeViewFrames.begin();
 }

 void SkyBoxRenderer::onEndTestFrame(SampleCallbacks* pSample, SampleTest* pSampleTest)
 {
     //initial target is (0, 0, -1)
     static uint32_t targetIndex = 0;
     static const uint32_t numTargets = 5;
     static const vec3 targets[numTargets] = {
         vec3(0,  0, 1),
         vec3(0.1,  0.9, 0), //camera doesn't like looking directly up or down
         vec3(-0.1, -0.9, 0),
         vec3(1,  0, 0),
         vec3(-1, 0, 0) 
     };
 
     uint64_t frameId = pSample->getFrameID();
     if (mChangeViewIt != mChangeViewFrames.end() && frameId >= *mChangeViewIt)
     {
         ++mChangeViewIt;
         mpCamera->setTarget(targets[targetIndex]);
         ++targetIndex;
         //wrap around so it doesn't crash if too many args are given
         if (targetIndex == numTargets)
         {
             targetIndex = 0;
         }
     }
 }

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    SkyBoxRenderer::UniquePtr pRenderer = std::make_unique<SkyBoxRenderer>();
    SampleConfig config;
    config.windowDesc.title = "Skybox Sample";
#ifdef _WIN32
    Sample::run(config, pRenderer);
#else
    config.argc = (uint32_t)argc;
    config.argv = argv;
    Sample::run(config, pRenderer);
#endif
    return 0;
}

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
#include "HDRToneMapping.h"

using namespace Falcor;

const Gui::DropdownList HDRToneMapping::kImageList = { { HdrImage::EveningSun, "Evening Sun" },
                                                    { HdrImage::AtTheWindow, "Window" },
                                                    { HdrImage::OvercastDay, "Overcast Day" } };

void HDRToneMapping::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    //Create model and camera
    mpTeapot = Model::createFromFile("teapot.obj");
    mpCamera = Camera::create();
    float nearZ = 0.1f;
    float farZ = mpTeapot->getRadius() * 1000;
    mpCamera->setDepthRange(nearZ, farZ);

    //Setup controller
    mCameraController.attachCamera(mpCamera);
    mCameraController.setModelParams(mpTeapot->getCenter(), mpTeapot->getRadius(), 2.0f);    
    
    //Program
    mpMainProg = GraphicsProgram::createFromFile("HDRToneMapping.hlsl", "vs", "ps");
    mpProgramVars = GraphicsVars::create(mpMainProg->getReflector());
    mpGraphicsState = GraphicsState::create();
    mpGraphicsState->setFbo(pSample->getCurrentFbo());
    
    //Sampler
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpTriLinearSampler = Sampler::create(samplerDesc);
    mpProgramVars->setSampler("gSampler", mpTriLinearSampler);

    mpToneMapper = ToneMapping::create(ToneMapping::Operator::HableUc2);
    mpToneMapper->setExposureKey(0.104f);
    mLightIntensity = 2.5f;

    loadImage();
}

void HDRToneMapping::loadImage()
{
    std::string filename;
    switch(mHdrImageIndex)
    {
    case HdrImage::AtTheWindow:
        filename = "LightProbes/20060807_wells6_hd.hdr";
        break;
    case HdrImage::EveningSun:
        filename = "LightProbes/hallstatt4_hd.hdr";
        break;
    case HdrImage::OvercastDay:
        filename = "LightProbes/20050806-03_hd.hdr";
        break;
    }

    mHdrImage = createTextureFromFile(filename, false, false, Resource::BindFlags::ShaderResource);
    mpSkyBox = SkyBox::create(mHdrImage, mpTriLinearSampler);
}

void HDRToneMapping::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    uint32_t uHdrIndex = static_cast<uint32_t>(mHdrImageIndex);
    if (pGui->addDropdown("HdrImage", kImageList, uHdrIndex))
    {
        mHdrImageIndex = static_cast<HdrImage>(uHdrIndex);
        loadImage();
    }
    pGui->addFloatVar("Surface Roughness", mSurfaceRoughness, 0.01f, 1000, 0.01f);
    pGui->addFloatVar("Light Intensity", mLightIntensity, 0.5f, FLT_MAX, 0.1f);
    mpToneMapper->renderUI(pGui, "HDR");
}

void HDRToneMapping::renderTeapot(RenderContext* pContext)
{
    //Update vars
    glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
    ConstantBuffer::SharedPtr pPerFrameCB = mpProgramVars["PerFrameCB"];
    pPerFrameCB["gWorldMat"] = glm::mat4();
    pPerFrameCB["gWvpMat"] = wvp;
    pPerFrameCB["gEyePosW"] = mpCamera->getPosition();
    pPerFrameCB["gLightIntensity"] = mLightIntensity;
    pPerFrameCB["gSurfaceRoughness"] = mSurfaceRoughness;
    mpProgramVars->setTexture("gEnvMap", mHdrImage);

    //Set Gfx state
    mpGraphicsState->setVao(mpTeapot->getMesh(0)->getVao());
    mpGraphicsState->setProgram(mpMainProg);
    pContext->setGraphicsVars(mpProgramVars);
    pContext->drawIndexed(mpTeapot->getMesh(0)->getIndexCount(), 0, 0);
}

void HDRToneMapping::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(mpHdrFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    mCameraController.update();

    //Render teapot to hdr fbo
    mpGraphicsState->pushFbo(mpHdrFbo);
    pRenderContext->setGraphicsState(mpGraphicsState);
    renderTeapot(pRenderContext.get());
    mpSkyBox->render(pRenderContext.get(), mpCamera.get());
    mpGraphicsState->popFbo();

    //Run tone mapping
    mpToneMapper->execute(pRenderContext.get(), mpHdrFbo->getColorTexture(0), pTargetFbo);

    std::string txt = pSample->getFpsMsg() + '\n';
    pSample->renderText(txt, glm::vec2(10, 10));
}

void HDRToneMapping::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    //Camera aspect 
    mpCamera->setFocalLength(21.0f);
    float aspectRatio = (float(width )/ float(height));
    mpCamera->setAspectRatio(aspectRatio);

    //recreate hdr fbo
    ResourceFormat format = ResourceFormat::RGBA32Float;
    Fbo::Desc desc;
    desc.setDepthStencilTarget(ResourceFormat::D16Unorm);
    desc.setColorTarget(0u, format);
    mpHdrFbo = FboHelper::create2D(width, height, desc);
}

bool HDRToneMapping::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    return mCameraController.onKeyEvent(keyEvent);
}

bool HDRToneMapping::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return mCameraController.onMouseEvent(mouseEvent);
}

 void HDRToneMapping::onInitializeTesting(SampleCallbacks* pSample)
 {
     auto argList = pSample->getArgList();
     std::vector<ArgList::Arg> modeFrames = argList.getValues("changeMode");
     if (!modeFrames.empty())
     {
         mChangeModeFrames.resize(modeFrames.size());
         for (uint32_t i = 0; i < modeFrames.size(); ++i)
         {
             mChangeModeFrames[i] = modeFrames[i].asUint();
         }
     }
 
     mChangeModeIt = mChangeModeFrames.begin();
     mToneMapOperatorIndex = 0;
     mHdrImageIndex = HdrImage::EveningSun;
     mpToneMapper->setOperator(ToneMapping::Operator::Clamp);
 }

 void HDRToneMapping::onEndTestFrame(SampleCallbacks* pSample, SampleTest* pSampleTest)
 {
     uint64_t frameId = pSample->getFrameID();
     if (mChangeModeIt != mChangeModeFrames.end() && frameId >= *mChangeModeIt)
     {
         ++mChangeModeIt;
         if (mToneMapOperatorIndex == static_cast<uint32_t>(ToneMapping::Operator::HableUc2))
         {
             //Done all operators on this image, go to next image
             mToneMapOperatorIndex = 0;
             mHdrImageIndex = static_cast<HdrImage>(min(mHdrImageIndex + 1u, static_cast<uint32_t>(AtTheWindow)));
             loadImage();
         }
         else
         {
             //Next operator
             ++mToneMapOperatorIndex;
         }
         mpToneMapper->setOperator(static_cast<ToneMapping::Operator>(mToneMapOperatorIndex));
     }
 }

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    HDRToneMapping::UniquePtr pRenderer = std::make_unique<HDRToneMapping>();
    SampleConfig config;
    config.windowDesc.title = "Post Processing";
#ifdef _WIN32
    Sample::run(config, pRenderer);
#else
    config.argc = (uint32_t)argc;
    config.argv = argv;
    Sample::run(config, pRenderer);
#endif
    return 0;
}

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
#include "PostProcess.h"

using namespace Falcor;

const Gui::DropdownList PostProcess::kImageList = { { HdrImage::EveningSun, "Evening Sun" },
                                                    { HdrImage::AtTheWindow, "Window" },
                                                    { HdrImage::OvercastDay, "Overcast Day" } };

void PostProcess::onLoad()
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
    mpMainProg = GraphicsProgram::createFromFile(appendShaderExtension("PostProcess.vs"), appendShaderExtension("PostProcess.ps"));
    mpProgramVars = GraphicsVars::create(mpMainProg->getActiveVersion()->getReflector());
    mpGraphicsState = GraphicsState::create();
    mpGraphicsState->setFbo(mpDefaultFBO);
    
    //Sampler
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpTriLinearSampler = Sampler::create(samplerDesc);
    mpProgramVars->setSampler("gSampler", mpTriLinearSampler);

    mpToneMapper = ToneMapping::create(ToneMapping::Operator::HableUc2);
    mpToneMapper->setExposureKey(0.104f);
    mLightIntensity = 2.5f;

    loadImage();
    
    initializeTesting();
}

void PostProcess::loadImage()
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

void PostProcess::onGuiRender()
{
    uint32_t uHdrIndex = static_cast<uint32_t>(mHdrImageIndex);
    if (mpGui->addDropdown("HdrImage", kImageList, uHdrIndex))
    {
        mHdrImageIndex = static_cast<HdrImage>(uHdrIndex);
        loadImage();
    }
    mpGui->addFloatVar("Surface Roughness", mSurfaceRoughness, 0.01f, 1000, 0.01f);
    mpGui->addFloatVar("Light Intensity", mLightIntensity, 0.5f, FLT_MAX, 0.1f);
    mpToneMapper->renderUI(mpGui.get(), "HDR");
}

void PostProcess::renderTeapot()
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
    mpRenderContext->setGraphicsVars(mpProgramVars);
    mpRenderContext->drawIndexed(mpTeapot->getMesh(0)->getIndexCount(), 0, 0);
}

void PostProcess::onFrameRender()
{
    beginTestFrame();

    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    mpRenderContext->clearFbo(mpHdrFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    mCameraController.update();

    //Render teapot to hdr fbo
    mpGraphicsState->pushFbo(mpHdrFbo);
    mpRenderContext->setGraphicsState(mpGraphicsState);
    renderTeapot();
    mpSkyBox->render(mpRenderContext.get(), mpCamera.get());
    mpGraphicsState->popFbo();

    //Run tone mapping
    mpToneMapper->execute(mpRenderContext.get(), mpHdrFbo, mpDefaultFBO);

    std::string Txt = getFpsMsg() + '\n';
    renderText(Txt, glm::vec2(10, 10));

    endTestFrame();
}

void PostProcess::onShutdown()
{
}

void PostProcess::onResizeSwapChain()
{
    //Camera aspect 
    float height = (float)mpDefaultFBO->getHeight();
    float width = (float)mpDefaultFBO->getWidth();
    mpCamera->setFocalLength(21.0f);
    float aspectRatio = (width / height);
    mpCamera->setAspectRatio(aspectRatio);

    //recreate hdr fbo
    ResourceFormat format = ResourceFormat::RGBA32Float;
    Fbo::Desc desc;
    desc.setDepthStencilTarget(ResourceFormat::D16Unorm);
    desc.setColorTarget(0u, format);
    mpHdrFbo = FboHelper::create2D(mpDefaultFBO->getWidth(), mpDefaultFBO->getHeight(), desc);
}

bool PostProcess::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return mCameraController.onKeyEvent(keyEvent);
}

bool PostProcess::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mCameraController.onMouseEvent(mouseEvent);
}

void PostProcess::onInitializeTesting()
{
    std::vector<ArgList::Arg> modeFrames = mArgList.getValues("changeMode");
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

void PostProcess::onEndTestFrame()
{
    uint32_t frameId = frameRate().getFrameCount();
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
    PostProcess postProcessSample;
    SampleConfig config;
    config.windowDesc.title = "Post Processing";
#ifdef _WIN32
    postProcessSample.run(config);
#else
    postProcessSample.run(config, (uint32_t)argc, argv);
#endif
    return 0;
}

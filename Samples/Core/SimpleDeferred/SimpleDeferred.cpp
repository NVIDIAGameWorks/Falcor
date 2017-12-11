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
#include "SimpleDeferred.h"

SimpleDeferred::~SimpleDeferred()
{
}

CameraController& SimpleDeferred::getActiveCameraController()
{
    switch(mCameraType)
    {
    case SimpleDeferred::ModelViewCamera:
        return mModelViewCameraController;
    case SimpleDeferred::FirstPersonCamera:
        return mFirstPersonCameraController;
    default:
        should_not_get_here();
        return mFirstPersonCameraController;
    }
}

void SimpleDeferred::loadModelFromFile(const std::string& filename)
{
    Model::LoadFlags flags = Model::LoadFlags::None;
    if (mGenerateTangentSpace == false)
    {
        flags |= Model::LoadFlags::DontGenerateTangentSpace;
    }
    auto fboFormat = mpDefaultFBO->getColorTexture(0)->getFormat();
    flags |= isSrgbFormat(fboFormat) ? Model::LoadFlags::None : Model::LoadFlags::AssumeLinearSpaceTextures;

    mpModel = Model::createFromFile(filename.c_str(), flags);

    if(mpModel == nullptr)
    {
        msgBox("Could not load model");
        return;
    }
    resetCamera();

    float Radius = mpModel->getRadius();
    mpPointLight->setWorldPosition(glm::vec3(0, Radius*1.25f, 0));
}

void SimpleDeferred::loadModel()
{
    std::string filename;
    if(openFileDialog("Supported Formats\0*.obj;*.bin;*.dae;*.x;*.md5mesh\0\0", filename))
    {
        loadModelFromFile(filename);
    }
}

void SimpleDeferred::onGuiRender()
{
    // Load model group
    if (mpGui->addButton("Load Model"))
    {
        loadModel();
    }

    if(mpGui->beginGroup("Load Options"))
    {
        mpGui->addCheckBox("Generate Tangent Space", mGenerateTangentSpace);
        mpGui->endGroup();
    }

    Gui::DropdownList debugModeList;
    debugModeList.push_back({ 0, "Disabled" });
    debugModeList.push_back({ 1, "Positions" });
    debugModeList.push_back({ 2, "Normals" });
    debugModeList.push_back({ 3, "Albedo" });
    debugModeList.push_back({ 4, "Illumination" });
    mpGui->addDropdown("Debug mode", debugModeList, (uint32_t&)mDebugMode);

    Gui::DropdownList cullList;
    cullList.push_back({0, "No Culling"});
    cullList.push_back({1, "Backface Culling"});
    cullList.push_back({2, "Frontface Culling"});
    mpGui->addDropdown("Cull Mode", cullList, (uint32_t&)mCullMode);

    if(mpGui->beginGroup("Lights"))
    {
        mpGui->addRgbColor("Ambient intensity", mAmbientIntensity);
        if(mpGui->beginGroup("Directional Light"))
        {
            mpDirLight->renderUI(mpGui.get());
            mpGui->endGroup();
        }
        if (mpGui->beginGroup("Point Light"))
        {
            mpPointLight->renderUI(mpGui.get());
            mpGui->endGroup();
        }
        mpGui->endGroup();
    }

    Gui::DropdownList cameraList;
    cameraList.push_back({ModelViewCamera, "Model-View"});
    cameraList.push_back({FirstPersonCamera, "First-Person"});
    mpGui->addDropdown("Camera Type", cameraList, (uint32_t&)mCameraType);

    if (mpModel)
    {
        renderModelUiElements();
    }
}

void SimpleDeferred::renderModelUiElements()
{
    bool bAnim = mpModel->hasAnimations();
    static const char* animateStr = "Animate";
    static const char* activeAnimStr = "Active Animation";

    if(bAnim)
    {
        mActiveAnimationID = sBindPoseAnimationID;

        mpGui->addCheckBox(animateStr, mAnimate);
        Gui::DropdownList list;
        list.resize(mpModel->getAnimationsCount() + 1);
        list[0].label = "Bind Pose";
        list[0].value = sBindPoseAnimationID;

        for(uint32_t i = 0; i < mpModel->getAnimationsCount(); i++)
        {
            list[i + 1].value = i;
            list[i + 1].label = mpModel->getAnimationName(i);
            if(list[i + 1].label.size() == 0)
            {
                list[i + 1].label = std::to_string(i);
            }
        }
        if (mpGui->addDropdown(activeAnimStr, list, mActiveAnimationID))
        {
            mpModel->setActiveAnimation(mActiveAnimationID);
        }
    }
    if(mpGui->beginGroup("Depth Range"))
    {
        const float minDepth = mpModel->getRadius() * 1 / 1000;
        mpGui->addFloatVar("Near Plane", mNearZ, minDepth, mpModel->getRadius() * 15, minDepth * 5);
        mpGui->addFloatVar("Far Plane", mFarZ, minDepth, mpModel->getRadius() * 15, minDepth * 5);
        mpGui->endGroup();
    }
}

void SimpleDeferred::onLoad()
{
    mpCamera = Camera::create();

	mpDeferredPassProgram = GraphicsProgram::createFromFile("", appendShaderExtension("DeferredPass.ps"));

    mpLightingPass = FullScreenPass::create(appendShaderExtension("LightingPass.ps"));

    // create rasterizer state
    RasterizerState::Desc rsDesc;
    mpCullRastState[0] = RasterizerState::create(rsDesc);
    rsDesc.setCullMode(RasterizerState::CullMode::Back);
    mpCullRastState[1] = RasterizerState::create(rsDesc);
    rsDesc.setCullMode(RasterizerState::CullMode::Front);
    mpCullRastState[2] = RasterizerState::create(rsDesc);

    // Depth test
    DepthStencilState::Desc dsDesc;
	dsDesc.setDepthTest(false);
	mpNoDepthDS = DepthStencilState::create(dsDesc);
    dsDesc.setDepthTest(true);
    mpDepthTestDS = DepthStencilState::create(dsDesc);

    // Blend state
    BlendState::Desc blendDesc;
    mpOpaqueBS = BlendState::create(blendDesc);

    mModelViewCameraController.attachCamera(mpCamera);
    mFirstPersonCameraController.attachCamera(mpCamera);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear).setMaxAnisotropy(8);
    mpLinearSampler = Sampler::create(samplerDesc);

    mpPointLight = PointLight::create();
    mpDirLight = DirectionalLight::create();
    mpDirLight->setWorldDirection(glm::vec3(-0.5f, -0.2f, -1.0f));

    mpDeferredVars = GraphicsVars::create(mpDeferredPassProgram->getActiveVersion()->getReflector());
    mpLightingVars = GraphicsVars::create(mpLightingPass->getProgram()->getActiveVersion()->getReflector());

    // Load default model
    loadModelFromFile("ogre/bs_rest.obj");

    initializeTesting();
}

void SimpleDeferred::onFrameRender()
{
    beginTestFrame();

    GraphicsState* pState = mpRenderContext->getGraphicsState().get();

    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);

    // G-Buffer pass
    if(mpModel)
    {
        mpRenderContext->clearFbo(mpGBufferFbo.get(), glm::vec4(0), 1.0f, 0, FboAttachmentType::Color | FboAttachmentType::Depth);
        pState->setFbo(mpGBufferFbo);

        mpCamera->setDepthRange(mNearZ, mFarZ);
        CameraController& ActiveController = getActiveCameraController();
        ActiveController.update();

        // Animate
        if(mAnimate)
        {
            PROFILE(Animate);
            mpModel->animate(mCurrentTime);
        }

        // Set render state
        pState->setRasterizerState(mpCullRastState[mCullMode]);
        pState->setDepthStencilState(mpDepthTestDS);

        // Render model
        mpModel->bindSamplerToMaterials(mpLinearSampler);
        mpRenderContext->setGraphicsVars(mpDeferredVars);
        pState->setProgram(mpDeferredPassProgram);
        ModelRenderer::render(mpRenderContext.get(), mpModel, mpCamera.get());
    }

    // Lighting pass (fullscreen quad)
    {
        pState->setFbo(mpDefaultFBO);
        mpRenderContext->clearFbo(mpDefaultFBO.get(), clearColor, 1.0f, 0, FboAttachmentType::Color);

        // Reset render state
        pState->setRasterizerState(mpCullRastState[0]);
        pState->setBlendState(mpOpaqueBS);
        pState->setDepthStencilState(mpNoDepthDS);

        // Set lighting params
        ConstantBuffer::SharedPtr pLightCB = mpLightingVars["PerImageCB"];
        pLightCB["gAmbient"] = mAmbientIntensity;
        mpDirLight->setIntoConstantBuffer(pLightCB.get(), "gDirLight");
        mpPointLight->setIntoConstantBuffer(pLightCB.get(), "gPointLight");
        // Debug mode
        pLightCB->setVariable("gDebugMode", (uint32_t)mDebugMode);

        // Set GBuffer as input
        mpLightingVars->setTexture("gGBuf0", mpGBufferFbo->getColorTexture(0));
        mpLightingVars->setTexture("gGBuf1", mpGBufferFbo->getColorTexture(1));
        mpLightingVars->setTexture("gGBuf2", mpGBufferFbo->getColorTexture(2));


        // Kick it off
        mpRenderContext->setGraphicsVars(mpLightingVars);
        mpLightingPass->execute(mpRenderContext.get());
    }

    endTestFrame();
}

void SimpleDeferred::onShutdown()
{
    mpModel.reset();
}

bool SimpleDeferred::onKeyEvent(const KeyboardEvent& keyEvent)
{
    bool bHandled = getActiveCameraController().onKeyEvent(keyEvent);
    if(bHandled == false)
    {
        if(keyEvent.type == KeyboardEvent::Type::KeyPressed)
        {
            switch(keyEvent.key)
            {
            case KeyboardEvent::Key::R:
                resetCamera();
                break;
            default:
                bHandled = false;
            }
        }
    }
    return bHandled;
}

bool SimpleDeferred::onMouseEvent(const MouseEvent& mouseEvent)
{
    return getActiveCameraController().onMouseEvent(mouseEvent);
}

void SimpleDeferred::onResizeSwapChain()
{
    uint32_t width = mpDefaultFBO->getWidth();
    uint32_t height = mpDefaultFBO->getHeight();

    mpCamera->setFocalLength(21.0f);
    mAspectRatio = (float(width) / float(height));
    mpCamera->setAspectRatio(mAspectRatio);
    // create G-Buffer
    const glm::vec4 clearColor(0.f, 0.f, 0.f, 0.f);
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, Falcor::ResourceFormat::RGBA16Float).setColorTarget(1, Falcor::ResourceFormat::RGBA16Float).setColorTarget(2, Falcor::ResourceFormat::RGBA16Float).setDepthStencilTarget(Falcor::ResourceFormat::D32Float);
    mpGBufferFbo = FboHelper::create2D(width, height, fboDesc);
}

void SimpleDeferred::resetCamera()
{
    if(mpModel)
    {
        // update the camera position
        float radius = mpModel->getRadius();
        const glm::vec3& modelCenter = mpModel->getCenter();
        glm::vec3 camPos = modelCenter;
        camPos.z += radius * 4;

        mpCamera->setPosition(camPos);
        mpCamera->setTarget(modelCenter);
        mpCamera->setUpVector(glm::vec3(0, 1, 0));

        // Update the controllers
        mModelViewCameraController.setModelParams(modelCenter, radius, 4);
        mFirstPersonCameraController.setCameraSpeed(radius*0.25f);
        mNearZ = std::max(0.1f, mpModel->getRadius() / 750.0f);
        mFarZ = radius * 10;
    }
}

void SimpleDeferred::onInitializeTesting()
{
    std::vector<ArgList::Arg> modeFrames = mArgList.getValues("incrementDebugMode");
    if (!modeFrames.empty())
    {
        mChangeModeFrames.resize(modeFrames.size());
        for (uint32_t i = 0; i < modeFrames.size(); ++i)
        {
            mChangeModeFrames[i] = modeFrames[i].asUint();
        }
    }

    mChangeModeIt = mChangeModeFrames.begin();
}

void SimpleDeferred::onEndTestFrame()
{
    uint32_t frameId = frameRate().getFrameCount();
    if (mChangeModeIt != mChangeModeFrames.end() && frameId >= *mChangeModeIt)
    {
        ++mChangeModeIt;
        uint32_t* pMode = (uint32_t*)&mDebugMode;
        *pMode = min(*pMode + 1, (uint32_t)ShowLighting);
    }
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    SimpleDeferred sample;
    SampleConfig config;
    config.windowDesc.width = 1280;
    config.windowDesc.height = 720;
    config.windowDesc.resizableWindow = true;
    config.windowDesc.title = "Simple Deferred";
#ifdef _WIN32
    sample.run(config);
#else
    sample.run(config, (uint32_t)argc, argv);
#endif
    return 0;
}

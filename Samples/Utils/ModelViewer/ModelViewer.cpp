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
#include "ModelViewer.h"

const std::string ModelViewer::skDefaultModel = "Arcade/Arcade.fbx";

void ModelViewer::deleteCulledMeshes()
{
    if(mpModel)
    {
        CpuTimer timer;
        timer.update();
        mpModel->deleteCulledMeshes(mpCamera.get());
        timer.update();

        setModelString(true, timer.getElapsedTime());
    }
}

CameraController& ModelViewer::getActiveCameraController()
{
    switch(mCameraType)
    {
    case ModelViewer::ModelViewCamera:
        return mModelViewCameraController;
    case ModelViewer::FirstPersonCamera:
        return mFirstPersonCameraController;
    case ModelViewer::SixDoFCamera:
        return m6DoFCameraController;
    default:
        should_not_get_here();
        return m6DoFCameraController;
    }
}

void ModelViewer::setModelString(bool isAfterCull, float loadTime)
{
    mModelString = isAfterCull ? "Mesh culling" : "Loading";
    mModelString += " took " + std::to_string(loadTime) + " seconds.\n";
    mModelString += "Model has " + std::to_string(mpModel->getVertexCount()) + " vertices, ";
    mModelString += std::to_string(mpModel->getIndexCount()) + " indices, ";
    mModelString += std::to_string(mpModel->getPrimitiveCount()) + " primitives, ";
    mModelString += std::to_string(mpModel->getMeshCount()) + " meshes, ";
    mModelString += std::to_string(mpModel->getInstanceCount()) + " mesh instances, ";
    mModelString += std::to_string(mpModel->getMaterialCount()) + " materials, ";
    mModelString += std::to_string(mpModel->getTextureCount()) + " textures, ";
    mModelString += std::to_string(mpModel->getBufferCount()) + " buffers.\n";
}

void ModelViewer::loadModelFromFile(const std::string& filename, ResourceFormat fboFormat)
{
    CpuTimer timer;
    timer.update();

    Model::LoadFlags flags = Model::LoadFlags::None;
    if (mGenerateTangentSpace == false)
    {
        flags |= Model::LoadFlags::DontGenerateTangentSpace;
    }

    flags |= isSrgbFormat(fboFormat) ? Model::LoadFlags::None : Model::LoadFlags::AssumeLinearSpaceTextures;
    mpModel = Model::createFromFile(filename.c_str(), flags);

    if(mpModel == nullptr)
    {
        msgBox("Could not load model");
        return;
    }
    resetCamera();

    float radius = mpModel->getRadius();
    float lightHeight = max(1.0f + radius, radius*1.25f);
    mpPointLight->setWorldPosition(glm::vec3(0, lightHeight, 0));
    timer.update();

    mActiveAnimationID = kBindPoseAnimationID;
    setModelString(false, timer.getElapsedTime());

    mpModel->bindSamplerToMaterials(mUseTriLinearFiltering ? mpLinearSampler : mpPointSampler);
}

void ModelViewer::loadModel(ResourceFormat fboFormat)
{
    std::string Filename;
    if(openFileDialog(Model::kSupportedFileFormatsStr, Filename))
    {
        loadModelFromFile(Filename, fboFormat);
    }
}

void ModelViewer::saveModel()
{
    if(mpModel == nullptr)
    {
        msgBox("No model was loaded. Nothing to save");
        return;

    }
    std::string filename;
    if(saveFileDialog("Binary Model\0*.bin\0\0", filename))
    {
        mpModel->exportToBinaryFile(filename);
    }
}

void ModelViewer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    // Load model group
    if (pGui->addButton("Load Model"))
    {
        loadModel(pSample->getCurrentFbo()->getColorTexture(0)->getFormat());
    }
    if (pGui->beginGroup("Load Options"))
    {
        pGui->addCheckBox("Generate Tangent Space", mGenerateTangentSpace);
        if (pGui->addButton("Export Model To Binary File"))
        {
            saveModel();
        }
        if (pGui->addButton("Delete Culled Meshes"))
        {
            deleteCulledMeshes();
        }
        pGui->endGroup();
    }

    pGui->addSeparator();
    pGui->addCheckBox("Wireframe", mDrawWireframe);
    if (pGui->addCheckBox("TriLinear Filtering", mUseTriLinearFiltering))
    {
        mpModel->bindSamplerToMaterials(mUseTriLinearFiltering ? mpLinearSampler : mpPointSampler);
    }

    Gui::DropdownList cullList;
    cullList.push_back({ 0, "No Culling" });
    cullList.push_back({ 1, "Backface Culling" });
    cullList.push_back({ 2, "Frontface Culling" });
    pGui->addDropdown("Cull Mode", cullList, mCullMode);

    if (pGui->beginGroup("Lights"))
    {
        pGui->addRgbColor("Ambient intensity", mAmbientIntensity);
        if (pGui->beginGroup("Directional Light"))
        {
            mpDirLight->renderUI(pGui);
            pGui->endGroup();
        }
        if (pGui->beginGroup("Point Light"))
        {
            mpPointLight->renderUI(pGui);
            pGui->endGroup();
        }
        pGui->endGroup();
    }

    Gui::DropdownList cameraDropdown;
    cameraDropdown.push_back({ ModelViewCamera, "Model-View" });
    cameraDropdown.push_back({ FirstPersonCamera, "First-Person" });
    cameraDropdown.push_back({ SixDoFCamera, "6 DoF" });

    pGui->addDropdown("Camera Type", cameraDropdown, (uint32_t&)mCameraType);

    if(mpModel)
    {
        renderModelUI(pGui);
    }
}

void ModelViewer::renderModelUI(Gui* pGui)
{
    bool bAnim = mpModel && mpModel->hasAnimations();
    static const char* animateStr = "Animate";
    static const char* activeAnimStr = "Active Animation";

    if(bAnim)
    {
        pGui->addCheckBox(animateStr, mAnimate);
        Gui::DropdownList list;
        list.resize(mpModel->getAnimationsCount() + 1);
        list[0].label = "Bind Pose";
        list[0].value = kBindPoseAnimationID;

        for(uint32_t i = 0; i < mpModel->getAnimationsCount(); i++)
        {
            list[i + 1].value = i;
            list[i + 1].label = mpModel->getAnimationName(i);
            if(list[i + 1].label.size() == 0)
            {
                list[i + 1].label = std::to_string(i);
            }
        }

        if (pGui->addDropdown(activeAnimStr, list, mActiveAnimationID))
        {
            mpModel->setActiveAnimation(mActiveAnimationID);
        }
    }

    const float minDepth = mpModel->getRadius() * 1 / 1000;
    if(pGui->beginGroup("Depth Range"))
    {
        pGui->addFloatVar("Near Plane", mNearZ, minDepth, mpModel->getRadius() * 15, minDepth * 5);
        pGui->addFloatVar("Far Plane", mFarZ, minDepth, mpModel->getRadius() * 15, minDepth * 5);
        pGui->endGroup();
    }
}

void ModelViewer::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mpCamera = Camera::create();
    mpProgram = GraphicsProgram::createFromFile("ModelViewer.ps.hlsl", "", "main");

    // create rasterizer state
    RasterizerState::Desc wireframeDesc;
    wireframeDesc.setFillMode(RasterizerState::FillMode::Wireframe);
    wireframeDesc.setCullMode(RasterizerState::CullMode::None);
    mpWireframeRS = RasterizerState::create(wireframeDesc);

    RasterizerState::Desc solidDesc;
    solidDesc.setCullMode(RasterizerState::CullMode::None);
    mpCullRastState[0] = RasterizerState::create(solidDesc);
    solidDesc.setCullMode(RasterizerState::CullMode::Back);
    mpCullRastState[1] = RasterizerState::create(solidDesc);
    solidDesc.setCullMode(RasterizerState::CullMode::Front);
    mpCullRastState[2] = RasterizerState::create(solidDesc);

    // Depth test
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthTest(false);
    mpNoDepthDS = DepthStencilState::create(dsDesc);
    dsDesc.setDepthTest(true);
    mpDepthTestDS = DepthStencilState::create(dsDesc);

    mModelViewCameraController.attachCamera(mpCamera);
    mFirstPersonCameraController.attachCamera(mpCamera);
    m6DoFCameraController.attachCamera(mpCamera);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
    mpPointSampler = Sampler::create(samplerDesc);
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpLinearSampler = Sampler::create(samplerDesc);

    mpDirLight = DirectionalLight::create();
    mpPointLight = PointLight::create();
    mpDirLight->setWorldDirection(glm::vec3(0.13f, 0.27f, -0.9f));

    mpProgramVars = GraphicsVars::create(mpProgram->getReflector());
    mpGraphicsState = GraphicsState::create();
    mpGraphicsState->setProgram(mpProgram);

    loadModelFromFile(skDefaultModel, pRenderContext->getGraphicsState()->getFbo()->getColorTexture(0)->getFormat());
}

void ModelViewer::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    mpGraphicsState->setFbo(pTargetFbo);

    if(mpModel)
    {
        mpCamera->setDepthRange(mNearZ, mFarZ);
        getActiveCameraController().update();

        // Animate
        if(mAnimate)
        {
            PROFILE(animate);
            mpModel->animate(pSample->getCurrentTime());
        }

        // Set render state
        if(mDrawWireframe)
        {
            mpGraphicsState->setRasterizerState(mpWireframeRS);
            mpGraphicsState->setDepthStencilState(mpNoDepthDS);
            mpProgramVars["PerFrameCB"]["gConstColor"] = true;
        }
        else
        {
            mpGraphicsState->setRasterizerState(mpCullRastState[mCullMode]);
            mpGraphicsState->setDepthStencilState(mpDepthTestDS);
            mpProgramVars["PerFrameCB"]["gConstColor"] = false;

            ConstantBuffer* pCB = mpProgramVars["PerFrameCB"].get();
            mpDirLight->setIntoProgramVars(mpProgramVars.get(), pCB, "gDirLight");
            mpPointLight->setIntoProgramVars(mpProgramVars.get(), pCB, "gPointLight");
        }

        mpProgramVars["PerFrameCB"]["gAmbient"] = mAmbientIntensity;
        mpGraphicsState->setProgram(mpProgram);
        pRenderContext->setGraphicsState(mpGraphicsState);
        pRenderContext->setGraphicsVars(mpProgramVars);
        ModelRenderer::render(pRenderContext.get(), mpModel, mpCamera.get());
    }

    pSample->renderText(mModelString, glm::vec2(10, 30));
}

bool ModelViewer::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
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
                bHandled = true;
                break;
            }
        }
    }
    return bHandled;
}

bool ModelViewer::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return getActiveCameraController().onMouseEvent(mouseEvent);
}

void ModelViewer::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    float h = (float)height;
    float w = (float)width;

    mpCamera->setFocalLength(21.0f);
    float aspectRatio = (w / h);
    mpCamera->setAspectRatio(aspectRatio);
}

void ModelViewer::resetCamera()
{
    if(mpModel)
    {
        // update the camera position
        float Radius = mpModel->getRadius();
        const glm::vec3& ModelCenter = mpModel->getCenter();
        glm::vec3 CamPos = ModelCenter;
        CamPos.z += Radius * 5;

        mpCamera->setPosition(CamPos);
        mpCamera->setTarget(ModelCenter);
        mpCamera->setUpVector(glm::vec3(0, 1, 0));

        // Update the controllers
        mModelViewCameraController.setModelParams(ModelCenter, Radius, 3.5f);
        mFirstPersonCameraController.setCameraSpeed(Radius*0.25f);
        m6DoFCameraController.setCameraSpeed(Radius*0.25f);

        mNearZ = std::max(0.1f, mpModel->getRadius() / 750.0f);
        mFarZ = Radius * 10;
    }
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    ModelViewer::UniquePtr pRenderer = std::make_unique<ModelViewer>();

    SampleConfig config;
    config.windowDesc.title = "Falcor Model Viewer";
    config.windowDesc.resizableWindow = true;
#ifdef _WIN32
    Sample::run(config, pRenderer);
#else
    config.argc = (uint32_t)argc;
    config.argv = argv;
    Sample::run(config, pRenderer);
#endif
    return 0;
}
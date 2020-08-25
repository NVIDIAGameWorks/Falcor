/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "ModelViewer.h"

void ModelViewer::setModelString(double loadTime)
{
    assert(mpScene != nullptr);

    mModelString = "Loading took " + std::to_string(loadTime) + " seconds.\n";
    //mModelString += "Model has " + std::to_string(pModel->getVertexCount()) + " vertices, ";
    //mModelString += std::to_string(pModel->getIndexCount()) + " indices, ";
    //mModelString += std::to_string(pModel->getPrimitiveCount()) + " primitives, ";
    mModelString += std::to_string(mpScene->getMeshCount()) + " meshes, ";
    mModelString += std::to_string(mpScene->getMeshInstanceCount()) + " mesh instances, ";
    mModelString += std::to_string(mpScene->getMaterialCount()) + " materials, ";
    //mModelString += std::to_string(pModel->getTextureCount()) + " textures, ";
    //mModelString += std::to_string(pModel->getBufferCount()) + " buffers.\n";
}

void ModelViewer::loadModelFromFile(const std::string& filename, ResourceFormat fboFormat)
{
    CpuTimer timer;
    timer.update();

    SceneBuilder::Flags flags = SceneBuilder::Flags::None;
    if (mUseOriginalTangents) flags |= SceneBuilder::Flags::UseOriginalTangentSpace;
    if (mRemoveDuplicateMaterials) flags |= SceneBuilder::Flags::RemoveDuplicateMaterials;
    flags |= isSrgbFormat(fboFormat) ? SceneBuilder::Flags::None : SceneBuilder::Flags::AssumeLinearSpaceTextures;

    SceneBuilder::SharedPtr pBuilder = SceneBuilder::create(filename, flags);

    if (!pBuilder)
    {
        msgBox("Could not load model");
        return;
    }

    mpScene = pBuilder->getScene();
    mpProgram->addDefines(mpScene->getSceneDefines());
    mpProgramVars = GraphicsVars::create(mpProgram->getReflector());
    mpScene->bindSamplerToMaterials(mUseTriLinearFiltering ? mpLinearSampler : mpPointSampler);
    setCamController();

    timer.update();
    setModelString(timer.delta());
}

void ModelViewer::loadModel(ResourceFormat fboFormat)
{
    std::string Filename;
    if(openFileDialog(Scene::getFileExtensionFilters(), Filename))
    {
        loadModelFromFile(Filename, fboFormat);
    }
}

void ModelViewer::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Model Viewer", { 400, 300 }, { 0, 100 });

    // Load model group
    if (w.button("Load Model"))
    {
        loadModel(gpFramework->getTargetFbo()->getColorTexture(0)->getFormat());
    }

    {
        auto loadGroup = w.group("Load Options");
        loadGroup.checkbox("Use Original Tangents", mUseOriginalTangents);
        loadGroup.tooltip("If this is unchecked, we will ignore the tangents that were loaded from the model and calculate them internally. Check this box if you'd like to use the original tangents");
        loadGroup.checkbox("Remove Duplicate Materials", mRemoveDuplicateMaterials);
        loadGroup.tooltip("Deduplicate materials that have the same properties. The material name is ignored during the search");
    }

    w.separator();
    w.checkbox("Wireframe", mDrawWireframe);

    if(mDrawWireframe == false)
    {
        w.checkbox("Override Rasterizer State", mOverrideRS);

        if(mOverrideRS)
        {
            Gui::DropdownList cullList;
            cullList.push_back({ 0, "No Culling" });
            cullList.push_back({ 1, "Backface Culling" });
            cullList.push_back({ 2, "Frontface Culling" });
            w.dropdown("Cull Mode", cullList, mCullMode);
        }
    }

    Gui::DropdownList cameraDropdown;
    cameraDropdown.push_back({ (uint32_t)Scene::CameraControllerType::FirstPerson, "First-Person" });
    cameraDropdown.push_back({ (uint32_t)Scene::CameraControllerType::Orbiter, "Orbiter" });
    cameraDropdown.push_back({ (uint32_t)Scene::CameraControllerType::SixDOF, "6-DoF" });

    if (w.dropdown("Camera Type", cameraDropdown, (uint32_t&)mCameraType)) setCamController();
    if (mpScene) mpScene->renderUI(w);
}

void ModelViewer::onLoad(RenderContext* pRenderContext)
{
    mpProgram = GraphicsProgram::createFromFile("Samples/ModelViewer/ModelViewer.ps.slang", "", "main");
    mpGraphicsState = GraphicsState::create();
    mpGraphicsState->setProgram(mpProgram);

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
    dsDesc.setDepthEnabled(false);
    mpNoDepthDS = DepthStencilState::create(dsDesc);
    dsDesc.setDepthFunc(ComparisonFunc::Less).setDepthEnabled(true);
    mpDepthTestDS = DepthStencilState::create(dsDesc);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
    mpPointSampler = Sampler::create(samplerDesc);
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpLinearSampler = Sampler::create(samplerDesc);

    resetCamera();
}

void ModelViewer::onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const float4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    mpGraphicsState->setFbo(pTargetFbo);

    if(mpScene)
    {
        mpScene->update(pRenderContext, gpFramework->getGlobalClock().getTime());

        // Set render state
        Scene::RenderFlags renderFlags = Scene::RenderFlags::None;
        if(mDrawWireframe)
        {
            renderFlags |= Scene::RenderFlags::UserRasterizerState;
            mpGraphicsState->setRasterizerState(mpWireframeRS);
            mpGraphicsState->setDepthStencilState(mpNoDepthDS);
            mpProgramVars["PerFrameCB"]["gConstColor"] = true;
        }
        else
        {
            mpProgramVars["PerFrameCB"]["gConstColor"] = false;
            mpGraphicsState->setDepthStencilState(mpDepthTestDS);
            if (mOverrideRS)
            {
                renderFlags |= Scene::RenderFlags::UserRasterizerState;
                mpGraphicsState->setRasterizerState(mpCullRastState[mCullMode]);
            }
        }

        mpGraphicsState->setProgram(mpProgram);
        mpScene->render(pRenderContext, mpGraphicsState.get(), mpProgramVars.get(), renderFlags);
    }

    TextRenderer::render(pRenderContext, mModelString, pTargetFbo, float2(10, 30));
}

bool ModelViewer::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (mpScene && mpScene->onKeyEvent(keyEvent)) return true;

    if ((keyEvent.type == KeyboardEvent::Type::KeyPressed) && (keyEvent.key == KeyboardEvent::Key::R))
    {
        resetCamera();
        return true;
    }
    return false;
}

bool ModelViewer::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpScene ? mpScene->onMouseEvent(mouseEvent) : false;
}

void ModelViewer::onResizeSwapChain(uint32_t width, uint32_t height)
{
    float h = (float)height;
    float w = (float)width;
}

void ModelViewer::setCamController()
{
    if(mpScene) mpScene->setCameraController(mCameraType);
}

void ModelViewer::resetCamera()
{
    if(mpScene)
    {
        mpScene->resetCamera(true);
        setCamController();
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

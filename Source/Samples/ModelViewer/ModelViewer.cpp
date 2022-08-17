/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/UI/TextRenderer.h"

void ModelViewer::setModelString(double loadTime)
{
    FALCOR_ASSERT(mpScene != nullptr);

    mModelString = "Loading took " + std::to_string(loadTime) + " seconds.\n";
    //mModelString += "Model has " + std::to_string(pModel->getVertexCount()) + " vertices, ";
    //mModelString += std::to_string(pModel->getIndexCount()) + " indices, ";
    //mModelString += std::to_string(pModel->getPrimitiveCount()) + " primitives, ";
    mModelString += std::to_string(mpScene->getMeshCount()) + " meshes, ";
    mModelString += std::to_string(mpScene->getGeometryInstanceCount()) + " instances, ";
    mModelString += std::to_string(mpScene->getMaterialCount()) + " materials, ";
    //mModelString += std::to_string(pModel->getTextureCount()) + " textures, ";
    //mModelString += std::to_string(pModel->getBufferCount()) + " buffers.\n";
}

void ModelViewer::loadModelFromFile(const std::filesystem::path& path, ResourceFormat fboFormat)
{
    CpuTimer timer;
    timer.update();

    SceneBuilder::Flags flags = SceneBuilder::Flags::None;
    if (mUseOriginalTangents) flags |= SceneBuilder::Flags::UseOriginalTangentSpace;
    if (mDontMergeMaterials) flags |= SceneBuilder::Flags::DontMergeMaterials;
    flags |= isSrgbFormat(fboFormat) ? SceneBuilder::Flags::None : SceneBuilder::Flags::AssumeLinearSpaceTextures;

    try
    {
        mpScene = SceneBuilder::create(path, flags)->getScene();
    }
    catch (const std::exception& e)
    {
        msgBox(fmt::format("Failed to load model.\n\n{}", e.what()));
        return;
    }

    Program::Desc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary("Samples/ModelViewer/ModelViewer.3d.slang").vsEntry("vsMain").psEntry("psMain");
    desc.addTypeConformances(mpScene->getTypeConformances());

    mpProgram = GraphicsProgram::create(desc, mpScene->getSceneDefines());
    mpProgramVars = GraphicsVars::create(mpProgram->getReflector());
    mpGraphicsState->setProgram(mpProgram);

    mpScene->getMaterialSystem()->setDefaultTextureSampler(mUseTriLinearFiltering ? mpLinearSampler : mpPointSampler);
    setCamController();

    timer.update();
    setModelString(timer.delta());
}

void ModelViewer::loadModel(ResourceFormat fboFormat)
{
    std::filesystem::path path;
    if (openFileDialog(Scene::getFileExtensionFilters(), path))
    {
        loadModelFromFile(path, fboFormat);
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
        loadGroup.checkbox("Don't Merge Materials", mDontMergeMaterials);
        loadGroup.tooltip("Don't merge materials that have the same properties. Use this option to preserve the original material names.");
    }

    w.separator();
    w.checkbox("Wireframe", mDrawWireframe);

    if (mDrawWireframe == false)
    {
        Gui::DropdownList cullList;
        cullList.push_back({ (uint32_t)RasterizerState::CullMode::None, "No Culling" });
        cullList.push_back({ (uint32_t)RasterizerState::CullMode::Back, "Backface Culling" });
        cullList.push_back({ (uint32_t)RasterizerState::CullMode::Front, "Frontface Culling" });
        w.dropdown("Cull Mode", cullList, (uint32_t&)mCullMode);
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
    mpGraphicsState = GraphicsState::create();

    // Create rasterizer state
    RasterizerState::Desc wireframeDesc;
    wireframeDesc.setFillMode(RasterizerState::FillMode::Wireframe);
    wireframeDesc.setCullMode(RasterizerState::CullMode::None);
    mpWireframeRS = RasterizerState::create(wireframeDesc);

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

    if (mpScene)
    {
        mpScene->update(pRenderContext, gpFramework->getGlobalClock().getTime());

        // Set render state
        if (mDrawWireframe)
        {
            mpGraphicsState->setDepthStencilState(mpNoDepthDS);
            mpProgramVars["PerFrameCB"]["gConstColor"] = true;

            mpScene->rasterize(pRenderContext, mpGraphicsState.get(), mpProgramVars.get(), mpWireframeRS, mpWireframeRS);
        }
        else
        {
            mpGraphicsState->setDepthStencilState(mpDepthTestDS);
            mpProgramVars["PerFrameCB"]["gConstColor"] = false;

            mpScene->rasterize(pRenderContext, mpGraphicsState.get(), mpProgramVars.get(), mCullMode);
        }
    }

    TextRenderer::render(pRenderContext, mModelString, pTargetFbo, float2(10, 30));
}

bool ModelViewer::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (mpScene && mpScene->onKeyEvent(keyEvent)) return true;

    if ((keyEvent.type == KeyboardEvent::Type::KeyPressed) && (keyEvent.key == Input::Key::R))
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

int main(int argc, char** argv)
{
    ModelViewer::UniquePtr pRenderer = std::make_unique<ModelViewer>();

    SampleConfig config;
    config.windowDesc.title = "Falcor Model Viewer";
    config.windowDesc.resizableWindow = true;

    Sample::run(config, pRenderer);
    return 0;
}

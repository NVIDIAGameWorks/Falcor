/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "LightProbeViewer.h"
#include "Data/HostDeviceSharedMacros.h"

const std::string LightProbeViewer::kEnvMapName = "LightProbes/10-Shiodome_Stairs_3k.dds";

Material::SharedPtr generateMaterial(const std::string& name, uint32_t roughI, uint32_t roughN, uint32_t metalI, uint32_t metalN)
{
    Material::SharedPtr pMat = Material::create(name);

    float roughness = float(roughI) / float(roughN - 1);
    float metallic = float(metalI) / float(metalN - 1);

    pMat->setShadingModel(ShadingModelMetalRough);
    pMat->setBaseColor(vec4(1));
    pMat->setSpecularParams(vec4(0, roughness, metallic, 0));

    return pMat;
}

void fillScene(Scene::SharedPtr pScene, const std::string& modelName, uint32_t width, uint32_t height)
{
    assert(pScene != nullptr);

    float offsetStep = 0.75f;
    const vec3 origin = vec3(-float(width) / 2.0f, float(height) / 2.0f, 0.0f) * offsetStep;

    vec3 posW = origin;
    for (uint32_t row = 0; row < height; row++)
    {
        posW.x = origin.x;
        for (uint32_t col = 0; col < width; col++)
        {
            std::string instName = std::to_string(col) + "_" + std::to_string(row);
            Model::SharedPtr pModel = Model::createFromFile(modelName.c_str());

            // Scale to about unit size
            float scaling = 0.5f / pModel->getRadius();

            Material::SharedPtr pMat = generateMaterial(instName, col, width, row, height);
            pModel->getMesh(0)->setMaterial(pMat);
            pScene->addModelInstance(pModel, instName, posW, vec3(), vec3(scaling));

            posW.x += offsetStep;
        }
        posW.y -= offsetStep;
    }
}

void LightProbeViewer::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mpCamera = Camera::create();
    mCameraController.attachCamera(mpCamera);

    mpState = GraphicsState::create();

    mpProgram = GraphicsProgram::createFromFile("LightProbeViewer.ps.hlsl", "", "main");
    mpState->setProgram(mpProgram);

    // States
    mpRasterizerState = RasterizerState::create(RasterizerState::Desc().setCullMode(RasterizerState::CullMode::Back));
    mpDepthState = DepthStencilState::create(DepthStencilState::Desc().setDepthTest(true));
    mpState->setRasterizerState(mpRasterizerState);
    mpState->setDepthStencilState(mpDepthState);

    mpVars = GraphicsVars::create(mpProgram->getActiveVersion()->getReflector());
    mpLinearSampler = Sampler::create(Sampler::Desc().setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear));

    mpScene = Scene::create();
    mpSceneRenderer = SceneRenderer::create(mpScene);

    fillScene(mpScene, "UnitSphere.fbx", 8, 3);
    resetCamera();

    pSample->setDefaultGuiSize(250, 250);

    updateLightProbe(LightProbe::create(pRenderContext.get(), kEnvMapName, true, ResourceFormat::RGBA16Float, mDiffuseSamples, mSpecSamples));
}

void LightProbeViewer::updateLightProbe(LightProbe::SharedPtr pLightProbe)
{
    mpLightProbe = pLightProbe;
    mpLightProbe->setSampler(mpLinearSampler);

    if (mpScene->getLightProbeCount() > 0)
    {
        assert(mpScene->getLightProbeCount() == 1);
        mpScene->deleteLightProbe(0);
    }

    mpScene->addLightProbe(mpLightProbe);

    if (mpSkyBox == nullptr || pLightProbe->getOrigTexture()->getSourceFilename() != mpSkyBox->getTexture()->getSourceFilename())
    {
        mpSkyBox = SkyBox::create(pLightProbe->getOrigTexture()->getSourceFilename());
    }
}

void LightProbeViewer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    pGui->addText("Press 'R' to reset scene camera");

    if (pGui->addButton("Load Light Probe"))
    {
        std::string filename;
        if (openFileDialog("Image files\0*.hdr;*.exr\0\0", filename))
        {
            updateLightProbe(LightProbe::create(pSample->getRenderContext().get(), filename, true, ResourceFormat::RGBA16Float, mDiffuseSamples, mSpecSamples));
        }
    }
    if (mpLightProbe != nullptr)
    {
        std::string diffText = "Diffuse Sample Count: " + std::to_string(mpLightProbe->getDiffSampleCount());
        pGui->addText(diffText.c_str());
        int32_t diffSamples = int32_t(mDiffuseSamples);
        if (pGui->addIntVar("Diffuse##Samples", diffSamples, 1, 128 * 1024))
        {
            mDiffuseSamples = uint32_t(diffSamples);
        }

        std::string specText = "Spec Sample Count: " + std::to_string(mpLightProbe->getSpecSampleCount());
        pGui->addText(specText.c_str());
        int32_t specSamples = int32_t(mSpecSamples);
        if (pGui->addIntVar("Specular##Samples", specSamples, 1, 32 * 1024))
        {
            mSpecSamples = uint32_t(specSamples);
        }

        if (pGui->addButton("Apply"))
        {
            if (mDiffuseSamples != mpLightProbe->getDiffSampleCount() || mSpecSamples != mpLightProbe->getSpecSampleCount())
            {
                updateLightProbe(LightProbe::create(pSample->getRenderContext().get(), mpLightProbe->getOrigTexture(), mDiffuseSamples, mSpecSamples));
            }
        }

        pGui->addText("Specular Viewport Mip Level");
        pGui->addIntVar("##SpecMip", mSpecMip, 0, mpLightProbe->getSpecularTexture()->getMipCount() - 1);
    }
}

void LightProbeViewer::onDataReload(SampleCallbacks* pSample)
{
}

void LightProbeViewer::onDroppedFile(SampleCallbacks* pSample, const std::string& filename)
{
    updateLightProbe(LightProbe::create(pSample->getRenderContext().get(), filename, true, ResourceFormat::RGBA16Float, mDiffuseSamples, mSpecSamples));
}

void LightProbeViewer::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if (mpLightProbe != nullptr)
    {
        mCameraController.update();
        mpState->setFbo(pTargetFbo);

        // Where to render the scene to
        uvec4 destRect = (mSelectedView == Viewport::Scene) ? mMainRect : mRects[(uint32_t)Viewport::Scene];
        float w = (float)destRect.z - (float)destRect.x;
        float h = (float)destRect.w - (float)destRect.y;
        mpState->setViewport(0, GraphicsState::Viewport((float)destRect.x, (float)destRect.y, w, h, 0.0f, 1.0f));

        pRenderContext->pushGraphicsVars(mpVars);
        pRenderContext->pushGraphicsState(mpState);

        mpSceneRenderer->renderScene(pRenderContext.get(), mpCamera.get());
        mpSkyBox->render(pRenderContext.get(), mpCamera.get());

        pRenderContext->popGraphicsVars();
        pRenderContext->popGraphicsState();

        pRenderContext->blit(mpLightProbe->getOrigTexture()->getSRV(0, 1), pTargetFbo->getRenderTargetView(0), uvec4(-1), (mSelectedView == Viewport::Orig) ? mMainRect : mRects[(uint32_t)Viewport::Orig]);
        pRenderContext->blit(mpLightProbe->getDiffuseTexture()->getSRV(0, 1), pTargetFbo->getRenderTargetView(0), uvec4(-1), (mSelectedView == Viewport::Diffuse) ? mMainRect : mRects[(uint32_t)Viewport::Diffuse]);
        pRenderContext->blit(mpLightProbe->getSpecularTexture()->getSRV(mSpecMip, 1), pTargetFbo->getRenderTargetView(0), uvec4(-1), (mSelectedView == Viewport::Specular) ? mMainRect : mRects[(uint32_t)Viewport::Specular], Sampler::Filter::Point);

        renderInfoText(pSample);
    }
}

void LightProbeViewer::renderInfoText(SampleCallbacks* pSample)
{
    pSample->renderText("Click a viewport to expand", vec2(mMainRect.z + 5, 5));

#define bottom_left(viewport) vec2(mRects[(uint32_t)viewport].x, mRects[(uint32_t)viewport].w) + vec2(5.0f, -20.0f) // bottom left plus offset
#define dim_to_string(texture) std::string(std::to_string(texture->getWidth()) + " x " + std::to_string(texture->getHeight()))

    pSample->renderText("Scene",                                                           bottom_left(Viewport::Scene));
    pSample->renderText("Original - " + dim_to_string(mpLightProbe->getOrigTexture()),     bottom_left(Viewport::Orig));
    pSample->renderText("Diffuse - "  + dim_to_string(mpLightProbe->getDiffuseTexture()),  bottom_left(Viewport::Diffuse));
    pSample->renderText("Specular - " + dim_to_string(mpLightProbe->getSpecularTexture()), bottom_left(Viewport::Specular));

#undef bottom_left
#undef dim_to_string
}

bool LightProbeViewer::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    bool bHandled = mCameraController.onKeyEvent(keyEvent);
    if (bHandled == false)
    {
        if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
        {
            switch (keyEvent.key)
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

bool LightProbeViewer::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    Fbo::SharedPtr pFbo = pSample->getCurrentFbo();
    uvec2 pos(mouseEvent.pos * vec2(pFbo->getWidth(), pFbo->getHeight()));

    // If clicked to the right of main viewport (i.e. the sidebar)
    if (pos.x > mMainRect.z && (mouseEvent.type == MouseEvent::Type::LeftButtonUp || mouseEvent.type == MouseEvent::Type::LeftButtonDown))
    {
        // Find clicked region
        Viewport selected = Viewport::Count;
        for (uint32_t i = 0; i < (uint32_t)Viewport::Count; i++)
        {
            uvec4 curr = mRects[i];
            if (pos.x > curr.x && pos.x < curr.z && pos.y > curr.y && pos.y < curr.w)
            {
                selected = (Viewport)i;
                break;
            }
        }

        if (selected != Viewport::Count)
        {
            if (mouseEvent.type == MouseEvent::Type::LeftButtonUp)
            {
                mSelectedView = selected;
                mpCamera->setAspectRatio(mSelectedView == Viewport::Scene ? ((float)mMainRect.z / (float)mMainRect.w) : 1.0f);
            }

            return true;
        }
    }
    else if (mSelectedView == Viewport::Scene)
    {
        return mCameraController.onMouseEvent(mouseEvent);
    }

    return false;    
}

void LightProbeViewer::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    float h = (float)height;
    float w = (float)width;

    mpCamera->setFocalLength(21.0f);

    uint32_t viewportSize = height / (uint32_t)Viewport::Count;
    uint32_t left = width - viewportSize;
    uint32_t top = 0;

    // Left, Top, Right, Down
    for (uint32_t i = 0; i < (uint32_t)Viewport::Count; i++)
    {
        mRects[i] = uvec4(left, top, width, top + viewportSize);
        top += viewportSize;
    }

    mMainRect = uvec4(0, 0, left, height);

    float aspectRatio = (mSelectedView == Viewport::Scene) ? (float)left / (float)height : 1.0f;
    mpCamera->setAspectRatio(aspectRatio);
}

void LightProbeViewer::resetCamera()
{
    if (mpScene)
    {
        // update the camera position
        float radius = mpScene->getRadius();
        mpCamera->setPosition(vec3(0.0f, 0.0f, radius * 1.5f));
        mpCamera->setTarget(vec3());
        mpCamera->setUpVector(glm::vec3(0, 1, 0));
        mpCamera->setDepthRange(0.001f, radius * 10.0f);

        // Update the controllers
        mCameraController.setCameraSpeed(radius * 0.25f);
    }
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    LightProbeViewer::UniquePtr pRenderer = std::make_unique<LightProbeViewer>();

    SampleConfig config;
    config.windowDesc.title = "Light Probe Viewer";
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

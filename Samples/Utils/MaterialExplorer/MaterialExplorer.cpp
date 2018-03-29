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
#include "MaterialExplorer.h"
#include "Data/HostDeviceSharedMacros.h"

//const std::string MaterialExplorer::kEnvMapName = "panorama_map.hdr";
const std::string MaterialExplorer::kEnvMapName = "SunTemple_Reflection.hdr";
const uint32_t MaterialExplorer::kDefaultSamples = 16  *1024;

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

void MaterialExplorer::updateLightProbe(const LightProbe::SharedPtr pLightProbe)
{
    if (mpScene->getLightProbeCount() > 0)
    {
        assert(mpScene->getLightProbeCount() == 1);
        mpScene->deleteLightProbe(0);
    }

    mpLightProbe = pLightProbe;
    mpLightProbe->setRadius(mpScene->getRadius());
    mpLightProbe->setSampler(mpLinearSampler);

    mpScene->addLightProbe(mpLightProbe);
}

void MaterialExplorer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (pGui->addButton("Load Light Probe"))
    {
        std::string filename;
        if (openFileDialog("Image files\0*.hdr;*.exr\0\0", filename))
        {
            updateLightProbe(LightProbe::create(pSample->getRenderContext().get(), filename, true, ResourceFormat::RGBA16Float, mDiffuseSamples, mSpecSamples));
            mpSkyBox = SkyBox::createFromTexture(filename);
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
        if (pGui->addIntVar("Specular##Samples", specSamples, 1, 128 * 1024))
        {
            mSpecSamples = uint32_t(specSamples);
        }

        if (pGui->addButton("Apply"))
        {
            if (mDiffuseSamples != mpLightProbe->getDiffSampleCount() || mSpecSamples != mpLightProbe->getSpecSampleCount())
            {
                LightProbe::SharedPtr pProbe = LightProbe::create(pSample->getRenderContext().get(), mpLightProbe->getOrigTexture(), mDiffuseSamples, mSpecSamples);
                updateLightProbe(pProbe);
            }
        }

        pGui->addText("Specular Mip Level");
        pGui->addIntVar("##SpecMip", mSpecMip, 0, mpLightProbe->getSpecularTexture()->getMipCount() - 1);
    }
}

void MaterialExplorer::onDataReload(SampleCallbacks* pSample)
{
}

void MaterialExplorer::onDroppedFile(SampleCallbacks* pSample, const std::string& filename)
{
    if (hasSuffix(filename, ".hdr", false) || hasSuffix(filename, ".exr", false))
    {
        LightProbe::SharedPtr pProbe = LightProbe::create(pSample->getRenderContext().get(), filename, true, ResourceFormat::RGBA16Float, mDiffuseSamples, mSpecSamples);
        updateLightProbe(pProbe);
    }
    else
    {
        msgBox("Please load a .hdr or .exr file.");
    }
}

void MaterialExplorer::onLoad(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext)
{
    mpCamera = Camera::create();
    mCameraController.attachCamera(mpCamera);

    mpState = GraphicsState::create();

    mpProgram = GraphicsProgram::createFromFile("", appendShaderExtension("MaterialExplorer.ps"));
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

    //mpLightProbe = LightProbe::create(pRenderContext.get(), kEnvMapName, true, ResourceFormat::RGBA16Float);
    //mpLightProbe->setAttenuationRadius(vec2(mpScene->getRadius() * 5.0f));
    //mpLightProbe->setSampler(mpLinearSampler);
    //mpSkyBox = SkyBox::createFromTexture(kEnvMapName);
    //mpScene->addLightProbe(mpLightProbe);
}

void MaterialExplorer::onFrameRender(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext, Fbo::SharedPtr pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if (mpLightProbe != nullptr)
    {
        mpState->setFbo(pTargetFbo);

        mCameraController.update();

        pRenderContext->pushGraphicsVars(mpVars);
        pRenderContext->pushGraphicsState(mpState);

        mpSceneRenderer->renderScene(pRenderContext.get(), mpCamera.get());
        mpSkyBox->render(pRenderContext.get(), mpCamera.get());

        pRenderContext->popGraphicsVars();
        pRenderContext->popGraphicsState();

        pRenderContext->blit(mpLightProbe->getOrigTexture()->getSRV(0, 1), pTargetFbo->getRenderTargetView(0), uvec4(-1), mTopRect);
        pRenderContext->blit(mpLightProbe->getDiffuseTexture()->getSRV(0, 1), pTargetFbo->getRenderTargetView(0), uvec4(-1), mMidRect);
        pRenderContext->blit(mpLightProbe->getSpecularTexture()->getSRV(mSpecMip, 1), pTargetFbo->getRenderTargetView(0), uvec4(-1), mBotRect, Sampler::Filter::Point);
    }
}

bool MaterialExplorer::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    bool bHandled = mCameraController.onKeyEvent(keyEvent);
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

bool MaterialExplorer::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return mCameraController.onMouseEvent(mouseEvent);
}

void MaterialExplorer::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    float h = (float)height;
    float w = (float)width;

    mpCamera->setFocalLength(21.0f);
    float aspectRatio = (w / h);
    mpCamera->setAspectRatio(aspectRatio);


    uint32_t viewportSize = height / 3;
    uint32_t left = width - viewportSize;

    mTopRect = uvec4(left, 0, width, viewportSize);
    mMidRect = uvec4(left, viewportSize, width, viewportSize * 2);
    mBotRect = uvec4(left, viewportSize * 2, width, height);
}

void MaterialExplorer::resetCamera()
{
    if(mpScene)
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
    MaterialExplorer::UniquePtr pRenderer = std::make_unique<MaterialExplorer>();

    SampleConfig config;
    config.windowDesc.title = "Material Explorer";
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
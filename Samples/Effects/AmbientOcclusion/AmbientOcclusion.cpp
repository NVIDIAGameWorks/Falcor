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
#include "AmbientOcclusion.h"

const std::string AmbientOcclusion::skDefaultModel = "Arcade/Arcade.fbx";

void AmbientOcclusion::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (mpSSAO != nullptr)
    {
        if(pGui->beginGroup("SSAO"))
        {
            mpSSAO->renderUI(pGui);
            pGui->endGroup();
        }
    }
}

void AmbientOcclusion::resetCamera()
{
    if (mpModel)
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
        mCameraController.setModelParams(modelCenter, radius, 4);
        mNearZ = std::max(0.1f, mpModel->getRadius() / 750.0f);
        mFarZ = radius * 10;
    }
}

void AmbientOcclusion::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    //
    // "GBuffer" rendering
    //

    mpPrePassProgram = GraphicsProgram::createFromFile("AOPrePass.ps.hlsl", "", "main");
    mpPrePassState = GraphicsState::create();
    mpPrePassVars = GraphicsVars::create(mpPrePassProgram->getReflector());

    RasterizerState::Desc rsDesc;
    rsDesc.setCullMode(RasterizerState::CullMode::Back);
    mpPrePassState->setRasterizerState(RasterizerState::create(rsDesc));

    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthTest(true);
    mpPrePassState->setDepthStencilState(DepthStencilState::create(dsDesc));

    mpPrePassState->setProgram(mpPrePassProgram);

    //
    // Apply AO pass
    //

    Sampler::Desc pointDesc;
    pointDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
    mpPointSampler = Sampler::create(pointDesc);

    Sampler::Desc linearDesc;
    linearDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpLinearSampler = Sampler::create(linearDesc);

    mpCopyPass = FullScreenPass::create("ApplyAO.ps.hlsl");
    mpCopyVars = GraphicsVars::create(mpCopyPass->getProgram()->getReflector());

    // Effects
    mpSSAO = SSAO::create(uvec2(1024));

    // Model
    mpModel = Model::createFromFile(skDefaultModel.c_str());

    mpCamera = Camera::create();
    mpCamera->setAspectRatio((float)pSample->getCurrentFbo()->getWidth() / (float)pSample->getCurrentFbo()->getHeight());
    mCameraController.attachCamera(mpCamera);

    resetCamera();
}

void AmbientOcclusion::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    mpCamera->setDepthRange(mNearZ, mFarZ);
    mCameraController.update();

    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpGBufferFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    // Render Scene
    mpPrePassState->setFbo(mpGBufferFbo);

    pRenderContext->setGraphicsState(mpPrePassState);
    pRenderContext->setGraphicsVars(mpPrePassVars);
    ModelRenderer::render(pRenderContext.get(), mpModel, mpCamera.get());

    // Generate AO Map
    Texture::SharedPtr pAOMap = mpSSAO->generateAOMap(pRenderContext.get(), mpCamera.get(), mpGBufferFbo->getDepthStencilTexture(), mpGBufferFbo->getColorTexture(1));

    // Apply AO Map to scene
    mpCopyVars->setSampler("gSampler", mpLinearSampler);
    mpCopyVars->setTexture("gColor", mpGBufferFbo->getColorTexture(0));
    mpCopyVars->setTexture("gAOMap", pAOMap);
    pRenderContext->setGraphicsVars(mpCopyVars);
    pRenderContext->getGraphicsState()->setFbo(pTargetFbo);
    mpCopyPass->execute(pRenderContext.get());
}

bool AmbientOcclusion::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    return false;
}

bool AmbientOcclusion::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return mCameraController.onMouseEvent(mouseEvent);
}

void AmbientOcclusion::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    mpCamera->setFocalLength(21.0f);
    mpCamera->setAspectRatio((float)width / (float)height);

    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, Falcor::ResourceFormat::RGBA8Unorm).setColorTarget(1, Falcor::ResourceFormat::RGBA8Unorm).setDepthStencilTarget(Falcor::ResourceFormat::D24UnormS8);
    mpGBufferFbo = FboHelper::create2D(width, height, fboDesc);
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    AmbientOcclusion::UniquePtr pRenderer = std::make_unique<AmbientOcclusion>();
    SampleConfig config;
    config.windowDesc.title = "Ambient Occlusion";
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

/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
#include "HelloDXR.h"

static const glm::vec4 kClearColor(0.38f, 0.52f, 0.10f, 1);
static const std::string kDefaultScene = "Arcade/Arcade.fscene";

std::string to_string(const vec3& v)
{
    std::string s;
    s += "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
    return s;
}

void HelloDXR::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    pGui->addCheckBox("Ray Trace", mRayTrace);
    if (pGui->addButton("Load Scene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileFormatString, filename))
        {
            loadScene(filename, pSample->getCurrentFbo().get());
        }
    }

    for(uint32_t i = 0 ; i < mpScene->getLightCount() ; i++)
    {    
        std::string group = "Point Light" + std::to_string(i);
        mpScene->getLight(i)->renderUI(pGui, group.c_str());
    }
}

void HelloDXR::loadScene(const std::string& filename, const Fbo* pTargetFbo)
{
    mpScene = RtScene::loadFromFile(filename, RtBuildFlags::None, Model::LoadFlags::RemoveInstancing);
    Model::SharedPtr pModel = mpScene->getModel(0);
    float radius = pModel->getRadius();

    mpCamera = mpScene->getActiveCamera();
    assert(mpCamera);

    mCamController.attachCamera(mpCamera);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    Sampler::SharedPtr pSampler = Sampler::create(samplerDesc);
    pModel->bindSamplerToMaterials(pSampler);

    // Update the controllers
    mCamController.setCameraSpeed(radius * 0.25f);
    float nearZ = std::max(0.1f, pModel->getRadius() / 750.0f);
    float farZ = radius * 10;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)pTargetFbo->getWidth() / (float)pTargetFbo->getHeight());
    mpSceneRenderer = SceneRenderer::create(mpScene);
    mpRtVars = RtProgramVars::create(mpRaytraceProgram, mpScene);
    mpRtRenderer = RtSceneRenderer::create(mpScene);
}

void HelloDXR::onLoad(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext)
{
    RtProgram::MissProgramList missProgs;
    missProgs.push_back(MissProgram::createFromFile("HelloDXR.rt.hlsl", "primaryMiss"));
    missProgs.push_back(MissProgram::createFromFile("HelloDXR.rt.hlsl", "shadowMiss"));

    RtProgram::HitProgramList hitProgs;
    hitProgs.push_back(HitProgram::createFromFile("HelloDXR.rt.hlsl", "primaryClosestHit", "", ""));
    hitProgs.push_back(HitProgram::createFromFile("HelloDXR.rt.hlsl", "", "shadowAnyHit", ""));
    RayGenProgram::SharedPtr pRayGen = RayGenProgram::createFromFile("HelloDxr.rt.hlsl", "rayGen");
    mpRaytraceProgram = RtProgram::create(pRayGen, missProgs, hitProgs);
    mpRasterProgram = GraphicsProgram::createFromFile("", "HelloDXR.ps.hlsl");

    loadScene(kDefaultScene, pSample->getCurrentFbo().get());

    mpProgramVars = GraphicsVars::create(mpRasterProgram->getActiveVersion()->getReflector());
    mpGraphicsState = GraphicsState::create();
    mpGraphicsState->setProgram(mpRasterProgram);

    mpRtState = RtState::create();
    mpRtState->setProgram(mpRaytraceProgram);
    mpRtState->setMaxTraceRecursionDepth(2); // 1 for calling rtTrace from RayGen, 1 for calling it from the primary-ray ClosestHitShader
}

void HelloDXR::renderRaster(RenderContext* pContext)
{
    mpGraphicsState->setRasterizerState(nullptr);
    mpGraphicsState->setDepthStencilState(nullptr);
    mpGraphicsState->setProgram(mpRasterProgram);
    pContext->setGraphicsState(mpGraphicsState);
    pContext->setGraphicsVars(mpProgramVars);
    mpSceneRenderer->renderScene(pContext, mpCamera.get());
}

void HelloDXR::setRayGenVars(const Fbo* pTargetFbo)
{
    PROFILE(setRayGenVars);
    GraphicsVars* pVars = mpRtVars->getRayGenVars().get();
    ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer(0, 0, 0);
    pCB["invView"] = glm::inverse(mpCamera->getViewMatrix());
    pCB["viewportDims"] = vec2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
    float fovY = focalLengthToFovY(mpCamera->getFocalLength(), Camera::kDefaultFrameHeight);
    pCB["tanHalfFovY"] = tanf(fovY * 0.5f);
}

void HelloDXR::renderRT(RenderContext* pContext, const Fbo* pTargetFbo)
{
    PROFILE(renderRT);
    setRayGenVars(pTargetFbo);

    pContext->clearUAV(mpRtOut->getUAV().get(), kClearColor);
    mpRtVars->getRayGenVars()->setUav(0, 1, 0, mpRtOut->getUAV(0, 0, 1));

    mpRtRenderer->renderScene(pContext, mpRtVars, mpRtState, uvec2(pTargetFbo->getWidth(), pTargetFbo->getHeight()), mpCamera.get());
    pContext->blit(mpRtOut->getSRV(), pTargetFbo->getRenderTargetView(0));
}

void HelloDXR::onFrameRender(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext, Fbo::SharedPtr pTargetFbo)
{
    pRenderContext->clearFbo(pTargetFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);

    if(mpScene)
    {
        mpGraphicsState->setFbo(pTargetFbo);
        mCamController.update();

        if (mRayTrace)
        {
            renderRT(pRenderContext.get(), pTargetFbo.get());
        }
        else
        {
            renderRaster(pRenderContext.get());
        }

        std::string camprops = "CamPos:     " + to_string(mpCamera->getPosition()) + "\n";
        camprops += "CamTarget:  " + to_string(mpCamera->getTarget()) + "\n";
        camprops += "CamUp:      " + to_string(mpCamera->getUpVector()) + "\n";

        camprops += "CamLookAt:  " + to_string(mpCamera->getTarget() - mpCamera->getPosition()) + "\n";
        pSample->renderText(camprops, vec2(120, 200));
    }
}

bool HelloDXR::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    if (mCamController.onKeyEvent(keyEvent))
    {
        return true;
    }
    if (keyEvent.key == KeyboardEvent::Key::Space && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        mRayTrace = !mRayTrace;
        return true;
    }
    return false;
}

bool HelloDXR::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return mCamController.onMouseEvent(mouseEvent);
}

void HelloDXR::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    float h = (float)height;
    float w = (float)width;

    mpCamera->setFocalLength(18);
    float aspectRatio = (w / h);
    mpCamera->setAspectRatio(aspectRatio);

    mpRtOut = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    HelloDXR::UniquePtr pRenderer = std::make_unique<HelloDXR>();
    SampleConfig config;
    config.windowDesc.title = "HelloDXR";
    config.windowDesc.resizableWindow = true;

    RtSample::run(config, pRenderer);
}

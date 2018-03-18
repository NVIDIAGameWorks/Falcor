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
#include "Shadows.h"

const std::string Shadows::skDefaultScene = "Arcade/Arcade.fscene";

void Shadows::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (pGui->addButton("Load Scene"))
    {
        displayLoadSceneDialog();
    }

    pGui->addCheckBox("Update Shadow Map", mControls.updateShadowMap);
    if(pGui->addIntVar("Cascade Count", mControls.cascadeCount, 1u, CSM_MAX_CASCADES))
    {
        for (uint32_t i = 0; i < mpCsmTech.size(); i++)
        {
            mpCsmTech[i]->setCascadeCount(mControls.cascadeCount);
        }
        createVisualizationProgram();
    }

    bool visualizeCascades = mPerFrameCBData.visualizeCascades != 0;
    pGui->addCheckBox("Visualize Cascades", visualizeCascades);
    mPerFrameCBData.visualizeCascades = visualizeCascades;
    pGui->addCheckBox("Display Shadow Map", mControls.showShadowMap);
    pGui->addIntVar("Displayed Cascade", mControls.displayedCascade, 0u, mControls.cascadeCount - 1);
    if (pGui->addIntVar("LightIndex", mControls.lightIndex, 0u, mpScene->getLightCount() - 1))
    {
        mLightingPass.pProgram->addDefine("_LIGHT_INDEX", std::to_string(mControls.lightIndex));
    }

    std::string groupName = "Light " + std::to_string(mControls.lightIndex);
    if (pGui->beginGroup(groupName.c_str()))
    {
        mpScene->getLight(mControls.lightIndex)->renderUI(pGui);
        pGui->endGroup();
    }
    mpCsmTech[mControls.lightIndex]->renderUi(pGui, "CSM");
}

void Shadows::displayLoadSceneDialog()
{
    std::string filename;
    if(openFileDialog(Scene::kFileFormatString, filename))
    {
        createScene(filename);
    }
}

void Shadows::setLightIndex(int32_t index)
{
    mControls.lightIndex = max(min(index, (int32_t)mpScene->getLightCount() - 1), 0);
}

void Shadows::createScene(const std::string& filename)
{
    // Load the scene
    mpScene = Scene::loadFromFile(filename);

    // Create the renderer
    mpRenderer = SceneRenderer::create(mpScene);
    mpRenderer->setCameraControllerType(SceneRenderer::CameraControllerType::FirstPerson);

    mpCsmTech.resize(mpScene->getLightCount());
    for(uint32_t i = 0; i < mpScene->getLightCount(); i++)
    {
        mpCsmTech[i] = CascadedShadowMaps::create(2048, 2048, mpScene->getLight(i).get(), mpScene.get(), mControls.cascadeCount);
        mpCsmTech[i]->setFilterMode(CsmFilterHwPcf);
        mpCsmTech[i]->setVsmLightBleedReduction(0.3f);
    }
    setLightIndex(0);

    // Create the main effect
    mLightingPass.pProgram = GraphicsProgram::createFromFile(appendShaderExtension("Shadows.vs"), appendShaderExtension("Shadows.ps"));
    mLightingPass.pProgram->addDefine("_LIGHT_COUNT", std::to_string(mpScene->getLightCount()));
    mLightingPass.pProgram->addDefine("_LIGHT_INDEX", std::to_string(mControls.lightIndex));
    mLightingPass.pProgramVars = GraphicsVars::create(mLightingPass.pProgram->getActiveVersion()->getReflector());
    ConstantBuffer::SharedPtr pCB = mLightingPass.pProgramVars->getConstantBuffer(0, 0, 0);
    mOffsets.visualizeCascades = static_cast<uint32_t>(pCB->getVariableOffset("visualizeCascades"));
}

void Shadows::onLoad(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext)
{
    createScene(skDefaultScene);
    createVisualizationProgram();
}

void Shadows::runMainPass(RenderContext* pContext)
{
    //Only part of the gfx state I actually want to set
    pContext->getGraphicsState()->setProgram(mLightingPass.pProgram);

    //vars
    ConstantBuffer::SharedPtr pPerFrameCB = mLightingPass.pProgramVars->getConstantBuffer(0, 0, 0);
    pPerFrameCB->setBlob(&mPerFrameCBData, mOffsets.visualizeCascades, sizeof(mPerFrameCBData));
    pContext->pushGraphicsVars(mLightingPass.pProgramVars);
    
    mpRenderer->renderScene(pContext);

    pContext->popGraphicsVars();
}

void Shadows::displayShadowMap(RenderContext* pContext)
{
    mShadowVisualizer.pProgramVars->setTexture("gTexture", mpCsmTech[mControls.lightIndex]->getShadowMap());
    if (mControls.cascadeCount > 1)
    {
        mShadowVisualizer.pProgramVars->getConstantBuffer("PerImageCB")->setBlob(&mControls.displayedCascade, mOffsets.displayedCascade, sizeof(mControls.displayedCascade));
    }
    pContext->pushGraphicsVars(mShadowVisualizer.pProgramVars);
    mShadowVisualizer.pProgram->execute(pContext);
    pContext->popGraphicsVars();
}

void Shadows::onFrameRender(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext, Fbo::SharedPtr pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if(mpScene)
    {
        // Update the scene
        mpRenderer->update(pSample->getCurrentTime());

        // Run the shadow pass
        if(mControls.updateShadowMap)
        {
            mPerFrameCBData.camVpAtLastCsmUpdate = mpScene->getActiveCamera()->getViewProjMatrix();
            for(uint32_t i = 0; i < mpCsmTech.size(); i++)
            {
                mpCsmTech[i]->setup(pRenderContext.get(), mpScene->getActiveCamera().get(), nullptr);
            }
        }

        // Put shadow data in program vars
        for(uint32_t i = 0; i < mpCsmTech.size(); i++)
        {
            std::string var = "gCsmData[" + std::to_string(i) + "]";
            mpCsmTech[i]->setDataIntoGraphicsVars(mLightingPass.pProgramVars, var);
        }

        if(mControls.showShadowMap)
        {
            displayShadowMap(pRenderContext.get());
        }
        else
        {
            runMainPass(pRenderContext.get());
        }
    }

    pSample->renderText(pSample->getFpsMsg(), glm::vec2(10, 10));
}

bool Shadows::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    return mpRenderer->onKeyEvent(keyEvent);
}

bool Shadows::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return mpRenderer->onMouseEvent(mouseEvent);
}

void Shadows::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    //Camera aspect 
    Camera::SharedPtr activeCamera = mpScene->getActiveCamera();
    activeCamera->setFocalLength(21.0f);
    float aspectRatio = (float(width) / float(height));
    activeCamera->setAspectRatio(aspectRatio);
}

void Shadows::createVisualizationProgram()
{
    // Create the shadow visualizer
    mShadowVisualizer.pProgram = FullScreenPass::create(appendShaderExtension("VisualizeMap.ps"));
    if(mControls.cascadeCount > 1)
    {
        mShadowVisualizer.pProgram->getProgram()->addDefine("_USE_2D_ARRAY");
        mShadowVisualizer.pProgramVars = GraphicsVars::create(mShadowVisualizer.pProgram->getProgram()->getActiveVersion()->getReflector());
        mOffsets.displayedCascade = static_cast<uint32_t>(mShadowVisualizer.pProgramVars->getConstantBuffer("PerImageCB")->getVariableOffset("cascade"));
    }
    else
    {
        mShadowVisualizer.pProgramVars = GraphicsVars::create(mShadowVisualizer.pProgram->getProgram()->getActiveVersion()->getReflector());
    }
}

 void Shadows::onInitializeTesting(SampleCallbacks* pSample)
 {
     auto argList = pSample->getArgList();
     std::vector<ArgList::Arg> specifiedScene = argList.getValues("loadscene");
     if (!specifiedScene.empty())
     {
         createScene(specifiedScene[0].asString());
     }
 
     std::vector<ArgList::Arg> filterFrames = argList.getValues("incrementFilter");
     if (!filterFrames.empty())
     {
         mFilterFrames.resize(filterFrames.size());
         for (uint32_t i = 0; i < filterFrames.size(); ++i)
         {
             mFilterFrames[i] = filterFrames[i].asUint();
         }
     }
 
     //Set to first filter mode because it's going to be incrementing
     mFilterFramesIt = mFilterFrames.begin();
     for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
     {
         mpCsmTech[i]->setFilterMode(CsmFilterPoint);
     }
 }

 void Shadows:: onEndTestFrame(SampleCallbacks* pSample, SampleTest* pSampleTest)
 {
     uint32_t frameId = pSample->getFrameID();
     if (mFilterFramesIt != mFilterFrames.end() && frameId >= *mFilterFramesIt)
     {
         ++mFilterFramesIt;
         uint32_t nextFilterMode = mpCsmTech[0]->getFilterMode() + 1;
         nextFilterMode = min(nextFilterMode, static_cast<uint32_t>(CsmFilterStochasticPcf));
         for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
         {
             mpCsmTech[i]->setFilterMode(nextFilterMode);
         }
     }
 }

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    Shadows::UniquePtr pRenderer = std::make_unique<Shadows>();
    SampleConfig config;
    config.windowDesc.title = "Shadows Sample";
#ifdef _WIN32
    Sample::run(config, pRenderer);
#else
    config.argc = (uint32_t)argc;
    config.argv = argv;
    Sample::run(config, pRenderer);
#endif
    return 0;
}

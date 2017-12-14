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

void Shadows::onGuiRender()
{
    if (mpGui->addButton("Load Scene"))
    {
        displayLoadSceneDialog();
    }

    mpGui->addCheckBox("Update Shadow Map", mControls.updateShadowMap);
    if(mpGui->addIntVar("Cascade Count", mControls.cascadeCount, 1u, CSM_MAX_CASCADES))
    {
        for (uint32_t i = 0; i < mpCsmTech.size(); i++)
        {
            mpCsmTech[i]->setCascadeCount(mControls.cascadeCount);
        }
        createVisualizationProgram();
    }

    bool visualizeCascades = mPerFrameCBData.visualizeCascades != 0;
    mpGui->addCheckBox("Visualize Cascades", visualizeCascades);
    mPerFrameCBData.visualizeCascades = visualizeCascades;
    mpGui->addCheckBox("Display Shadow Map", mControls.showShadowMap);
    mpGui->addIntVar("Displayed Cascade", mControls.displayedCascade, 0u, mControls.cascadeCount - 1);
    if (mpGui->addIntVar("LightIndex", mControls.lightIndex, 0u, mpScene->getLightCount() - 1))
    {
        mLightingPass.pProgram->addDefine("_LIGHT_INDEX", std::to_string(mControls.lightIndex));
    }

    std::string groupName = "Light " + std::to_string(mControls.lightIndex);
    if (mpGui->beginGroup(groupName.c_str()))
    {
        mpScene->getLight(mControls.lightIndex)->renderUI(mpGui.get());
        mpGui->endGroup();
    }
    mpCsmTech[mControls.lightIndex]->renderUi(mpGui.get(), "CSM");
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
    for (uint32_t i = 0; i < mpScene->getPathCount(); i++)
    {
        mpScene->getPath(i)->detachAllObjects();
    }
    // Create the renderer
    mpRenderer = SceneRenderer::create(mpScene);
    mpRenderer->setCameraControllerType(SceneRenderer::CameraControllerType::FirstPerson);

    mpCsmTech.resize(mpScene->getLightCount());
    for(uint32_t i = 0; i < mpScene->getLightCount(); i++)
    {
        mpCsmTech[i] = CascadedShadowMaps::create(2048, 2048, mpScene->getLight(i), mpScene, mControls.cascadeCount);
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

void Shadows::onLoad()
{
    createScene("Scenes/DragonPlane.fscene");
    createVisualizationProgram();
    initializeTesting();
}

void Shadows::runMainPass()
{
    //Only part of the gfx state I actually want to set
    mpRenderContext->getGraphicsState()->setProgram(mLightingPass.pProgram);

    //vars
    ConstantBuffer::SharedPtr pPerFrameCB = mLightingPass.pProgramVars->getConstantBuffer(0, 0, 0);
    pPerFrameCB->setBlob(&mPerFrameCBData, mOffsets.visualizeCascades, sizeof(mPerFrameCBData));
    mpRenderContext->pushGraphicsVars(mLightingPass.pProgramVars);
    
    mpRenderer->renderScene(mpRenderContext.get());

    mpRenderContext->popGraphicsVars();
}

void Shadows::displayShadowMap()
{
    mShadowVisualizer.pProgramVars->setSrv(0, 0, 0, mpCsmTech[mControls.lightIndex]->getShadowMap()->getSRV());
    if (mControls.cascadeCount > 1)
    {
        mShadowVisualizer.pProgramVars->getConstantBuffer(0, 0, 0)->setBlob(&mControls.displayedCascade, mOffsets.displayedCascade, sizeof(mControls.displayedCascade));
    }
    mpRenderContext->pushGraphicsVars(mShadowVisualizer.pProgramVars);
    mShadowVisualizer.pProgram->execute(mpRenderContext.get());
    mpRenderContext->popGraphicsVars();
}

void Shadows::onFrameRender()
{
    beginTestFrame();

    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    mpRenderContext->clearFbo(mpDefaultFBO.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if(mpScene)
    {
        // Update the scene
        mpRenderer->update(mCurrentTime);

        // Run the shadow pass
        if(mControls.updateShadowMap)
        {
            mPerFrameCBData.camVpAtLastCsmUpdate = mpScene->getActiveCamera()->getViewProjMatrix();
            for(uint32_t i = 0; i < mpCsmTech.size(); i++)
            {
                mpCsmTech[i]->setup(mpRenderContext.get(), mpScene->getActiveCamera().get(), nullptr);
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
            displayShadowMap();
        }
        else
        {
            runMainPass();
        }
    }

    renderText(getFpsMsg(), glm::vec2(10, 10));
    
    endTestFrame();
}

void Shadows::onShutdown()
{
}

bool Shadows::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return mpRenderer->onKeyEvent(keyEvent);
}

bool Shadows::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpRenderer->onMouseEvent(mouseEvent);
}

void Shadows::onResizeSwapChain()
{
    //Camera aspect 
    float height = (float)mpDefaultFBO->getHeight();
    float width = (float)mpDefaultFBO->getWidth();
    Camera::SharedPtr activeCamera = mpScene->getActiveCamera();
    activeCamera->setFocalLength(21.0f);
    float aspectRatio = (width / height);
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

void Shadows::onInitializeTesting()
{
    std::vector<ArgList::Arg> specifiedScene = mArgList.getValues("loadscene");

    if (!specifiedScene.empty())
    {
        createScene(specifiedScene[0].asString());
    }

    std::vector<ArgList::Arg> filterFrames = mArgList.getValues("incrementFilter");
    



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

void Shadows:: onEndTestFrame()
{
    uint32_t frameId = frameRate().getFrameCount();
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
    Shadows shadows;
    SampleConfig config;
    config.windowDesc.title = "Shadows Sample";
#ifdef _WIN32
    shadows.run(config);
#else
    shadows.run(config, (uint32_t)argc, argv);
#endif
    return 0;
}

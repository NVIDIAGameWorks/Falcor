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
#include "NormalMapFiltering.h"

void NormalMapFiltering::onGuiRender()
{
    const Scene* pScene = mpRenderer->getScene().get();
    for(uint32_t i = 0; i < pScene->getLightCount(); i++)
    {
        std::string group = "Light " + std::to_string(i);
        if(mpGui->beginGroup(group.c_str()))
        {
            pScene->getLight(i)->renderUI(mpGui.get());
            mpGui->endGroup();
        }
    }

    if (mpGui->addCheckBox("Lean Map", mUseLeanMap) || mpGui->addCheckBox("Specular AA", mUseSpecAA))
    {
        updateProgram();
    }
}

void NormalMapFiltering::updateProgram()
{
    std::string lights;
    mpProgram->clearDefines();
    mpProgram->addDefine("_LIGHT_COUNT", std::to_string(mpRenderer->getScene()->getLightCount()));
    if (mUseLeanMap)
    {
        mpProgram->addDefine("_MS_LEAN_MAPPING");
        mpProgram->addDefine("_LEAN_MAP_COUNT", std::to_string(mpLeanMap->getRequiredLeanMapShaderArraySize()));
    }
    if (mUseSpecAA == false)
    {
        mpProgram->addDefine("_MS_DISABLE_ROUGHNESS_FILTERING");
    }

    mpVars = GraphicsVars::create(mpProgram->getActiveVersion()->getReflector());
    if(mUseLeanMap)
    {
        mpLeanMap->setIntoProgramVars(mpVars.get(), mpLinearSampler);
    }
}

void NormalMapFiltering::onLoad()
{
    Scene::SharedPtr pScene = Scene::loadFromFile("Scenes/ogre.fscene");
    if(pScene == nullptr)
    {
        exit(1);
    }
    mpRenderer = SceneRenderer::create(pScene);
    mpLeanMap = LeanMap::create(pScene.get());
    mpProgram = GraphicsProgram::createFromFile("", "NormalMapFiltering.ps.hlsl");
    
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpLinearSampler = Sampler::create(samplerDesc);
    pScene->getModel(0)->bindSamplerToMaterials(mpLinearSampler);

    updateProgram();
    mCameraController.attachCamera(pScene->getCamera(0));
    mCameraController.setModelParams(pScene->getModel(0)->getCenter(), pScene->getModel(0)->getRadius(), 4);

    initializeTesting();
}

void NormalMapFiltering::onFrameRender()
{
    beginTestFrame();

    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    mpRenderContext->clearFbo(mpDefaultFBO.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    auto pState = mpRenderContext->getGraphicsState();
    pState->setBlendState(nullptr);
    pState->setDepthStencilState(nullptr);
    pState->setProgram(mpProgram);
    mpRenderContext->pushGraphicsVars(mpVars);
    mCameraController.update();
    mpRenderer->renderScene(mpRenderContext.get());
    mpRenderContext->popGraphicsVars();

    endTestFrame();
}

void NormalMapFiltering::onShutdown()
{

}

bool NormalMapFiltering::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return mCameraController.onKeyEvent(keyEvent);
}

bool NormalMapFiltering::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mCameraController.onMouseEvent(mouseEvent);
}

void NormalMapFiltering::onResizeSwapChain()
{
    float aspect = (float)mpDefaultFBO->getWidth() / (float)mpDefaultFBO->getHeight();
    mpRenderer->getScene()->getActiveCamera()->setAspectRatio(aspect);
}

void NormalMapFiltering::onInitializeTesting()
{
    std::vector<ArgList::Arg> modeFrames = mArgList.getValues("changeMode");
    if (!modeFrames.empty())
    {
        mChangeModeFrames.resize(modeFrames.size());
        for (uint32_t i = 0; i < modeFrames.size(); ++i)
        {
            mChangeModeFrames[i] = modeFrames[i].asUint();
        }
    }

    mChangeModeIt = mChangeModeFrames.begin();
    mUseLeanMap = false;
    mUseSpecAA = false;
    updateProgram();
}

void NormalMapFiltering::onEndTestFrame()
{
    static const uint32_t numCombos = 4;
    static const bool useLeanMap[numCombos] = {false, false, true, true };
    static const bool useSpecAA[numCombos] = {false, true, false, true };
    static uint32_t index = 0;

    uint32_t frameId = frameRate().getFrameCount();
    if (mChangeModeIt != mChangeModeFrames.end() && frameId >= *mChangeModeIt)
    {
        ++mChangeModeIt;
        ++index;
        //wrap around so no crash if too many args supplied
        if (index == numCombos)
        {
            index = 0;
        }

        mUseLeanMap = useLeanMap[index];
        mUseSpecAA = useSpecAA[index];
        updateProgram();
    }
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    NormalMapFiltering sample;
    SampleConfig config;
    config.windowDesc.title = "Normal Map Filtering";
    config.windowDesc.width = 1350;
    config.windowDesc.height = 1080;
#ifdef _WIN32
    sample.run(config);
#else
    sample.run(config, (uint32_t)argc, argv);
#endif
    return 0;
}

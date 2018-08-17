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
#include "Particles.h"
#include <limits>

const Gui::DropdownList kPixelShaders 
{
    {0, "ConstColor"},
    {1, "ColorInterp"},
    {2, "Textured"}
};

const char* kConstColorPs = "Effects/ParticleConstColor.ps.slang";
const char* kColorInterpPs = "Effects/ParticleInterpColor.ps.slang";
const char* kTexturedPs = "Effects/ParticleTexture.ps.slang";
const std::string kDefaultTexture = "smoke-puff.png";

void Particles::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    createSystemGui(pSample->getRenderContext().get(), pGui);
    pGui->addSeparator();
    editPropertiesGui(pGui);
}

void Particles::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mpCamera = Camera::create();
    mpCamera->setPosition(mpCamera->getPosition() + glm::vec3(0, 5, 10));
    mpCamController.attachCamera(mpCamera);
    mpTextures.push_back(createTextureFromFile(kDefaultTexture, true, false));
    mGuiData.mTexDropdown.push_back({ 0, kDefaultTexture });
    BlendState::Desc blendDesc;
    blendDesc.setRtBlend(0, true);
    blendDesc.setRtParams(0, BlendState::BlendOp::Add, BlendState::BlendOp::Add, BlendState::BlendFunc::SrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha,
        BlendState::BlendFunc::SrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha);
    BlendState::SharedPtr pBlend = BlendState::create(blendDesc);
    pRenderContext->getGraphicsState()->setBlendState(pBlend);
}

void Particles::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
	const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    mpCamController.update();

    for (auto it = mpParticleSystems.begin(); it != mpParticleSystems.end(); ++it)
    {
        (*it)->update(pRenderContext.get(), pSample->getLastFrameTime(), mpCamera->getViewMatrix());
        (*it)->render(pRenderContext.get(), mpCamera->getViewMatrix(), mpCamera->getProjMatrix());
    }
}

bool Particles::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    mpCamController.onKeyEvent(keyEvent);
    return false;
}

bool Particles::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    mpCamController.onMouseEvent(mouseEvent);
    return false;
}

void Particles::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    mpCamera->setFocalLength(21.0f);
    float aspectRatio = ((float)width / (float)height);
    mpCamera->setAspectRatio(aspectRatio);
}

void Particles::createSystemGui(RenderContext* pContext, Gui* pGui)
{
    if (pGui->beginGroup("Create System"))
    {
        pGui->addIntVar("Max Particles", mGuiData.mMaxParticles, 0);
        pGui->addIntVar("Max Emit Per Frame", mGuiData.mMaxEmitPerFrame, 0);
        pGui->addCheckBox("Sorted", mGuiData.mSortSystem);
        pGui->addDropdown("PixelShader", kPixelShaders, mGuiData.mPixelShaderIndex);
        if (pGui->addButton("Create"))
        {
            switch ((ExamplePixelShaders)mGuiData.mPixelShaderIndex)
            {
            case ExamplePixelShaders::ConstColor:
            {
                ParticleSystem::SharedPtr pSys = ParticleSystem::create(pContext, mGuiData.mMaxParticles,
                    mGuiData.mMaxEmitPerFrame, kConstColorPs, ParticleSystem::kDefaultSimulateShader, mGuiData.mSortSystem);
                mpParticleSystems.push_back(pSys);
                mPsData.push_back(vec4(0.f, 0.f, 0.f, 1.f));
                mpParticleSystems[mpParticleSystems.size() - 1]->getDrawVars()->getDefaultBlock()->getConstantBuffer("PsPerFrame")->setBlob(&mPsData[mPsData.size() - 1].colorData.color1, 0, sizeof(vec4));
                break;
            }
            case ExamplePixelShaders::ColorInterp:
            {
                ParticleSystem::SharedPtr pSys = ParticleSystem::create(pContext, mGuiData.mMaxParticles,
                    mGuiData.mMaxEmitPerFrame, kColorInterpPs, ParticleSystem::kDefaultSimulateShader, mGuiData.mSortSystem);
                mpParticleSystems.push_back(pSys);
                ColorInterpPsPerFrame perFrame;
                perFrame.color1 = vec4(1.f, 0.f, 0.f, 1.f);
                perFrame.colorT1 = pSys->getParticleDuration();
                perFrame.color2 = vec4(0.f, 0.f, 1.f, 1.f);
                perFrame.colorT2 = 0.f;
                mPsData.push_back(perFrame);
                mpParticleSystems[mpParticleSystems.size() - 1]->getDrawVars()->getDefaultBlock()->getConstantBuffer("PsPerFrame")->setBlob(&mPsData[mPsData.size() - 1].colorData, 0, sizeof(ColorInterpPsPerFrame));
                break;
            }
            case ExamplePixelShaders::Textured:
            {
                ParticleSystem::SharedPtr pSys = ParticleSystem::create(pContext, mGuiData.mMaxParticles,
                    mGuiData.mMaxEmitPerFrame, kTexturedPs, ParticleSystem::kDefaultSimulateShader, mGuiData.mSortSystem);
                mpParticleSystems.push_back(pSys);
                ColorInterpPsPerFrame perFrame;
                perFrame.color1 = vec4(1.f, 1.f, 1.f, 1.f);
                perFrame.colorT1 = pSys->getParticleDuration();
                perFrame.color2 = vec4(1.f, 1.f, 1.f, 0.1f);
                perFrame.colorT2 = 0.f;
                mPsData.push_back(PixelShaderData(0, perFrame));
                pSys->getDrawVars()->setTexture("gTex", mpTextures[0]);
                mpParticleSystems[mpParticleSystems.size() - 1]->getDrawVars()->getConstantBuffer("PsPerFrame")->setBlob(&mPsData[mPsData.size() - 1].colorData, 0, sizeof(ColorInterpPsPerFrame));
                break;
            }
            default:
            {
                should_not_get_here();
            }
            }
        }
        pGui->endGroup();
    }
}

void Particles::editPropertiesGui(Gui* pGui)
{
    pGui->addIntVar("System index", mGuiData.mSystemIndex, 0, ((int32_t)mpParticleSystems.size()) - 1);
    pGui->addSeparator();

    //If there are no systems yet, don't let user touch properties
    if (mGuiData.mSystemIndex < 0)
        return;

    //properties shared by all systems
    if (pGui->beginGroup("Common Properties"))
    {
        mpParticleSystems[mGuiData.mSystemIndex]->renderUi(pGui);
        pGui->endGroup();
    }
    //pixel shader specific properties
    if (pGui->beginGroup("Pixel Shader Properties"))
    {
        switch ((ExamplePixelShaders)mPsData[mGuiData.mSystemIndex].type)
        {
        case ExamplePixelShaders::ConstColor:
        {
            if (pGui->addRgbaColor("Color", mPsData[mGuiData.mSystemIndex].colorData.color1))
            {
                mpParticleSystems[mGuiData.mSystemIndex]->getDrawVars()->getConstantBuffer("PsPerFrame")->setBlob(&mPsData[mGuiData.mSystemIndex].colorData.color1, 0, sizeof(vec4));
            }
            break;
        }
        case ExamplePixelShaders::ColorInterp:
        {
            updateColorInterpolation(pGui);
            break;
        }
        case ExamplePixelShaders::Textured:
        {
            if (pGui->addButton("Add Texture"))
            {
                std::string filename;
                openFileDialog("Supported Formats\0*.png;*.dds;*.jpg;\0\0", filename);
                mpTextures.push_back(createTextureFromFile(filename, true, false));
                mGuiData.mTexDropdown.push_back({ (int32_t)mGuiData.mTexDropdown.size(), filename });
            }

            uint32_t texIndex = mPsData[mGuiData.mSystemIndex].texIndex;
            if (pGui->addDropdown("Texture", mGuiData.mTexDropdown, texIndex))
            {
                mpParticleSystems[mGuiData.mSystemIndex]->getDrawVars()->setTexture("gTex", mpTextures[texIndex]);
                mPsData[mGuiData.mSystemIndex].texIndex = texIndex;
            }

            updateColorInterpolation(pGui);
            break;
        }
        default:
            should_not_get_here();
        }

        pGui->endGroup();
    }
}

void Particles::updateColorInterpolation(Gui* pGui)
{
    bool dirty = pGui->addRgbaColor("Color1", mPsData[mGuiData.mSystemIndex].colorData.color1);
    dirty |= pGui->addFloatVar("Color T1", mPsData[mGuiData.mSystemIndex].colorData.colorT1);
    dirty |= pGui->addRgbaColor("Color2", mPsData[mGuiData.mSystemIndex].colorData.color2);
    dirty |= pGui->addFloatVar("Color T2", mPsData[mGuiData.mSystemIndex].colorData.colorT2);

    if (dirty)
    {
        mpParticleSystems[mGuiData.mSystemIndex]->getDrawVars()->getConstantBuffer("PsPerFrame")->setBlob(&mPsData[mGuiData.mSystemIndex].colorData, 0, sizeof(ColorInterpPsPerFrame));
    }
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
#ifdef FALCOR_VK
    msgBox("Support for Particle Systems in Vulkan is coming soon!");
    return 0 ;
#endif

    Particles::UniquePtr pRenderer = std::make_unique<Particles>();

    SampleConfig config;
    config.windowDesc.title = "Particles";
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

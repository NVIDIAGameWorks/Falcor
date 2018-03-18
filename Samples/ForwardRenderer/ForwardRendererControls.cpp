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
#include "ForwardRenderer.h"

Gui::DropdownList kSampleCountList = 
{
    { 1, "1" },
    { 2, "2" },
    { 4, "4" },
    { 8, "8" },
};

const Gui::DropdownList aaModeList = 
{
    { 0, "MSAA" },
    { 1, "TAA" }
};


void ForwardRenderer::initControls()
{
    mControls.resize(ControlID::Count);
    mControls[ControlID::SuperSampling] = { false, false, "INTERPOLATION_MODE", "sample" };
    mControls[ControlID::EnableShadows] = { true, false, "_ENABLE_SHADOWS" };
    mControls[ControlID::EnableReflections] = { false, false, "_ENABLE_REFLECTIONS" };
    mControls[ControlID::EnableHashedAlpha] = { true, true, "_DEFAULT_ALPHA_TEST" };
    mControls[ControlID::EnableTransparency] = { false, false, "_ENABLE_TRANSPARENCY" };
    mControls[ControlID::EnableSSAO] = { true, false, "" };
    mControls[ControlID::VisualizeCascades] = { false, false, "_VISUALIZE_CASCADES" };

    for (uint32_t i = 0 ; i < ControlID::Count ; i++)
    {
        applyLightingProgramControl((ControlID)i);
    }
}

void ForwardRenderer::applyLightingProgramControl(ControlID controlId)
{
    const ProgramControl control = mControls[controlId];
    if(control.define.size())
    {
        bool add = control.unsetOnEnabled ? !control.enabled : control.enabled;
        if (add)
        {
            mLightingPass.pProgram->addDefine(control.define, control.value);
            if (controlId == ControlID::EnableHashedAlpha) mDepthPass.pProgram->addDefine(control.define, control.value);
        }
        else
        {
            mLightingPass.pProgram->removeDefine(control.define);
            if (controlId == ControlID::EnableHashedAlpha) mDepthPass.pProgram->removeDefine(control.define);
        }
    }
}

void ForwardRenderer::applyAaMode(SampleCallbacks* pSample)
{
    if (mLightingPass.pProgram == nullptr) return;

    uint32_t w = pSample->getCurrentFbo()->getWidth();
    uint32_t h = pSample->getCurrentFbo()->getHeight();

    // Common FBO desc (2 color outputs - color and normal)
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setColorTarget(1, ResourceFormat::RGBA8Unorm).setDepthStencilTarget(ResourceFormat::D32Float);

    // Release the TAA FBOs
    mTAA.resetFbos();

    if (mAAMode == AAMode::MSAA)
    {
        mLightingPass.pProgram->removeDefine("_OUTPUT_MOTION_VECTORS");
        applyLightingProgramControl(SuperSampling);
        fboDesc.setSampleCount(mMSAASampleCount);
    }
    else if (mAAMode == AAMode::TAA)
    {
        mLightingPass.pProgram->removeDefine("INTERPOLATION_MODE");
        mLightingPass.pProgram->addDefine("_OUTPUT_MOTION_VECTORS");
        fboDesc.setColorTarget(2, ResourceFormat::RG16Float);

        Fbo::Desc taaFboDesc;
        taaFboDesc.setColorTarget(0, ResourceFormat::RGBA32Float);
        mTAA.createFbos(w, h, taaFboDesc);
    }

    mpMainFbo = FboHelper::create2D(w, h, fboDesc);
    mpDepthPassFbo = Fbo::create();
    mpDepthPassFbo->attachDepthStencilTarget(mpMainFbo->getDepthStencilTexture());
}

void ForwardRenderer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    static const char* kImageFileString = "Image files\0*.jpg;*.bmp;*.dds;*.png;*.tiff;*.tif;*.tga;*.hdr;*.exr\0\0";
    if (pGui->addButton("Load Model"))
    {
        std::string filename;
        if (openFileDialog(Model::kSupportedFileFormatsStr, filename))
        {
            loadModel(pSample, filename, true);
        }
    }

    if (pGui->addButton("Load Scene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileFormatString, filename))
        {
            loadScene(pSample, filename, true);
        }
    }

    if (mpSceneRenderer)
    {
        if (pGui->addButton("Load SkyBox Texture"))
        {
            std::string filename;
            if (openFileDialog(kImageFileString, filename))
            {
                initSkyBox(filename);
            }
        }

        if(pGui->beginGroup("Scene Settings"))
        {
            Scene* pScene = mpSceneRenderer->getScene().get();
            float camSpeed = pScene->getCameraSpeed();
            if (pGui->addFloatVar("Camera Speed", camSpeed))
            {
                pScene->setCameraSpeed(camSpeed);
            }

            vec2 depthRange(pScene->getActiveCamera()->getNearPlane(), pScene->getActiveCamera()->getFarPlane());
            if (pGui->addFloat2Var("Depth Range", depthRange, 0, FLT_MAX))
            {
                pScene->getActiveCamera()->setDepthRange(depthRange.x, depthRange.y);
            }

            if (pScene->getPathCount() > 0)
            {
                if (pGui->addCheckBox("Camera Path", mUseCameraPath))
                {
                    applyCameraPathState();
                }
            }

            if (pScene->getLightCount() && pGui->beginGroup("Light Sources"))
            {
                for (uint32_t i = 0; i < pScene->getLightCount(); i++)
                {
                    Light* pLight = pScene->getLight(i).get();
                    pLight->renderUI(pGui, pLight->getName().c_str());
                }
                pGui->endGroup();
            }
            pGui->endGroup();
        }

        if(pGui->beginGroup("Renderer Settings"))
        {
            pGui->addCheckBox("Depth Pass", mEnableDepthPass);
            pGui->addTooltip("Run a depth-pass at the beginning of the frame");

            if (pGui->addCheckBox("Per-Material Shaders", mPerMaterialShader))
            {
                mpSceneRenderer->toggleStaticMaterialCompilation(mPerMaterialShader);
            }
            pGui->addTooltip("Create a specialized version of the lighting program for each material in the scene");

            uint32_t maxAniso = mpSceneSampler->getMaxAnisotropy();
            if (pGui->addIntVar("Max Anisotropy", (int&)maxAniso, 1, 16))
            {
                setSceneSampler(maxAniso);
            }

            pGui->endGroup();
        }

        //  Anti-Aliasing Controls.
        if (pGui->beginGroup("Anti-Aliasing"))
        {
            bool reapply = false;
            reapply = reapply || pGui->addDropdown("AA Mode", aaModeList, (uint32_t&)mAAMode);

            if (mAAMode == AAMode::MSAA)
            {
                reapply = reapply || pGui->addDropdown("Sample Count", kSampleCountList, mMSAASampleCount);

                if (pGui->addCheckBox("Super Sampling", mControls[ControlID::SuperSampling].enabled))
                {
                    applyLightingProgramControl(ControlID::SuperSampling);
                }
            }

            //  Temporal Anti-Aliasing.
            if (mAAMode == AAMode::TAA)
            {
                if (pGui->beginGroup("TAA"))
                {
                    //  Render the TAA UI.
                    mTAA.pTAA->renderUI(pGui);

                    //  Choose the Sample Pattern for TAA.
                    Gui::DropdownList samplePatternList;
                    samplePatternList.push_back({ (uint32_t)SamplePattern::Halton, "Halton" });
                    samplePatternList.push_back({ (uint32_t)SamplePattern::DX11, "DX11" });
                    pGui->addDropdown("Sample Pattern", samplePatternList, (uint32_t&)mTAASamplePattern);

                    // Disable super-sampling
                    pGui->endGroup();
                }
            }

            if (reapply) applyAaMode(pSample);

            pGui->endGroup();
        }

        if (pGui->beginGroup("Light Probes"))
        {
            if (pGui->addButton("Add/Change Light Probe"))
            {
                std::string filename;
                if (openFileDialog(kImageFileString, filename))
                {
                    initLightProbe(filename);
                }
            }

            Scene::SharedPtr pScene = mpSceneRenderer->getScene();
            if (pScene->getLightProbeCount() > 0)
            {
                if (pGui->addCheckBox("Enable", mControls[ControlID::EnableReflections].enabled))
                {
                    applyLightingProgramControl(ControlID::EnableReflections);
                }
                pGui->addSeparator();
                pScene->getLightProbe(0)->renderUI(pGui);
            }
            
            pGui->endGroup();
        }

        mpToneMapper->renderUI(pGui, "Tone-Mapping");

        if (pGui->beginGroup("Shadows"))
        {
            if (pGui->addCheckBox("Enable Shadows", mControls[ControlID::EnableShadows].enabled))
            {
                applyLightingProgramControl(ControlID::EnableShadows);
            }
            pGui->endGroup();
        }

        if (pGui->beginGroup("SSAO"))
        {
            if (pGui->addCheckBox("Enable SSAO", mControls[ControlID::EnableSSAO].enabled))
            {
                applyLightingProgramControl(ControlID::EnableSSAO);
            }

            if (mControls[ControlID::EnableSSAO].enabled)
            {
                mSSAO.pSSAO->renderGui(pGui);
            }
            pGui->endGroup();
        }

        if (pGui->beginGroup("Transparency"))
        {
            if (pGui->addCheckBox("Enable Transparency", mControls[ControlID::EnableTransparency].enabled))
            {
                applyLightingProgramControl(ControlID::EnableTransparency);
            }
            pGui->addFloatVar("Opacity Scale", mOpacityScale, 0, 1);
            pGui->endGroup();
        }

        if (pGui->addCheckBox("Hashed-Alpha Test", mControls[ControlID::EnableHashedAlpha].enabled))
        {
            applyLightingProgramControl(ControlID::EnableHashedAlpha);
        }
    }
}

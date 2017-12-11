/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "HashedAlpha.h"

const Gui::DropdownList HashedAlpha::kModeList = { 
    { (uint32_t)AlphaTestMode::HashedAlphaIsotropic, "Hashed Alpha Isotropic" },
    { (uint32_t)AlphaTestMode::HashedAlphaAnisotropic, "Hashed Alpha Anisotropic" },
    { (uint32_t)AlphaTestMode::AlphaTest, "Alpha Test" } 
};

void HashedAlpha::onGuiRender()
{
    if (mpGui->addButton("Load Model"))
    {
        loadModel();
    }

    mpGui->addSeparator();

    uint32_t mode = (uint32_t)mAlphaTestMode;
    if (mpGui->addDropdown("Mode", kModeList, mode))
    {
        mAlphaTestMode = (AlphaTestMode)mode;
        mDirty = true;
    }

    mpGui->addFloatVar("Hash Scale", mHashScale, 0.01f, 10.0f, 0.01f);
    if (mpGui->addButton("Apply Scale"))
    {
        mDirty = true;
    }
}

void HashedAlpha::loadModel()
{
    std::string filename;
    if (openFileDialog(Model::kSupportedFileFormatsStr, filename))
    {
        mpModel = Model::createFromFile(filename.c_str());

        if (mpModel == nullptr)
        {
            msgBox("Could not load model");
            return;
        }

        // update the camera position
        float radius = mpModel->getRadius();
        const glm::vec3& modelCenter = mpModel->getCenter();
        glm::vec3 camPos = modelCenter;
        camPos.z += radius * 5.0f;

        mpCamera->setPosition(camPos);
        mpCamera->setTarget(modelCenter);
        mpCamera->setUpVector(glm::vec3(0, 1, 0));
        mpCamera->setDepthRange(std::max(0.01f, radius / 750.0f), radius * 50.0f);

        mCameraController.setModelParams(modelCenter, radius, 3.5f);
    }
}

void HashedAlpha::updateProgram()
{
    if (mDirty)
    {
        mpProgram->clearDefines();

        switch (mAlphaTestMode)
        {
        case AlphaTestMode::AlphaTest:
            mpProgram->addDefine("_DEFAULT_ALPHA_TEST");
            break;
        case AlphaTestMode::HashedAlphaAnisotropic:
            mpProgram->addDefine("_HASHED_ALPHA_TEST_ANISOTROPIC");
            break;
        }

        mpProgram->addDefine("_HASHED_ALPHA_SCALE", std::to_string(mHashScale));
        mDirty = false;
    }
}

void HashedAlpha::onLoad()
{
    mpProgram = GraphicsProgram::createFromFile("", appendShaderExtension("HashedAlpha.ps"));
    mpVars = GraphicsVars::create(mpProgram->getActiveVersion()->getReflector());
    updateProgram();

    mpState = GraphicsState::create();
    mpState->setProgram(mpProgram);

    mpCamera = Camera::create();
    mpCamera->setAspectRatio((float)mpDefaultFBO->getWidth() / (float)mpDefaultFBO->getHeight());
    mCameraController.attachCamera(mpCamera);
}

void HashedAlpha::onFrameRender()
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    mpRenderContext->clearFbo(mpDefaultFBO.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    mpState->setFbo(mpDefaultFBO);
    mCameraController.update();

    if (mpModel)
    {
        updateProgram();
        mpRenderContext->setGraphicsState(mpState);
        mpRenderContext->setGraphicsVars(mpVars);
        ModelRenderer::render(mpRenderContext.get(), mpModel, mpCamera.get());
    }
}

void HashedAlpha::onShutdown()
{

}

bool HashedAlpha::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return mCameraController.onKeyEvent(keyEvent);
}

bool HashedAlpha::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mCameraController.onMouseEvent(mouseEvent);
}

void HashedAlpha::onDataReload()
{

}

void HashedAlpha::onResizeSwapChain()
{
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    HashedAlpha sample;
    SampleConfig config;
    config.windowDesc.title = "Hashed Alpha Test";
    config.windowDesc.resizableWindow = true;
#ifdef _WIN32
    sample.run(config);
#else
    sample.run(config, (uint32_t)argc, argv);
#endif
    return 0;
}

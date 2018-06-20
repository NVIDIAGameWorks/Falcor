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
#include "MultiPassPostProcess.h"

void MultiPassPostProcess::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (pGui->addButton("Load Image"))
    {
        loadImage(pSample);
    }
    pGui->addCheckBox("Gaussian Blur", mEnableGaussianBlur);
    if(mEnableGaussianBlur)
    {
        mpGaussianBlur->renderUI(pGui, "Blur Settings");
        pGui->addCheckBox("Grayscale", mEnableGrayscale);
    }
}

void MultiPassPostProcess::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pContext)
{
    mpLuminance = FullScreenPass::create("Luminance.ps.hlsl");
    mpGaussianBlur = GaussianBlur::create(5);
    mpBlit = FullScreenPass::create("Blit.ps.hlsl");
    mpProgVars = GraphicsVars::create(mpBlit->getProgram()->getReflector());
}

void MultiPassPostProcess::loadImage(SampleCallbacks* pSample)
{
    std::string filename;
    if(openFileDialog("Supported Formats\0*.jpg;*.bmp;*.dds;*.png;*.tiff;*.tif;*.tga\0\0", filename))
    {
        loadImageFromFile(pSample, filename);
    }
}

void MultiPassPostProcess::loadImageFromFile(SampleCallbacks* pSample, std::string filename)
{
    auto fboFormat = pSample->getCurrentFbo()->getColorTexture(0)->getFormat();
    mpImage = createTextureFromFile(filename, false, isSrgbFormat(fboFormat));

    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, mpImage->getFormat());
    mpTempFB = FboHelper::create2D(mpImage->getWidth(), mpImage->getHeight(), fboDesc);

    pSample->resizeSwapChain(mpImage->getWidth(), mpImage->getHeight());
}

void MultiPassPostProcess::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pContext->clearFbo(pTargetFbo.get(), clearColor, 0, 0, FboAttachmentType::Color);

    if(mpImage)
    {
        // Grayscale is only with radial blur
        mEnableGrayscale = mEnableGaussianBlur && mEnableGrayscale;

        pContext->setGraphicsVars(mpProgVars);

        if(mEnableGaussianBlur)
        {
            mpGaussianBlur->execute(pContext.get(), mpImage, mpTempFB);
            mpProgVars->setTexture("gTexture", mpTempFB->getColorTexture(0));
            const FullScreenPass* pFinalPass = mEnableGrayscale ? mpLuminance.get() : mpBlit.get();
            pFinalPass->execute(pContext.get());
        }
        else
        {
            mpProgVars->setTexture("gTexture", mpImage);
            mpBlit->execute(pContext.get());
        }
    }
}

void MultiPassPostProcess::onShutdown(SampleCallbacks* pSample)
{
}

bool MultiPassPostProcess::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        switch (keyEvent.key)
        {
        case KeyboardEvent::Key::L:
            loadImage(pSample);
            return true;
        case KeyboardEvent::Key::G:
            mEnableGrayscale = true;
            return true;
        case KeyboardEvent::Key::B:
            mEnableGaussianBlur = true;
            return true;
        }
    }
    return false;
}

 void MultiPassPostProcess::onInitializeTesting(SampleCallbacks* pSample)
 {
     auto argList = pSample->getArgList();
     std::vector<ArgList::Arg> filenames = argList.getValues("loadimage");
     if (!filenames.empty())
     {
         loadImageFromFile(pSample, filenames[0].asString());
     }
 
     if (argList.argExists("gaussianblur"))
     {
         mEnableGaussianBlur = true;
         if (argList.argExists("grayscale"))
         {
             mEnableGrayscale = true;
         }
     }
 }

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    MultiPassPostProcess::UniquePtr pRenderer = std::make_unique<MultiPassPostProcess>();

    SampleConfig config;
    config.windowDesc.title = "Multi-pass post-processing";
#ifdef _WIN32
    Sample::run(config, pRenderer);
#else
    config.argc = (uint32_t)argc;
    config.argv = argv;
    Sample::run(config, pRenderer);
#endif
    return 0;
}

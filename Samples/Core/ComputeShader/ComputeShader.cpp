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
#include "ComputeShader.h"

void ComputeShader::onGuiRender()
{
    if (mpGui->addButton("Load Image"))
    {
        loadImage();
    }
    mpGui->addCheckBox("Pixelate", mbPixelate);
}

Texture::SharedPtr createTmpTex(const Fbo* pFbo)
{
    return Texture::create2D(pFbo->getWidth(), pFbo->getHeight(), ResourceFormat::RGBA8Unorm, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
}

void ComputeShader::onLoad()
{
    mpProg = ComputeProgram::createFromFile(appendShaderExtension("compute"));
    mpState = ComputeState::create();
    mpState->setProgram(mpProg);
    mpProgVars = ComputeVars::create(mpProg->getActiveVersion()->getReflector());

    mpTmpTexture = createTmpTex(mpDefaultFBO.get());

    initializeTesting();
}

void ComputeShader::loadImage()
{
    std::string filename;
    if(openFileDialog("Supported Formats\0*.jpg;*.bmp;*.dds;*.png;*.tiff;*.tif;*.tga\0\0", filename))
    {
        loadImageFromFile(filename);
    }
}

void ComputeShader::loadImageFromFile(std::string filename)
{
    mpImage = createTextureFromFile(filename, false, true);

    resizeSwapChain(mpImage->getWidth(), mpImage->getHeight());
    mpProgVars->setTexture("gInput", mpImage);
    mpTmpTexture = createTmpTex(mpDefaultFBO.get());
}

void ComputeShader::onFrameRender()
{
    beginTestFrame();

	const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);

    if(mpImage)
    {
        mpRenderContext->clearUAV(mpTmpTexture->getUAV().get(), clearColor);

        if (mbPixelate)
        {
            mpProg->addDefine("_PIXELATE");
        }
        else
        {
            mpProg->removeDefine("_PIXELATE");
        }
        mpProgVars->setTexture("gOutput", mpTmpTexture);

        mpRenderContext->setComputeState(mpState);
        mpRenderContext->setComputeVars(mpProgVars);

        uint32_t w = (mpImage->getWidth() / 16) + 1;
        uint32_t h = (mpImage->getHeight() / 16) + 1;
        mpRenderContext->dispatch(w, h, 1);
        mpRenderContext->copyResource(mpDefaultFBO->getColorTexture(0).get(), mpTmpTexture.get());
    }
    else
    {
        mpRenderContext->clearRtv(mpDefaultFBO->getRenderTargetView(0).get(), clearColor);
    }

    endTestFrame();
}

void ComputeShader::onInitializeTesting()
{
    std::vector<ArgList::Arg> filenames = mArgList.getValues("loadimage");
    if (!filenames.empty())
    {
        loadImageFromFile(filenames[0].asString());
    }

    if (mArgList.argExists("pixelate"))
    {
        mbPixelate = true;
    }
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    ComputeShader sample;
    SampleConfig config;
    config.windowDesc.title = "Compute Shader";
    config.windowDesc.resizableWindow = true;
    config.deviceDesc.depthFormat = ResourceFormat::Unknown;
#ifdef _WIN32
    sample.run(config);
#else
    sample.run(config, (uint32_t)argc, argv);
#endif
    return 0;
}

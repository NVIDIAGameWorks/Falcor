/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "FilmGrainSample.h"

using namespace Falcor;

const char* FilmGrainSample::kDefaultImageName = "SunTemple/SunTemple_Reflection.hdr";

void FilmGrainSample::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mpFilmGrainPass = FilmGrain::create(1.0f);

    mpGraphicsState = GraphicsState::create();
    loadImage();
}

void FilmGrainSample::loadImage(const std::string& name)
{
    mpImage = createTextureFromFile(name, false, true, Resource::BindFlags::ShaderResource);
}

void FilmGrainSample::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    std::string name;
    if (pGui->addButton("Open Image"))
    {
        if (openFileDialog("", name))
        {
            loadImage(name);
        }
    }
    pGui->addCheckBox("Enable Film Grain", mEnableFilmGrain);
    if (mEnableFilmGrain)
    {
        mpFilmGrainPass->renderUI(pGui, "Film Grain");
    }
}

void FilmGrainSample::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    mpGraphicsState->pushFbo(pTargetFbo);

    pRenderContext->blit(mpImage->getSRV(), pTargetFbo->getColorTexture(0)->getRTV());

    //Render teapot to hdr fbo
    pRenderContext->setGraphicsState(mpGraphicsState);
    if (mEnableFilmGrain)
    {
        mpFilmGrainPass->execute(pRenderContext.get(), pTargetFbo);
    }
    mpGraphicsState->popFbo();
}

void FilmGrainSample::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    //recreate hdr fbo
    ResourceFormat format = ResourceFormat::RGBA32Float;
    Fbo::Desc desc;
    desc.setDepthStencilTarget(ResourceFormat::D16Unorm);
    desc.setColorTarget(0u, format);

    mpFbo = FboHelper::create2D(width, height, desc);
}

bool FilmGrainSample::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    return false;
}

bool FilmGrainSample::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return false;
}

void FilmGrainSample::onInitializeTesting(SampleCallbacks* pSample)
{
}

void FilmGrainSample::onEndTestFrame(SampleCallbacks* pSample, SampleTest* pSampleTest)
{

}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    FilmGrainSample::UniquePtr pRenderer = std::make_unique<FilmGrainSample>();
    SampleConfig config;
    config.windowDesc.title = "Film Grain Sample";
#ifdef _WIN32
    Sample::run(config, pRenderer);
#else
    config.argc = (uint32_t)argc;
    config.argv = argv;
    Sample::run(config, pRenderer);
#endif
    return 0;
}

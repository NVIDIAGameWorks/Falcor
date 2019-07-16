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
#include "PathTracer.h"
#include "RenderPasses/GGXGlobalIllumination.h"
#include "RenderPasses/GBufferRaster.h"
#include "RenderPasses/TemporalAccumulation.h"

void PathTracer::onGuiRender(SampleCallbacks* pCallbacks, Gui* pGui)
{
    if (pGui->addButton("Load Scene"))
    {
        assert(mpGraph != nullptr);
        std::string filename;
        if (openFileDialog(Scene::kFileExtensionFilters, filename))
        {
            ProgressBar::SharedPtr pBar = ProgressBar::create("Loading Scene", 100);

            RtScene::SharedPtr pScene = RtScene::loadFromFile(filename);
            if (pScene != nullptr)
            {
                Fbo::SharedPtr pFbo = pCallbacks->getCurrentFbo();
                pScene->setCamerasAspectRatio(float(pFbo->getWidth()) / float(pFbo->getHeight()));
                mpGraph->setScene(pScene);
            }
        }
    }

    pGui->addSeparator();

    if (pGui->addButton(mDisableCameraPath ? "Enable Camera Path" : "Disable Camera Path"))
    {
        toggleCameraPathState();
    }

    if (mpGraph != nullptr)
    {
        mpGraph->renderUI(pGui, nullptr);
    }
}

void PathTracer::toggleCameraPathState()
{
    Scene::SharedPtr pScene = mpGraph->getScene();
    if (pScene != nullptr && pScene->getPathCount() > 0)
    {
        mDisableCameraPath = !mDisableCameraPath;
        if (mDisableCameraPath)
        {
            pScene->getPath(0)->detachObject(pScene->getActiveCamera());
        }
        else
        {
            pScene->getPath(0)->attachObject(pScene->getActiveCamera());
        }
    }
}

void PathTracer::onLoad(SampleCallbacks* pCallbacks, RenderContext* pRenderContext)
{
    mpGraph = RenderGraph::create("Path Tracer");
    mpGraph->addPass(GBufferRaster::create(), "GBuffer");
    auto pGIPass = GGXGlobalIllumination::create();
    mpGraph->addPass(pGIPass, "GlobalIllumination");
    mpGraph->addPass(TemporalAccumulation::create(), "TemporalAccumulation");
    mpGraph->addPass(ToneMapping::create(), "ToneMapping");

    mpGraph->addEdge("GBuffer.posW", "GlobalIllumination.posW");
    mpGraph->addEdge("GBuffer.normW", "GlobalIllumination.normW");
    mpGraph->addEdge("GBuffer.diffuseOpacity", "GlobalIllumination.diffuseOpacity");
    mpGraph->addEdge("GBuffer.specRough", "GlobalIllumination.specRough");
    mpGraph->addEdge("GBuffer.emissive", "GlobalIllumination.emissive");
    mpGraph->addEdge("GBuffer.matlExtra", "GlobalIllumination.matlExtra");

    mpGraph->addEdge("GlobalIllumination.output", "TemporalAccumulation.input");

    mpGraph->addEdge("TemporalAccumulation.output", "ToneMapping.src");

    mpGraph->markOutput("ToneMapping.dst");

    // When GI pass changes, tell temporal accumulation to reset
    pGIPass->setPassChangedCB([this]() {(*mpGraph->getPassesDictionary())["_dirty"] = true; });

    // Initialize the graph's record of what the swapchain size is, for texture creation
    mpGraph->onResize(pCallbacks->getCurrentFbo().get());

    {
        ProgressBar::SharedPtr pBar = ProgressBar::create("Loading Scene", 100);

        RtScene::SharedPtr pScene = RtScene::loadFromFile("Arcade/Arcade.fscene");
        if (pScene != nullptr)
        {
            Fbo::SharedPtr pFbo = pCallbacks->getCurrentFbo();
            pScene->setCamerasAspectRatio(float(pFbo->getWidth()) / float(pFbo->getHeight()));
            mpGraph->setScene(pScene);
        }
    }
}

void PathTracer::onFrameRender(SampleCallbacks* pCallbacks, RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if (mpGraph->getScene() != nullptr)
    {
        mpGraph->getScene()->update(pCallbacks->getCurrentTime(), &mCamController);
        mpGraph->execute(pRenderContext);

        pRenderContext->blit(mpGraph->getOutput("ToneMapping.dst")->getSRV(), pTargetFbo->getRenderTargetView(0));
    }
}

void PathTracer::onShutdown(SampleCallbacks* pCallbacks)
{
}

bool PathTracer::onKeyEvent(SampleCallbacks* pCallbacks, const KeyboardEvent& keyEvent)
{
    if (keyEvent.key == KeyboardEvent::Key::Minus && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        toggleCameraPathState();
        return true;
    }

    bool handled = false;
    if (mpGraph->getScene() != nullptr) handled = mpGraph->onKeyEvent(keyEvent);
    return handled ? true : mCamController.onKeyEvent(keyEvent);
}

bool PathTracer::onMouseEvent(SampleCallbacks* pCallbacks, const MouseEvent& mouseEvent)
{
    bool handled = false;
    if (mpGraph->getScene() != nullptr) handled = mpGraph->onMouseEvent(mouseEvent);
    return handled ? true : mCamController.onMouseEvent(mouseEvent);
}

void PathTracer::onDataReload(SampleCallbacks* pCallbacks)
{

}

void PathTracer::onResizeSwapChain(SampleCallbacks* pCallbacks, uint32_t width, uint32_t height)
{
    if (mpGraph)
    {
        mpGraph->onResize(pCallbacks->getCurrentFbo().get());
        if(mpGraph->getScene() != nullptr) mpGraph->getScene()->setCamerasAspectRatio((float)width / (float)height);
    }
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    PathTracer::UniquePtr pRenderer = std::make_unique<PathTracer>();
    SampleConfig config;
    config.deviceDesc.enableRaytracing = true;
    config.windowDesc.title = "Path Tracer";
    config.windowDesc.resizableWindow = true;
    config.freezeTimeOnStartup = true;
    Sample::run(config, pRenderer);
    return 0;
}

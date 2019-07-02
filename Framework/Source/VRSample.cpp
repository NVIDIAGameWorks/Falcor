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
#include "Framework.h"
#include "VRSample.h"
#include <map>
#include <fstream>
#include "API/Window.h"
#include "Graphics/Program/Program.h"
#include "Utils/Platform/OS.h"
#include "API/FBO.h"
#include "VR/OpenVR/VRSystem.h"
#include "Utils/Platform/ProgressBar.h"
#include "Utils/StringUtils.h"
#include "Graphics/FboHelper.h"
#include <sstream>
#include <iomanip>
#include "Experimental/RenderGraph/RenderPassLibrary.h"

namespace Falcor
{    

    void VRSample::run(const SampleConfig& config, VRRenderer::UniquePtr& pRenderer)
    {
        VRSample s(pRenderer);
        s.runInternal(config, config.argc, config.argv);
    }

    void VRSample::renderFrame()
    {
        if (gpDevice && gpDevice->isWindowOccluded())
        {
            return;
        }

        mFrameRate.newFrame();
        {
            PROFILE("onFrameRender");
            calculateTime();
            // The swap-chain FBO might have changed between frames, so get it
            if (!mFreezeRendering)
            {
                RenderContext* pRenderContext = nullptr;
                if (gpDevice)
                {
                    // Bind the default state
                    pRenderContext = gpDevice->getRenderContext();
                    mpDefaultPipelineState->setFbo(mpTargetFBO);
                    pRenderContext->setGraphicsState(mpDefaultPipelineState);
                }
                mpRenderer->onFrameRender(this, pRenderContext, mpTargetFBO);
            }
        }
        
        if (gpDevice)
        {
            // Copy the render-target
            const auto& pSwapChainFbo = gpDevice->getSwapChainFbo();
            RenderContext* pCtx = getRenderContext();
            getRenderContext()->copyResource(pSwapChainFbo->getColorTexture(0).get(), mpTargetFBO->getColorTexture(0).get());

            if (mTestingFrames.size())  onTestFrame();

            // Capture video frame before UI is rendered
            bool captureVideoUI = mVideoCapture.pUI && mVideoCapture.pUI->captureUI();  // Check capture mode here once only, as its value may change after renderGUI()
            if (!captureVideoUI)
            {
                captureVideoFrame();
            }

            //Swaps back to backbuffer to render fps text and gui directly onto it
            mpDefaultPipelineState->setFbo(pSwapChainFbo);
            pCtx->setGraphicsState(mpDefaultPipelineState);
            {
                PROFILE("renderGUI");
                renderGUI();
            }

            renderText(getFpsMsg(), glm::vec2(10, 10));
            if (mpPixelZoom)
            {
                mpPixelZoom->render(pCtx, pSwapChainFbo.get());
            }

#if _PROFILING_ENABLED
            Profiler::endFrame();
#endif
            // Capture video frame after UI is rendered
            if (captureVideoUI)
            {
                captureVideoFrame();
            }

            if (mCaptureScreen)
            {
                captureScreen();
            }

            {
                PROFILE("present");
                gpDevice->present();
                VRRenderer* vrRenderer = static_cast<VRRenderer*>(mpRenderer.get());
                vrRenderer->onFrameSubmit(this, getRenderContext(), mpTargetFBO);
            }
        }
    }
}

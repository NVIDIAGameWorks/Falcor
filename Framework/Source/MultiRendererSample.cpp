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

#include "MultiRendererSample.h"
#include "Externals/dear_imgui/imgui.h"

namespace Falcor
{
    // You may want to override this function in derived classes for a more complex scene loader
    Scene::SharedPtr MultiRendererSample::loadScene(const std::string& filename)
    {
        return Scene::loadFromFile(filename);
    }

    uint32_t MultiRendererSample::addRenderer(Renderer::SharedPtr pRenderer)
    { 
        size_t id = mRenderer.size(); 
        mRenderer.push_back(pRenderer);
        return uint32_t(id); 
    }

    void MultiRendererSample::switchRenderers(uint32_t fromId, uint32_t toId)
    {
        // Sanity checking; avoid doing a bunch of work if we're not actually switching
        if (fromId == toId)
        {
            return;
        }

        // Tell the current renderer that we're moving on to someone else
        if (mRenderer[fromId])
        {
            mRenderer[fromId]->onSwitchFrom();
        }

        // Switch renderers
        mCurRenderer = toId;
        mSelectedRenderer = toId;  // Seems redundant, but needed to update GUI in all cases

         // For the new renderer
        if (mRenderer[mCurRenderer])
        {
            // Tell it the current window size
            mRenderer[mCurRenderer]->onResizeSwapChain(mpDefaultFBO->getWidth(), mpDefaultFBO->getHeight());

            // Let it know that it's the current renderer
            mRenderer[mCurRenderer]->onSwitchTo();
        }
    }

    void MultiRendererSample::onLoad()
    {
        // Get a graphics state to control the DX pipe
        mpState = GraphicsState::create();

        // Initialize our renderers
        for (size_t i = 0; i < mRenderer.size(); i++)
        {
            mRenderer[i]->initializeSharedState(mpGui, mpState);
            mRenderer[i]->onInitialize(mpRenderContext);
            mRendererList.push_back({ int32_t(i), mRenderer[i]->getRendererName() });
        }

        // Call onSwitchTo() for the initial/default renderer
        if (mRenderer[mCurRenderer])
        {
            mRenderer[mCurRenderer]->onSwitchTo();
        }
    }

    void MultiRendererSample::onFrameRender()
    {
        // Push state to ensure we can get back to a consistent state after our renderer runs
        mpRenderContext->pushGraphicsState(mpState);

        // Do render callback.  
        if (mRenderer[mCurRenderer])
        {
            // Send the current time to the renderer
            mRenderer[mCurRenderer]->setCurrentTime(mCurrentTime);

            // Go ahead and draw
            mRenderer[mCurRenderer]->onFrameRender(mpRenderContext, mpDefaultFBO);
        }

        // We're done rendering.  Pop state so we're back to a guaranteed state.
        mpRenderContext->popGraphicsState();
    }
    
    void MultiRendererSample::onShutdown()
    {
        // As part of shutdown, we're switching away from the current renderer.
        if (mRenderer[mCurRenderer])
        {
            mRenderer[mCurRenderer]->onSwitchFrom();
        }

        // Shutdown our renderers
        for (size_t i = 0; i < mRenderer.size(); i++)
        {
            mRenderer[i]->onShutdown();
        }
    }

    bool MultiRendererSample::onKeyEvent(const KeyboardEvent& keyEvent)
    {
        // Handle some global key controls
        if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
        {
            // Tab increments the renderer, shift-tab decrements the renderer.
            if (keyEvent.key == KeyboardEvent::Key::Tab)
            {
                int32_t direction = keyEvent.mods.isShiftDown ? -1 : 1;
                uint32_t newRender = uint32_t(mRenderer.size() + mCurRenderer + direction) % mRenderer.size();
                switchRenderers(mCurRenderer, newRender);
            }
        }

        // Handle any renderer-specific events
        if (mRenderer[mCurRenderer] && mRenderer[mCurRenderer]->onKeyEvent(keyEvent))
        {
            return true;
        }

        // Handle any scene-specific events
        return mpSceneRenderer ? mpSceneRenderer->onKeyEvent(keyEvent) : false;
    }

    bool MultiRendererSample::onMouseEvent(const MouseEvent& mouseEvent)
    {
        if (mRenderer[mCurRenderer] && mRenderer[mCurRenderer]->onMouseEvent(mouseEvent))
        {
            return true;
        }
        return mpSceneRenderer ? mpSceneRenderer->onMouseEvent(mouseEvent) : true;
    }

    void MultiRendererSample::onResizeSwapChain()
    {
        // Get our current size
        uint32_t w = mpDefaultFBO->getWidth();
        uint32_t h = mpDefaultFBO->getHeight();

        // Let our scene (if any) know the correct aspect ratio.  Make sure this goes before the
        //     renderer's onResizeSwapChain(), so it can redo the scene's aspect ratio, if desired.
        if (mpSceneRenderer)
        {
            mpSceneRenderer->getScene()->getActiveCamera()->setAspectRatio((float)w / (float)h);
        }

        // Resize any resources used by the current rendering mode.  
        //    -> NOTE: Resources in other rendering modes are potentially resized before switchTo() is called
        if (mRenderer[mCurRenderer])
        {
            mRenderer[mCurRenderer]->onResizeSwapChain(w, h);
        }
    }

    void MultiRendererSample::onGuiRender()
    {
        // Create a common button to load a scene
        mpGui->addSeparator();
        if ((mRenderer.size() > size_t(1)) && 
            mpGui->addDropdown("Renderer", mRendererList, mSelectedRenderer))
        {
            // Check:  Did the user change which is the currently selected render mode?
            if (mSelectedRenderer != mCurRenderer)
            {
                switchRenderers(mCurRenderer, mSelectedRenderer);
            }
        }
        if (mpGui->addButton("Load Scene"))
        {
            std::string filename;
            if (openFileDialog(Scene::kFileFormatString, filename))
            {
                internalSceneLoader(filename);
            }
        }
        mpGui->addSeparator();

        // Let our current rendering mode draw its controls
        if (mRenderer[mCurRenderer])
        {
            bool useSeparateWindow = mRenderer[mCurRenderer]->useUIWindow();
            ivec2 winPos = mRenderer[mCurRenderer]->getGuiWindowLocation();
            ivec2 winSize = mRenderer[mCurRenderer]->getGuiWindowSize();

            // If we're asking for a separate UI widget window for each rendering mode, do it.
            if ((mRenderer.size() > size_t(1)) && useSeparateWindow)
            {
                // Could use mpGui->pushWindow() if we could pass custom flags to the GUI call
                customGuiWindow(mRenderer[mCurRenderer]->getGuiGroupName().c_str(),
                    winSize.x, winSize.y, winPos.x, winPos.y);
                mRenderer[mCurRenderer]->onGuiRender();
                mpGui->popWindow();
            }
            else
            {
                // Adds some space between global and renderer-specific UI in a combined window
                mpGui->addText("");  

                // Draw the renderer's UI in the single global widget window
                mRenderer[mCurRenderer]->onGuiRender();
            }
        }
    }

    // A workaround to stop a renderer-specific window from stealing focus.  (Without this, you can
    //    use Tab to switch between renderers once, but then Tab starts walking through UI controls.)
    void MultiRendererSample::customGuiWindow(const char label[], uint32_t width, uint32_t height, uint32_t x, uint32_t y)
    {
        ImVec2 pos{ float(x), float(y) };
        ImVec2 size{ float(width), float(height) };
        ImGui::SetNextWindowSize(size, ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowPos(pos, ImGuiSetCond_FirstUseEver);
        ImGui::Begin(label, nullptr, ImGuiWindowFlags_NoFocusOnAppearing);
    }

    void MultiRendererSample::internalSceneLoader(const std::string& filename)
    {
        // Create a loading bar while loading a scene
        ProgressBar::SharedPtr pBar = ProgressBar::create("Loading Scene", 100);

        // Call our overloadable method to actually do scene loading
        Scene::SharedPtr pScene = loadScene(filename);

        // If it was sucessful, tell our renderers about the new scene.
        if (pScene != nullptr)
        {
            // Create a scene renderer with first-person shooter camera and aspect ration as per the FBO size.
            mpSceneRenderer = SceneRenderer::create(pScene);
            mpSceneRenderer->setCameraControllerType(SceneRenderer::CameraControllerType::FirstPerson);
            mpSceneRenderer->getScene()->getActiveCamera()->setAspectRatio((float)mpDefaultFBO->getWidth() / (float)mpDefaultFBO->getHeight());

            // Let all the renderers know we loaded a new scene
            for (size_t i = 0; i < mRenderer.size(); i++)
            {
                mRenderer[i]->onInitNewScene(mpSceneRenderer);
            }
        }
    }

} // End namespace Falcor

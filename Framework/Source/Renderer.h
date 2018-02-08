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
#pragma once

namespace Falcor
{
    class Gui;
    class Window;
    class RenderContext;
    class Fbo;

    class SampleCallbacks
    {
    public:
        /** Get the render-context for the current frame. This might change each frame*/
        virtual std::shared_ptr<RenderContext> getRenderContext() = 0;

        /** Get the current FBO*/
        virtual std::shared_ptr<Fbo> getCurrentFbo() = 0;

        /** Get the default GUI*/
        virtual Gui* getGui() = 0;

        /** Get the current time*/
        virtual float getCurrentTime() = 0;

        /** Get the time that passed since the last frame*/
        virtual float getTimeSinceLastFrame() = 0;

        /** Set the current time*/
        virtual void setCurrentTime(float time) = 0;

        /** Resize the swap-chain buffers*/
        virtual void resizeSwapChain(uint32_t width, uint32_t height) = 0;

        /** Get the average framerate */
        virtual float getFrameRate() = 0;

        /** Render text */
        virtual void renderText(const std::string& str, const glm::vec2& position, glm::vec2 shadowOffset = glm::vec2(1)) = 0;

        /** Get a string with the framerate information */
        virtual std::string getFpsMsg() = 0;

        /** Check if a key is pressed*/
        virtual bool isKeyPressed(const KeyboardEvent::Key& key) = 0;

        /** Get the window object */
        virtual Window* getWindow() = 0;

        /** Show/hide the UI */
        virtual void toggleUI(bool showUI) = 0;

        /** Set the default GUI size */
        virtual void setDefaultGuiSize(uint32_t width, uint32_t height) = 0;
    };

    class Renderer : std::enable_shared_from_this<Renderer>
    {
    public:
        using UniquePtr = std::unique_ptr<Renderer>;
        Renderer() = default;
        virtual ~Renderer() {};

        /** Called once right after context creation.
        */
        virtual void onLoad(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext) {}

        /** Called on each frame render.
        */
        virtual void onFrameRender(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext, Fbo::SharedPtr pCurrentFbo) {}

        /** Called right before the context is destroyed.
        */
        virtual void onShutdown(SampleCallbacks* pSample) {}

        /** Called every time the swap-chain is resized. You can query the default FBO for the new size and sample count of the window.
        */
        virtual void onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height) {}

        /** Called every time the user requests shader recompilation (by pressing F5)
        */
        virtual void onDataReload(SampleCallbacks* pSample) {}

        /** Called every time a key event occurred.
        \param[in] keyEvent The keyboard event
        \return true if the event was consumed by the callback, otherwise false
        */
        virtual bool onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent) { return false; }

        /** Called every time a mouse event occurred.
        \param[in] mouseEvent The mouse event
        \return true if the event was consumed by the callback, otherwise false
        */
        virtual bool onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent) { return false; }

        /** Called after onFrameRender().
        It is highly recommended to use onGuiRender() exclusively for GUI handling. onGuiRender() will not be called when the GUI is hidden, which should help reduce CPU overhead.
        You could also ignore this and render the GUI directly in your onFrameRender() function, but that is discouraged.
        */
        virtual void onGuiRender(SampleCallbacks* pSample, Gui* pGui) {};

        /** Called when a file is dropped into the window
        */
        virtual void onDroppedFile(SampleCallbacks* pSample, const std::string& filename) {}

        // Deleted copy operators (copy a pointer type!)
        Renderer(const Renderer&) = delete;
        Renderer& operator=(const Renderer &) = delete;
    };
}
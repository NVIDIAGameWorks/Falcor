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
#include "Utils/UserInput.h"
#include <glm/vec2.hpp>

struct GLFWwindow;

namespace Falcor
{
    class Window
    {
    public:
        using SharedPtr = std::shared_ptr<Window>;
        using SharedConstPtr = std::shared_ptr<const Window>;
        using ApiHandle = WindowHandle;

        /** Window configuration configuration
        */
        struct Desc
        {
            uint32_t width = 1920;                  ///< The width of the client area size
            uint32_t height = 1080;                 ///< The height of the client area size
            bool fullScreen = false;                ///< Set to true to run the sample in full-screen mode
            std::string title = "Falcor Sample";    ///< Window title
            bool resizableWindow = true;            ///< Allow the user to resize the window.
        };

        /** Callbacks interface to be used when creating a new object
        */
        class ICallbacks
        {
        public:
            virtual void handleWindowSizeChange() = 0;
            virtual void renderFrame() = 0;
            virtual void handleKeyboardEvent(const KeyboardEvent& keyEvent) = 0;
            virtual void handleMouseEvent(const MouseEvent& mouseEvent) = 0;
        };

        /** Create a new window.
            \param[in] desc Window configuration
            \param[in] pCallbacks User callbacks
            \return A new object if creation succeeded, otherwise nullptr
        */
        static SharedPtr create(const Desc& desc, ICallbacks* pCallbacks);

        /** Destructor
        */
        ~Window();

        /** Destroy the window. This will cause the msgLoop() to stop its execution
        */
        void shutdown();

        /** Resize the window
            \param[in] width The new width of the client-area
            \param[in] height The new height of the client-area
            There is not guarantee that the call will succeed. You should call getClientAreaHeight() and getClientAreaWidth() to get the actual new size of the window
        */
        void resize(uint32_t width, uint32_t height);

        /** Start executing the msgLoop. The only way to stop it is to call shutdown()
        */
        void msgLoop();

        /** Force event polling. Useful if your rendering loop is slow and you would like to get a recent keyboard/mouse status
        */
        void pollForEvents();

        /** Change the window's title
        */
        void setWindowTitle(const std::string& title);

        /** Get the native window handle
        */
        ApiHandle getApiHandle() const { return mApiHandle; }

        /** Get the width of the window's client area
        */
        uint32_t getClientAreaWidth() const { return mWidth; }

        /** Get the height of the window's client area
        */
        uint32_t getClientAreaHeight() const { return mHeight; }

    private:
        friend class ApiCallbacks;
        Window(ICallbacks * pCallbacks, uint32_t width, uint32_t height);

        GLFWwindow* mpGLFWWindow;
        ApiHandle mApiHandle;
        uint32_t mWidth;
        uint32_t mHeight;
        glm::vec2 mMouseScale;
        const glm::vec2& getMouseScale() const { return mMouseScale; }
        ICallbacks* mpCallbacks = nullptr;
    };
}
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
#include "glm/glm.hpp"
#include <set>
#include <string>
#include <stdint.h>
#include "API/Window.h"
#include "utils/FrameRate.h"
#include "utils/Gui.h"
#include "utils/TextRenderer.h"
#include "API/RenderContext.h"
#include "Utils/Video/VideoEncoderUI.h"
#include "API/Device.h"
#include "ArgList.h"
#include "Utils/PixelZoom.h"

namespace Falcor
{
#ifdef _DEBUG
#define _SHOW_MB_BY_DEFAULT true
#else
#define _SHOW_MB_BY_DEFAULT false
#endif

    /** Sample configuration.
    */
    struct SampleConfig
    {
        /** Flags to control different sample controls
        */
        enum class Flags
        {
            None              = 0x0,  ///< No flags 
            DoNotCreateDevice = 0x1,  ///< Do not create a device. Services that depends on the device - such as GUI text - will be disabled. Use this only if you are writing raw-API sample
        };

        Window::Desc windowDesc;                                    ///< Controls window creation
        Device::Desc deviceDesc;                                    ///< Controls device creation;
        bool showMessageBoxOnError = _SHOW_MB_BY_DEFAULT;           ///< Show message box on framework/API errors.
        float timeScale = 1.0f;                                     ///< A scaling factor for the time elapsed between frames.
        float fixedTimeDelta = 0.0f;                                ///< If non-zero, specifies a fixed simulation time step per frame, which is further affected by time scale.
        bool freezeTimeOnStartup = false;                           ///< Control whether or not to start the clock when the sample start running.
        std::function<void(void)> deviceCreatedCallback = nullptr;  ///< Callback function which will be called after the device is created
        Flags flags = Flags::None;                                  ///< Sample flags
    };

    /** Bootstrapper class for Falcor.
        User should create a class which inherits from Sample, then call Sample::run() to start the sample.
        The render loop will then call the user's overridden callback functions.
    */
    class Sample : public Window::ICallbacks
    {
    public:
        Sample();
        virtual ~Sample();
        Sample(const Sample&) = delete;
        Sample& operator=(const Sample&) = delete;

        /** Entry-point to Sample. User should call this to start processing.
            \param[in] config Requested sample configuration
        */
        virtual void run(const SampleConfig& config);

    protected:
        // Callbacks

        /** Called once right after context creation.
        */
        virtual void onLoad() {}

        /** Called on each frame render.
        */
        virtual void onFrameRender() {}

        /** Called right before the context is destroyed.
        */
        virtual void onShutdown() {}

        /** Called every time the swap-chain is resized. You can query the default FBO for the new size and sample count of the window.
        */
        virtual void onResizeSwapChain() {}

        /** Called every time the user requests shader recompilation (by pressing F5)
        */
        virtual void onDataReload() {}

        /** Called every time a key event occurred.
            \param[in] keyEvent The keyboard event
            \return true if the event was consumed by the callback, otherwise false
        */
        virtual bool onKeyEvent(const KeyboardEvent& keyEvent) { return false; }

        /** Called every time a mouse event occurred.
            \param[in] mouseEvent The mouse event
            \return true if the event was consumed by the callback, otherwise false
        */
        virtual bool onMouseEvent(const MouseEvent& mouseEvent) { return false; }

        /** Called after onFrameRender().
            It is highly recommended to use onGuiRender() exclusively for GUI handling. onGuiRender() will not be called when the GUI is hidden, which should help reduce CPU overhead.
            You could also ignore this and render the GUI directly in your onFrameRender() function, but that is discouraged.
        */
        virtual void onGuiRender() {};

        /** Resize the swap-chain buffers
            \param[in] width Requested width
            \param[in] height Requested height
        */
        void resizeSwapChain(uint32_t width, uint32_t height);

        /** Get whether the given key is pressed
            \param[in] key The key
        */
        bool isKeyPressed(const KeyboardEvent::Key& key) const;

        /** Get information about the framerate
        */
        const FrameRate& frameRate() const { return mFrameRate; }

        /** Render a text string.
            \param[in] str The string to render
            \param[in] position Window position of the string in pixels from the top-left corner
            \param[in] shadowOffset Offset for an outline shadow. Disabled if zero.
        */
        void renderText(const std::string& str, const glm::vec2& position, const glm::vec2 shadowOffset = glm::vec2(1.f, 1.f)) const;

        /** Get the FPS message string
        */
        const std::string getFpsMsg() const;

        /** Close the window and exit the application
        */
        void shutdownApp();

        /** Poll for window events (useful when running long pieces of code)
        */
        void pollForEvents();

        /** Change the title of the window
        */
        void setWindowTitle(const std::string& title);

        /** Show/hide the UI
        */
        void toggleUI(bool showUI) { mShowUI = showUI && gpDevice; }

        /** Set the main GUI window size
        */
        void setSampleGuiWindowSize(uint32_t width, uint32_t height);

        Gui::UniquePtr mpGui;                               ///< Main sample GUI
        RenderContext::SharedPtr mpRenderContext;           ///< The rendering context
        GraphicsState::SharedPtr mpDefaultPipelineState;    ///< The default pipeline state
        Fbo::SharedPtr mpDefaultFBO;                        ///< The default FBO object
        bool mFreezeTime;                                   ///< Whether global time is frozen
        float mCurrentTime = 0;                             ///< Global time
        float mTimeScale;                                   ///< Global time scale
        ArgList mArgList;                                   ///< Arguments passed in by command line
        Window::SharedPtr mpWindow;                         ///< The application's window

    protected:
        void renderFrame() override;
        void handleWindowSizeChange() override;
        void handleKeyboardEvent(const KeyboardEvent& keyEvent) override;
        void handleMouseEvent(const MouseEvent& mouseEvent) override;
        virtual float getTimeScale() final { return mTimeScale; }
        float getFixedTimeDelta() { return mFixedTimeDelta; }
        void setFixedTimeDelta(float newFixedTimeDelta) { mFixedTimeDelta = newFixedTimeDelta; }
        void initVideoCapture();

        std::string captureScreen(const std::string explicitFilename = "", const std::string explicitOutputDirectory = "");

        void toggleText(bool enabled);
        uint32_t getFrameID() const { return mFrameRate.getFrameCount(); }
    private:
        // Private functions
        void initUI();
        void printProfileData();
        void calculateTime();

        void startVideoCapture();
        void endVideoCapture();
        void captureVideoFrame();
        void renderGUI();

        bool mVsyncOn = false;
        bool mShowText = true;
        bool mShowUI = true;
        bool mCaptureScreen = false;

        struct VideoCaptureData
        {
            VideoEncoderUI::UniquePtr pUI;
            VideoEncoder::UniquePtr pVideoCapture;
            uint8_t* pFrame = nullptr;
            float sampleTimeDelta; // Saves the sample's fixed time delta because video capture overwrites it while recording
        };

        VideoCaptureData mVideoCapture;

        FrameRate mFrameRate;
        
        float mFixedTimeDelta;

        TextRenderer::UniquePtr mpTextRenderer;
        std::set<KeyboardEvent::Key> mPressedKeys;
        PixelZoom::SharedPtr mpPixelZoom;
        uint32_t mSampleGuiWidth = 250;
        uint32_t mSampleGuiHeight = 200;
    };
    enum_class_operators(SampleConfig::Flags);
};
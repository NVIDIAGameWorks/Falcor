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
#include "Utils/FrameRate.h"
#include "Utils/Gui.h"
#include "Utils/TextRenderer.h"
#include "API/RenderContext.h"
#include "Utils/Video/VideoEncoderUI.h"
#include "API/Device.h"
#include "ArgList.h"
#include "Utils/PixelZoom.h"
#include "Renderer.h"

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

    /** Bootstrapper class for Falcor
        Call Sample::run() to start the sample.
        The render loop will then call the user's Renderer object
    */
    class Sample : public Window::ICallbacks, public SampleCallbacks
    {
    public:
        /** Entry-point to Sample. User should call this to start processing.
            On Windows, command line args will be retrieved and parsed even if not passed through this function.
            On Linux, this function is the only way to feed the sample command line args.

            \param[in] config Requested sample configuration
            \param[in] pRenderer The user's renderer
            \param[in] argc Optional. Number of command line arguments
            \param[in] argv Optional. Array of command line arguments
        */
        static void run(const SampleConfig& config, Renderer::UniquePtr& pRenderer, uint32_t argc = 0, char** argv = nullptr);

        virtual ~Sample();
    protected:
        /************************************************************************/
        /* Callback inherited from SampleCallbacks                                 */
        /************************************************************************/
        RenderContext::SharedPtr getRenderContext() override { return mpRenderContext; }
        Fbo::SharedPtr getCurrentFbo() override { return mpDefaultFBO; }
        Gui* getGui() override { return mpGui.get(); }
        float getCurrentTime() override { return mCurrentTime; }
        void resizeSwapChain(uint32_t width, uint32_t height) override;
        bool isKeyPressed(const KeyboardEvent::Key& key) override;
        float getFrameRate() override { return mFrameRate.getAverageFrameTime(); }
        float getTimeSinceLastFrame() override { return mFrameRate.getLastFrameTime(); }
        void renderText(const std::string& str, const glm::vec2& position, glm::vec2 shadowOffset = vec2(1)) override;
        std::string getFpsMsg() override;
        Window* getWindow() override { return mpWindow.get(); }
        void toggleUI(bool showUI) { mShowUI = showUI && gpDevice; }
        void setDefaultGuiSize(uint32_t width, uint32_t height) override;
        void setCurrentTime(float time) override { mCurrentTime = time; }
        uint32_t getCurrentFrameId() override { return mFrameRate.getFrameCount(); }

        /** Internal data structures
        */
        Gui::UniquePtr mpGui;                               ///< Main sample GUI
        RenderContext::SharedPtr mpRenderContext;           ///< The rendering context
        GraphicsState::SharedPtr mpDefaultPipelineState;    ///< The default pipeline state
        Fbo::SharedPtr mpDefaultFBO;                        ///< The default FBO object
        bool mFreezeTime;                                   ///< Whether global time is frozen
        float mCurrentTime = 0;                             ///< Global time
        float mTimeScale;                                   ///< Global time scale
        ArgList mArgList;                                   ///< Arguments passed in by command line
        Window::SharedPtr mpWindow;                         ///< The application's window

        void renderFrame() override;
        void handleWindowSizeChange() override;
        void handleKeyboardEvent(const KeyboardEvent& keyEvent) override;
        void handleMouseEvent(const MouseEvent& mouseEvent) override;
        void handleDroppedFile(const std::string& filename) override;

        virtual float getTimeScale() final { return mTimeScale; }
        float getFixedTimeDelta() { return mFixedTimeDelta; }
        void setFixedTimeDelta(float newFixedTimeDelta) { mFixedTimeDelta = newFixedTimeDelta; }
        void initVideoCapture();

        std::string captureScreen(const std::string explicitFilename = "", const std::string explicitOutputDirectory = "");

        void toggleText(bool enabled);
        uint32_t getFrameID() const { return mFrameRate.getFrameCount(); }

        // Private functions
        void initUI();
        void printProfileData();
        void calculateTime();

        void startVideoCapture();
        void endVideoCapture();
        void captureVideoFrame();
        void renderGUI();

        void runInternal(const SampleConfig& config, uint32_t argc, char** argv);

        bool mVsyncOn = false;
        bool mShowText = true;
        bool mShowUI = true;
        bool mCaptureScreen = false;

        Renderer::UniquePtr mpRenderer;

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

        Sample(Renderer::UniquePtr& pRenderer) : mpRenderer(std::move(pRenderer)) {}
        Sample(const Sample&) = delete;
        Sample& operator=(const Sample&) = delete;
    };
    enum_class_operators(SampleConfig::Flags);
};
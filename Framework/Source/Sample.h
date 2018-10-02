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
#include "SampleTest.h"

namespace Falcor
{
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
        bool showMessageBoxOnError = true;                          ///< Show message box on framework/API errors.
        float timeScale = 1.0f;                                     ///< A scaling factor for the time elapsed between frames.
        float fixedTimeDelta = 0.0f;                                ///< If non-zero, specifies a fixed simulation time step per frame, which is further affected by time scale.
        bool freezeTimeOnStartup = false;                           ///< Control whether or not to start the clock when the sample start running.
        Flags flags = Flags::None;                                  ///< Sample flags
        uint32_t argc = 0;                                          ///< Arg count 
        char** argv = nullptr;                                      ///< Arg values
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
        static void run(const SampleConfig& config, Renderer::UniquePtr& pRenderer);

        virtual ~Sample();
    protected:
        /************************************************************************/
        /* Callback inherited from SampleCallbacks                                 */
        /************************************************************************/
        RenderContext::SharedPtr getRenderContext() override { return mpRenderContext; }
        Fbo::SharedPtr getCurrentFbo() override { return mpTargetFBO; }
        Window* getWindow() override { return mpWindow.get(); }
        Gui* getGui() override { return mpGui.get(); }
        float getCurrentTime() override { return mCurrentTime; }
        void resizeSwapChain(uint32_t width, uint32_t height) override;
        bool isKeyPressed(const KeyboardEvent::Key& key) override;
        float getFrameRate() override { return mFrameRate.getAverageFrameTime(); }
        float getLastFrameTime() override { return mFrameRate.getLastFrameTime(); }
        uint64_t getFrameID() override { return mFrameRate.getFrameCount(); }
        void renderText(const std::string& str, const glm::vec2& position, glm::vec2 shadowOffset = vec2(1)) override;
        std::string getFpsMsg() override;
        void toggleText(bool showText) override { mShowText = showText && gpDevice; }
        void toggleUI(bool showUI) override { if (!gpDevice || showUI) mShowUI = UIStatus::HideAll; else mShowUI = UIStatus::ShowAll; }
        void toggleGlobalUI(bool showGlobalUI) override { if (!gpDevice || !showGlobalUI) mShowUI = UIStatus::HideGlobal; else mShowUI = UIStatus::ShowAll; }
        void setDefaultGuiSize(uint32_t width, uint32_t height) override;
        void setDefaultGuiPosition(uint32_t x, uint32_t y) override;
        void setCurrentTime(float time) override { mCurrentTime = time; }
        ArgList getArgList() override { return mArgList; }
        void setFixedTimeDelta(float newFixedTimeDelta) override { mFixedTimeDelta = newFixedTimeDelta; }
        float getFixedTimeDelta() override  { return mFixedTimeDelta; }
        void freezeTime(bool timeFrozen) override { mFreezeTime = timeFrozen; }
        bool isTimeFrozen() override { return mFreezeTime; }
        std::string captureScreen(const std::string explicitFilename = "", const std::string explicitOutputDirectory = "") override;
        void shutdown() override { if (mpWindow) { mpWindow->shutdown(); } }
        
        //Any cleanup required by renderer if its being shut down early via testing 
        void onTestShutdown() override { mpRenderer->onTestShutdown(mpSampleTest.get()); }

        //Non inherited testing functions 
        bool initializeTesting();
        void beginTestFrame();
        void endTestFrame();

        /** Internal data structures
        */
        Gui::UniquePtr mpGui;                               ///< Main sample GUI
        RenderContext::SharedPtr mpRenderContext;           ///< The rendering context
        GraphicsState::SharedPtr mpDefaultPipelineState;    ///< The default pipeline 
        Fbo::SharedPtr mpTargetFBO;                         ///< The FBO available to renderers
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
        void initVideoCapture();

        // Private functions
        void initUI();
        void calculateTime();

        void startVideoCapture();
        void endVideoCapture();
        void captureVideoFrame();
        void renderGUI();

        void runInternal(const SampleConfig& config, uint32_t argc, char** argv);


        bool mVsyncOn = false;
        bool mShowText = true;
        enum class UIStatus
        {
            HideAll = 0,
            HideGlobal,
            ShowAll
        };
        UIStatus mShowUI = UIStatus::ShowAll;
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
        uint32_t mSampleGuiPositionX = 20;
        uint32_t mSampleGuiPositionY = 40;

        Sample(Renderer::UniquePtr& pRenderer) : mpRenderer(std::move(pRenderer)) {}
        Sample(const Sample&) = delete;
        Sample& operator=(const Sample&) = delete;
        Fbo::SharedPtr mpBackBufferFBO;     ///< The FBO for the back buffer
        //Testing
        SampleTest::UniquePtr mpSampleTest = nullptr;
    };
    enum_class_operators(SampleConfig::Flags);
};
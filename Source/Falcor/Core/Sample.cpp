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
#include "stdafx.h"
#include "Sample.h"
#include "RenderGraph/RenderPassLibrary.h"
#include "Core/Platform/ProgressBar.h"
#include "Utils/StringUtils.h"
#include <sstream>
#include <fstream>
#include "Utils/Threading.h"
#include "dear_imgui/imgui.h"

namespace Falcor
{
    IFramework* gpFramework = nullptr;
    namespace
    {
        std::string kMonospaceFont = "monospace";
    }

    void Sample::handleWindowSizeChange()
    {
        if (!gpDevice) return;
        // Tell the device to resize the swap chain
        auto winSize = mpWindow->getClientAreaSize();
        auto pBackBufferFBO = gpDevice->resizeSwapChain(winSize.x, winSize.y);
        auto width = pBackBufferFBO->getWidth();
        auto height = pBackBufferFBO->getHeight();

        // Recreate target fbo
        auto pCurrentFbo = mpTargetFBO;
        mpTargetFBO = Fbo::create2D(width, height, pBackBufferFBO->getDesc());
        gpDevice->getRenderContext()->blit(pCurrentFbo->getColorTexture(0)->getSRV(), mpTargetFBO->getRenderTargetView(0));
        
        // Tell the GUI the swap-chain size changed
        if(mpGui) mpGui->onWindowResize(width, height);

        // Resize the pixel zoom
        if(mpPixelZoom) mpPixelZoom->onResizeSwapChain(gpDevice->getSwapChainFbo().get());

        // Call the user callback
        if(mpRenderer) mpRenderer->onResizeSwapChain(width, height);
    }

    void Sample::handleKeyboardEvent(const KeyboardEvent& keyEvent)
    {
        if (keyEvent.type == KeyboardEvent::Type::KeyPressed)       mPressedKeys.insert(keyEvent.key);
        else if (keyEvent.type == KeyboardEvent::Type::KeyReleased) mPressedKeys.erase(keyEvent.key);

        if (mShowUI && mpGui->onKeyboardEvent(keyEvent)) return;
        if (mpRenderer && mpRenderer->onKeyEvent(keyEvent)) return;

        // Checks if should toggle zoom
        mpPixelZoom->onKeyboardEvent(keyEvent);

        // Consume system messages first
        if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
        {
            if (keyEvent.mods.isShiftDown && keyEvent.key == KeyboardEvent::Key::F12)
            {
                initVideoCapture();
            }
            else if (keyEvent.mods.isCtrlDown)
            {
                switch (keyEvent.key)
                {
                case KeyboardEvent::Key::Pause:
                    mRendererPaused = !mRendererPaused;
                    break;
                }
            }
            else if (!keyEvent.mods.isAltDown && !keyEvent.mods.isCtrlDown && !keyEvent.mods.isShiftDown)
            {
                switch (keyEvent.key)
                {
                case KeyboardEvent::Key::F12:
                    mCaptureScreen = true;
                    break;
#if _PROFILING_ENABLED
                case KeyboardEvent::Key::P:
                    gProfileEnabled = !gProfileEnabled;
                    break;
#endif
                case KeyboardEvent::Key::V:
                    mVsyncOn = !mVsyncOn;
                    gpDevice->toggleVSync(mVsyncOn);
                    mFrameRate.reset();
                    mClock.now(0);
                    break;
                case KeyboardEvent::Key::F2:
                    toggleUI(!mShowUI);
                    break;
                case KeyboardEvent::Key::F5:
                    Program::reloadAllPrograms();
                    if (mpRenderer) mpRenderer->onDataReload();
                    break;
                case KeyboardEvent::Key::Escape:
                    if (mVideoCapture.pVideoCapture)
                    {
                        endVideoCapture();
                    }
                    else
                    {
                        mpWindow->shutdown();
                    }
                    break;
                case KeyboardEvent::Key::Pause:
                    mClock.isPaused() ? mClock.play() : mClock.pause();
                    break;
                }
            }
        }
    }

    void Sample::handleDroppedFile(const std::string& filename)
    {
        if(mpRenderer) mpRenderer->onDroppedFile(filename);
    }

    void Sample::handleMouseEvent(const MouseEvent& mouseEvent)
    {
        if (mShowUI && mpGui->onMouseEvent(mouseEvent)) return;
        if (mpRenderer && mpRenderer->onMouseEvent(mouseEvent)) return;
        if (mpPixelZoom->onMouseEvent(mouseEvent)) return;
    }

    // Sample functions
    Sample::~Sample()
    {
        if (mVideoCapture.pVideoCapture) endVideoCapture();

        Clock::shutdown();
        Threading::shutdown();
        Scripting::shutdown();
        RenderPassLibrary::instance().shutdown();
        TextRenderer::shutdown();
        mpGui.reset();
        mpTargetFBO.reset();        
        mpPixelZoom.reset();
        if(gpDevice) gpDevice->cleanup();
        gpDevice.reset();
        OSServices::stop();
    }
    
    void Sample::run(const SampleConfig& config, IRenderer::UniquePtr& pRenderer, uint32_t argc, char** argv)
    {
        Sample s(pRenderer);
        s.runInternal(config, argc, argv);
    }
    
    void Sample::run(const std::string& filename, IRenderer::UniquePtr& pRenderer, uint32_t argc, char** argv)
    {
        auto err = [filename](std::string_view msg) {logError("Error in Sample::Run(). `" + filename + "` " + msg); };
        Sample s(pRenderer);

        s.startScripting(); // We have to do that before running the script
        std::string fullpath;
        SampleConfig c;
        if (findFileInDataDirectories(filename, fullpath))
        {
            Scripting::Context ctx;
            std::string errorLog;
            try
            {
                Scripting::runScriptFromFile(fullpath, ctx);
                auto configs = ctx.getObjects<SampleConfig>();
                if (configs.empty()) err("doesn't contain any SampleConfig objects");
                else
                {
                    if (configs.size() > 1) err("contains multiple SampleConfig objects. Using the first one");
                    c = configs[0];
                }
            }
            catch (std::exception e)
            {
                err("\n" + errorLog);
            }            
        }
        else
        {
            err(" doesn't exist. Using default configuration");
        }
        s.runInternal(c, argc, argv);
    }

    void Sample::runInternal(const SampleConfig& config, uint32_t argc, char** argv)
    {
        gpFramework = this;

        OSServices::start();
        startScripting();
        Threading::start();

        mShowUI = config.showUI;
        mClock.timeScale(config.timeScale);
        if (config.pauseTime) mClock.pause();
        mVsyncOn = config.deviceDesc.enableVsync;
        Logger::showBoxOnError(config.showMessageBoxOnError);

        // Create the window
        mpWindow = Window::create(config.windowDesc, this);
        if (mpWindow == nullptr)
        {
            logError("Failed to create device and window");
            return;
        }

        // Show the progress bar
        ProgressBar::MessageList msgList =
        {
            { "Initializing Falcor" },
            { "Takes a while, doesn't it?" },
            { "Don't get too bored now" },
            { "Getting there" },
            { "Loading. Seriously, loading" },
            { "Are we there yet?"},
            { "NI!"}
        };

        ProgressBar::SharedPtr pBar = ProgressBar::show(msgList);

        Device::Desc d = config.deviceDesc;
        gpDevice = Device::create(mpWindow, config.deviceDesc);
        if (gpDevice == nullptr)
        {
            logError("Failed to create device");
            return;
        }

        Clock::start();

        // Get the default objects before calling onLoad()
        auto pBackBufferFBO = gpDevice->getSwapChainFbo();
        mpTargetFBO = Fbo::create2D(pBackBufferFBO->getWidth(), pBackBufferFBO->getHeight(), pBackBufferFBO->getDesc());

        // Init the UI
        initUI();
        mpPixelZoom = PixelZoom::create(mpTargetFBO.get());

#ifdef _WIN32
        // Set the icon
        setWindowIcon("Framework\\Nvidia.ico", mpWindow->getApiHandle());

        if (argc == 0 || argv == nullptr)
        {
            mArgList.parseCommandLine(GetCommandLineA());
        }
        else
#endif
        {
            mArgList.parseCommandLine(concatCommandLine(argc, argv));
        }

        // Load and run
        mpRenderer->onLoad(getRenderContext());
        pBar = nullptr;

        mFrameRate.reset();
        mpWindow->msgLoop();

        mpRenderer->onShutdown();
        if (gpDevice) gpDevice->flushAndSync();
        mpRenderer = nullptr;
        Logger::shutdown();
    }

    void screenSizeUI(Gui::Widgets& widget, uvec2 screenDims)
    {
        static const uvec2 resolutions[] =
        {
            {0, 0},
            {1280, 720},
            {1920, 1080},
            {1920, 1200},
            {2560, 1440},
            {3840, 2160},
        };

        static const auto initDropDown = [](const uvec2 resolutions[], uint32_t count) -> Gui::DropdownList
        {
            Gui::DropdownList list;
            for (uint32_t i = 0 ; i < count; i++)
            {
                list.push_back({ i, to_string(resolutions[i].x) + "x" + to_string(resolutions[i].y) });
            }
            list[0] = { 0, "Custom" };
            return list;
        };

        auto initDropDownVal = [](const uvec2 resolutions[], uint32_t count, uvec2 screenDims)
        {
            for (uint32_t i = 0; i < count; i++)
            {
                if (screenDims == resolutions[i]) return i;
            }
            return 0u;
        };

        static const Gui::DropdownList dropdownList = initDropDown(resolutions, arraysize(resolutions));
        uint32_t currentVal = initDropDownVal(resolutions, arraysize(resolutions), screenDims);

        widget.var("Screen Resolution", screenDims);
        if (widget.dropdown("Change Resolution", dropdownList, currentVal) && (currentVal != 0)) gpFramework->resizeSwapChain(resolutions[currentVal].x, resolutions[currentVal].y);
    }

    std::string Sample::getKeyboardShortcutsStr()
    {
        constexpr char help[] =
            "  'F2'      - Show\\Hide GUI\n"
            "  'F5'      - Reload shaders\n"
            "  'ESC'     - Quit\n"
            "  'V'       - Toggle VSync\n"
            "  'F12'     - Capture screenshot\n"
            "  'Shift+F12'  - Video capture\n"
            "  'Pause'      - Pause\\resume the global timer\n"
            "  'Ctrl+Pause' - Pause\\resume the renderer\n"
            "  'Z'       - Zoom in on a pixel\n"
            "  'MouseWheel' - Change level of zoom\n"
#if _PROFILING_ENABLED
            "  'P'       - Enable profiling\n"
#endif
            ;

        return help;
    }

    void Sample::renderGlobalUI(Gui* pGui)
    {
        ImGui::TextUnformatted("Keyboard Shortcuts");
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(450.0f);
            ImGui::TextUnformatted(getKeyboardShortcutsStr().c_str());
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }

        auto controlsGroup = Gui::Group(pGui, "Global Controls");
        if (controlsGroup.open())
        {            
            float t = (float)mClock.now();
            if (controlsGroup.var("Time", t, 0.f, FLT_MAX)) mClock.now(double(t));
            if (controlsGroup.button("Reset")) mClock.now(0.0);
            bool timePaused = mClock.isPaused();
            if (controlsGroup.button(timePaused ? "Play" : "Pause", true)) timePaused ? mClock.pause() : mClock.play();
            if (controlsGroup.button("Stop", true)) mClock.stop();

            float scale = (float)mClock.timeScale();
            if (controlsGroup.var("Scale", scale, 0.f, FLT_MAX)) mClock.timeScale(scale);
            controlsGroup.separator();

            if (controlsGroup.button(mRendererPaused ? "Resume Rendering" : "Pause Rendering")) mRendererPaused = !mRendererPaused;
            controlsGroup.tooltip("Freeze the renderer and keep displaying the last rendered frame. The renderer will keep accepting mouse/keyboard/GUI messages. Changes in the UI will not be reflected in the displayed image until the renderer is unfrozen");

            controlsGroup.separator();
            screenSizeUI(controlsGroup, mpWindow->getClientAreaSize());
            controlsGroup.separator();

            mCaptureScreen = controlsGroup.button("Screen Capture");
            if (controlsGroup.button("Video Capture", true)) initVideoCapture();
            if (controlsGroup.button("Save Config")) saveConfigToFile();

            controlsGroup.release();
        }
        
    }

    void Sample::renderUI()
    {
        PROFILE("renderUI");

        if (mShowUI || gProfileEnabled)
        {
            mpGui->beginFrame();

            if(mShowUI) mpRenderer->onGuiRender(mpGui.get());
            if (mVideoCapture.displayUI && mVideoCapture.pUI)
            {
                Gui::Window w(mpGui.get(), "Video Capture", mVideoCapture.displayUI, { 20, 300 }, { 300, 280 });
                mVideoCapture.pUI->render(w);
            }

            if (gProfileEnabled)
            {
                uint32_t y = gpDevice->getSwapChainFbo()->getHeight() - 360;

                mpGui->setActiveFont(kMonospaceFont);

                Gui::Window profilerWindow(mpGui.get(), "Profiler", gProfileEnabled, { 800, 350 }, { 10, y });
                Profiler::endEvent("renderUI"); // Stop the timer

                if(gProfileEnabled)
                {
                    profilerWindow.text(Profiler::getEventsString().c_str());
                    Profiler::startEvent("renderUI");
                    profilerWindow.release();
                }
                mpGui->setActiveFont("");
            }

            mpGui->render(getRenderContext(), gpDevice->getSwapChainFbo(), (float)mFrameRate.getLastFrameTime());
        }
    }

    void Sample::renderFrame()
    {
        if (gpDevice && gpDevice->isWindowOccluded()) return;

        mClock.tick();
        mFrameRate.newFrame();
        if (mVideoCapture.fixedTimeDelta) { mClock.now(mVideoCapture.currentTime); }

        {
            PROFILE("onFrameRender");

            // The swap-chain FBO might have changed between frames, so get it
            if (!mRendererPaused)
            {
                RenderContext* pRenderContext = gpDevice ? gpDevice->getRenderContext() : nullptr;
                mpRenderer->onFrameRender(pRenderContext, mpTargetFBO);
            }
        }
        
        if (gpDevice)
        {
            // Copy the render-target
            auto pSwapChainFbo = gpDevice->getSwapChainFbo();
            RenderContext* pRenderContext = getRenderContext();
            pRenderContext->copyResource(pSwapChainFbo->getColorTexture(0).get(), mpTargetFBO->getColorTexture(0).get());

            // Capture video frame before UI is rendered
            bool captureVideoUI = mVideoCapture.pUI && mVideoCapture.pUI->captureUI();  // Check capture mode here once only, as its value may change after renderGUI()
            if (!captureVideoUI) captureVideoFrame();
            renderUI();

            pSwapChainFbo = gpDevice->getSwapChainFbo(); // The UI might have triggered a swap-chain resize, invalidating the previous FBO            
            if (mpPixelZoom) mpPixelZoom->render(pRenderContext, pSwapChainFbo.get());

#if _PROFILING_ENABLED
            Profiler::endFrame();
#endif
            // Capture video frame after UI is rendered
            if (captureVideoUI) captureVideoFrame();
            if (mCaptureScreen) captureScreen();

            {
                PROFILE("present", Profiler::Flags::Internal);
                gpDevice->present();
            }
        }

        Console::flush();
    }

    std::string Sample::captureScreen(const std::string explicitFilename, const std::string explicitOutputDirectory)
    {
        mCaptureScreen = false;

        std::string filename = explicitFilename != "" ? explicitFilename : getExecutableName();
        std::string outputDirectory = explicitOutputDirectory != "" ? explicitOutputDirectory : getExecutableDirectory();

        std::string pngFile;
        if (findAvailableFilename(filename, outputDirectory, "png", pngFile))
        {
            Texture::SharedPtr pTexture;
            pTexture = gpDevice->getSwapChainFbo()->getColorTexture(0);
            pTexture->captureToFile(0, 0, pngFile);
        }
        else
        {
            logError("Could not find available filename when capturing screen");
            return "";
        }

         return pngFile;
    }

    void Sample::initUI()
    {
        float scaling = getDisplayScaleFactor();
        const auto& pSwapChainFbo = gpDevice->getSwapChainFbo();
        mpGui = Gui::create(uint32_t(pSwapChainFbo->getWidth()), uint32_t(pSwapChainFbo->getHeight()), scaling);
        mpGui->addFont(kMonospaceFont, "Framework/Fonts/consolab.ttf");
        TextRenderer::start();
    }

    void Sample::resizeSwapChain(uint32_t width, uint32_t height)
    {
        mpWindow->resize(width, height);
    }

    bool Sample::isKeyPressed(const KeyboardEvent::Key& key)
    {
        return mPressedKeys.find(key) != mPressedKeys.cend();
    }

    void Sample::initVideoCapture()
    {
        if (mVideoCapture.pUI == nullptr)
        {
            mVideoCapture.pUI = VideoEncoderUI::create([this]() {startVideoCapture(); }, [this]() {endVideoCapture(); });
        }
        mVideoCapture.displayUI = true;
    }

    void Sample::startVideoCapture()
    {
        // Create the Capture Object and Framebuffer.
        VideoEncoder::Desc desc;
        desc.flipY = false;
        desc.codec = mVideoCapture.pUI->getCodec();
        desc.filename = mVideoCapture.pUI->getFilename();
        const auto& pSwapChainFbo = gpDevice->getSwapChainFbo();
        desc.format = pSwapChainFbo->getColorTexture(0)->getFormat();
        desc.fps = mVideoCapture.pUI->getFPS();
        desc.height = pSwapChainFbo->getHeight();
        desc.width = pSwapChainFbo->getWidth();
        desc.bitrateMbps = mVideoCapture.pUI->getBitrate();
        desc.gopSize = mVideoCapture.pUI->getGopSize();

        mVideoCapture.pVideoCapture = VideoEncoder::create(desc);

        assert(mVideoCapture.pVideoCapture);
        mVideoCapture.pFrame.resize(desc.width*desc.height * 4);
        mVideoCapture.fixedTimeDelta = 1 / (double)desc.fps;

        if (mVideoCapture.pUI->useTimeRange())
        {
            if (mVideoCapture.pUI->getStartTime() > mVideoCapture.pUI->getEndTime())
            {
                mVideoCapture.fixedTimeDelta = -mVideoCapture.fixedTimeDelta;
            }
            mVideoCapture.currentTime = mVideoCapture.pUI->getStartTime();
        }
    }
 
    void Sample::endVideoCapture()
    {
        if (mVideoCapture.pVideoCapture)
        {
            mVideoCapture.pVideoCapture->endCapture();
            mShowUI = true;
        }
        mVideoCapture = {};
    }

    void Sample::captureVideoFrame()
    {
        if (mVideoCapture.pVideoCapture)
        {
            mVideoCapture.pVideoCapture->appendFrame(getRenderContext()->readTextureSubresource(gpDevice->getSwapChainFbo()->getColorTexture(0).get(), 0).data());

            if (mVideoCapture.pUI->useTimeRange())
            {
                if (mVideoCapture.fixedTimeDelta >= 0)
                {
                    if (mVideoCapture.currentTime >= mVideoCapture.pUI->getEndTime()) endVideoCapture();
                }
                else if (mVideoCapture.currentTime < mVideoCapture.pUI->getEndTime())
                {
                    endVideoCapture();
                }
            }

            mVideoCapture.currentTime += mVideoCapture.fixedTimeDelta;
        }
    }

    SampleConfig Sample::getConfig()
    {
        SampleConfig c;
        c.deviceDesc = gpDevice->getDesc();
        c.windowDesc = mpWindow->getDesc();
        c.showMessageBoxOnError = Logger::isBoxShownOnError();
        c.timeScale = (float)mClock.timeScale();
        c.pauseTime = mClock.isPaused();
        c.showUI = mShowUI;
        return c;
    }

    void Sample::saveConfigToFile()
    {
        std::string filename;
        if (saveFileDialog(Scripting::kFileExtensionFilters, filename))
        {
            SampleConfig c = getConfig();
            std::string s = "sampleConfig = " + ScriptBindings::to_string(c) + "\n";
            std::ofstream(filename) << s;
        }
    }

    void Sample::startScripting()
    {
        Scripting::start();
        auto bindFunc = [this](ScriptBindings::Module& m) { this->registerScriptBindings(m); };
        ScriptBindings::registerBinding(bindFunc);
    }

    void Sample::registerScriptBindings(ScriptBindings::Module& m)
    {
        auto sampleDesc = m.regClass(SampleConfig);
#define field(f_) rwField(#f_, &SampleConfig::f_)
        sampleDesc.field(windowDesc).field(deviceDesc).field(showMessageBoxOnError).field(timeScale);
        sampleDesc.field(pauseTime).field(showUI);
#undef field
        auto exit = [](int32_t errorCode) {postQuitMessage(errorCode); };
        m.func_("exit", exit, "errorCode"_a = 0);

        auto renderFrame = [this]() {ProgressBar::close(); this->renderFrame(); };
        m.func_("renderFrame", renderFrame);
    }
}

/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
 **************************************************************************/
#include "Sample.h"
#include "Macros.h"
#include "Platform/ProgressBar.h"
#include "Program/Program.h"
#include "Utils/Threading.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/Console.h"
#include "Utils/Scripting/Scripting.h"
#include "Utils/UI/TextRenderer.h"
#include "Utils/Settings.h"
#include "RenderGraph/RenderPassLibrary.h"

#include <imgui.h>

#include <sstream>
#include <fstream>

namespace Falcor
{
    IFramework* gpFramework = nullptr;

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
        if (mpGui) mpGui->onWindowResize(width, height);

        // Resize the pixel zoom
        if (mpPixelZoom) mpPixelZoom->onResizeSwapChain(gpDevice->getSwapChainFbo().get());

        // Call the user callback
        if (mpRenderer) mpRenderer->onResizeSwapChain(width, height);
    }

    void Sample::handleRenderFrame()
    {
        renderFrame();
    }

    void Sample::handleKeyboardEvent(const KeyboardEvent& keyEvent)
    {
        if (mSuppressInput)
        {
            if (keyEvent.key == Input::Key::Escape) mpWindow->shutdown();
            return;
        }

        if (mShowUI && mpGui->onKeyboardEvent(keyEvent)) return;
        mInputState.onKeyEvent(keyEvent);
        if (mpRenderer && mpRenderer->onKeyEvent(keyEvent)) return;

        // Checks if should toggle zoom
        mpPixelZoom->onKeyboardEvent(keyEvent);

        // Consume system messages first
        if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
        {
            if (keyEvent.hasModifier(Input::Modifier::Shift) && keyEvent.key == Input::Key::F12)
            {
                initVideoCapture();
            }
            else if (keyEvent.hasModifier(Input::Modifier::Ctrl))
            {
                switch (keyEvent.key)
                {
                case Input::Key::Pause:
                case Input::Key::Space:
                    mRendererPaused = !mRendererPaused;
                    break;
                default:
                    break;
                }
            }
            else if (keyEvent.mods == Input::ModifierFlags::None)
            {
                switch (keyEvent.key)
                {
                case Input::Key::F12:
                    mCaptureScreen = true;
                    break;
#if FALCOR_ENABLE_PROFILER
                case Input::Key::P:
                    Profiler::instance().setEnabled(!Profiler::instance().isEnabled());
                    break;
#endif
                case Input::Key::V:
                    mVsyncOn = !mVsyncOn;
                    gpDevice->toggleVSync(mVsyncOn);
                    mFrameRate.reset();
                    mClock.setTime(0);
                    break;
                case Input::Key::F2:
                    toggleUI(!mShowUI);
                    break;
                case Input::Key::F5:
                    {
                        HotReloadFlags reloaded = HotReloadFlags::None;
                        if (Program::reloadAllPrograms()) reloaded |= HotReloadFlags::Program;
                        if (mpRenderer) mpRenderer->onHotReload(reloaded);
                    }
                    break;
                case Input::Key::Escape:
                    if (mVideoCapture.pVideoCapture)
                    {
                        endVideoCapture();
                    }
                    else
                    {
                        mpWindow->shutdown();
                    }
                    break;
                case Input::Key::Pause:
                case Input::Key::Space:
                    mClock.isPaused() ? mClock.play() : mClock.pause();
                    break;
                default:
                    break;
                }
            }
        }
    }

    void Sample::handleMouseEvent(const MouseEvent& mouseEvent)
    {
        if (mSuppressInput) return;
        if (mShowUI && mpGui->onMouseEvent(mouseEvent)) return;
        mInputState.onMouseEvent(mouseEvent);
        if (mpRenderer && mpRenderer->onMouseEvent(mouseEvent)) return;
        if (mpPixelZoom->onMouseEvent(mouseEvent)) return;
    }

    void Sample::handleGamepadEvent(const GamepadEvent& gamepadEvent)
    {
        if (mpRenderer) mpRenderer->onGamepadEvent(gamepadEvent);
    }

    void Sample::handleGamepadState(const GamepadState& gamepadState)
    {
        if (mpRenderer) mpRenderer->onGamepadState(gamepadState);
    }

    void Sample::handleDroppedFile(const std::filesystem::path& path)
    {
        if (mpRenderer) mpRenderer->onDroppedFile(path);
    }

    Sample::Sample(IRenderer::UniquePtr& pRenderer) : mpRenderer(std::move(pRenderer)) {}

    // Sample functions
    Sample::~Sample()
    {
        mpRenderer.reset();
        if (mVideoCapture.pVideoCapture) endVideoCapture();

        // contains Python dictionaries, needs to be terminated before Scripting::shutdown()
        mpSettings.reset();

        Clock::shutdown();
        Threading::shutdown();
        Scripting::shutdown();
        RenderPassLibrary::instance().shutdown();
        TextRenderer::shutdown();
        mpGui.reset();
        mpTargetFBO.reset();
        mpPixelZoom.reset();
        if (gpDevice) gpDevice->cleanup();
        gpDevice.reset();
        OSServices::stop();
    }

    void Sample::run(const SampleConfig& config, IRenderer::UniquePtr& pRenderer, uint32_t argc, char** argv)
    {
        Sample s(pRenderer);
        try
        {
            s.startScripting();
            s.runInternal(config, argc, argv);
        }
        catch (const std::exception& e)
        {
            reportError("Caught exception:\n\n" + std::string(e.what()) + "\n\nEnable breaking on exceptions in the debugger to get a full stack trace.");
        }
        Logger::shutdown();
    }

    void Sample::run(const std::filesystem::path& path, IRenderer::UniquePtr& pRenderer, uint32_t argc, char** argv)
    {
        Sample s(pRenderer);
        try
        {
            s.startScripting(); // We have to do that before running the script
            SampleConfig c;

            std::filesystem::path fullPath;
            if (findFileInDataDirectories(path, fullPath))
            {
                Scripting::Context ctx;
                Scripting::runScriptFromFile(fullPath, ctx);
                auto configs = ctx.getObjects<SampleConfig>();
                if (configs.empty())
                {
                    logWarning("Configuration '{}' does not contain any SampleConfig objects. Using default configuration.", path);
                }
                else
                {
                    if (configs.size() > 1)
                    {
                        logWarning("Configuration '{}' does contain multiple SampleConfig objects. Using first the first one.", path);
                    }
                    c = configs[0];
                }
            }
            else
            {
                logWarning("Configuration '{}' does not exist. Using default configuration.", path);
            }

            s.runInternal(c, argc, argv);
        }
        catch (const std::exception& e)
        {
            reportError("Caught exception:\n\n" + std::string(e.what()) + "\n\nEnable breaking on exceptions in the debugger to get a full stack trace.");
        }

        Logger::shutdown();
    }

    void Sample::runInternal(const SampleConfig& config, uint32_t argc, char** argv)
    {
        gpFramework = this;

        setShowMessageBoxOnError(config.showMessageBoxOnError);

        OSServices::start();
        Threading::start();

        mpSettings.reset(new Settings);

        mSuppressInput = config.suppressInput;
        mShowUI = config.showUI;
        mClock.setTimeScale(config.timeScale);
        if (config.pauseTime) mClock.pause();
        mVsyncOn = config.deviceDesc.enableVsync;

        // Create the window
        mpWindow = Window::create(config.windowDesc, this);

        // Show the progress bar (unless window is minimized)
        ProgressBar::SharedPtr pBar;
        if (config.windowDesc.mode != Window::WindowMode::Minimized) pBar = ProgressBar::show("Initializing Falcor");

        // Create device
        Device::Desc d = config.deviceDesc;
        gpDevice = Device::create(mpWindow, config.deviceDesc);

        // Set global shader defines
        Program::DefineList globalDefines = {
            { "FALCOR_NVAPI_AVAILABLE", FALCOR_NVAPI_AVAILABLE ? "1" : "0" },
        };
        Program::addGlobalDefines(globalDefines);

        Clock::start();

        // Get the default objects before calling onLoad()
        auto pBackBufferFBO = gpDevice->getSwapChainFbo();
        mpTargetFBO = Fbo::create2D(pBackBufferFBO->getWidth(), pBackBufferFBO->getHeight(), pBackBufferFBO->getDesc());

        // Init the UI
        initUI();
        mpPixelZoom = PixelZoom::create(mpTargetFBO.get());

#if FALCOR_WINDOWS
        // Set the icon
        setWindowIcon("Framework/Nvidia.ico", mpWindow->getApiHandle());
#endif

        // Load and run
        mpRenderer->onLoad(getRenderContext());
        pBar = nullptr;

        mFrameRate.reset();
        mpWindow->msgLoop();

        mpRenderer->onShutdown();
        if (gpDevice) gpDevice->flushAndSync();
        mpRenderer = nullptr;
    }

    void screenSizeUI(Gui::Widgets& widget, uint2 screenDims)
    {
        static const uint2 resolutions[] =
        {
            {0, 0},
            {1280, 720},
            {1920, 1080},
            {1920, 1200},
            {2560, 1440},
            {3840, 2160},
        };

        static const auto initDropDown = [](const uint2 resolutions[], uint32_t count) -> Gui::DropdownList
        {
            Gui::DropdownList list;
            for (uint32_t i = 0 ; i < count; i++)
            {
                list.push_back({ i, std::to_string(resolutions[i].x) + "x" + std::to_string(resolutions[i].y) });
            }
            list[0] = { 0, "Custom" };
            return list;
        };

        auto initDropDownVal = [](const uint2 resolutions[], uint32_t count, uint2 screenDims)
        {
            for (uint32_t i = 0; i < count; i++)
            {
                if (screenDims == resolutions[i]) return i;
            }
            return 0u;
        };

        static const Gui::DropdownList dropdownList = initDropDown(resolutions, (uint32_t)std::size(resolutions));
        uint32_t currentVal = initDropDownVal(resolutions, (uint32_t)std::size(resolutions), screenDims);

        widget.var("Screen Resolution", screenDims);
        if (widget.dropdown("Change Resolution", dropdownList, currentVal) && (currentVal != 0)) gpFramework->resizeSwapChain(resolutions[currentVal].x, resolutions[currentVal].y);
    }

    std::string Sample::getKeyboardShortcutsStr()
    {
        constexpr char help[] =
            "ESC - Quit\n"
            "F2 - Show/hide UI\n"
            "F3 - Capture current camera location\n"
            "F5 - Reload shaders\n"
            "F12 - Capture screenshot\n"
            "Shift+F12 - Capture video\n"
            "V - Toggle VSync\n"
            "Pause|Space - Pause/resume the global timer\n"
            "Ctrl+Pause|Space - Pause/resume the renderer\n"
            "Z - Zoom in on a pixel\n"
            "MouseWheel - Change level of zoom\n"
#if FALCOR_ENABLE_PROFILER
            "P - Enable/disable profiler\n"
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
            float t = (float)mClock.getTime();
            if (controlsGroup.var("Time", t, 0.f, FLT_MAX)) mClock.setTime(double(t));
            if (controlsGroup.button("Reset")) mClock.setTime(0.0);
            bool timePaused = mClock.isPaused();
            if (controlsGroup.button(timePaused ? "Play" : "Pause", true)) timePaused ? mClock.pause() : mClock.play();
            if (controlsGroup.button("Stop", true)) mClock.stop();

            float scale = (float)mClock.getTimeScale();
            if (controlsGroup.var("Scale", scale, 0.f, FLT_MAX)) mClock.setTimeScale(scale);
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
        FALCOR_PROFILE("renderUI");

        auto& profiler = Profiler::instance();

        if (mShowUI || profiler.isEnabled())
        {
            mpGui->beginFrame();

            if (mShowUI) mpRenderer->onGuiRender(mpGui.get());
            if (mVideoCapture.displayUI && mVideoCapture.pUI)
            {
                Gui::Window w(mpGui.get(), "Video Capture", mVideoCapture.displayUI, { 350, 250 }, { 300, 280 });
                mVideoCapture.pUI->render(w);
            }

            if (profiler.isEnabled())
            {
                uint32_t y = gpDevice->getSwapChainFbo()->getHeight() - 360;

                bool open = profiler.isEnabled();
                Gui::Window profilerWindow(mpGui.get(), "Profiler", open, { 800, 350 }, { 10, y });
                profiler.endEvent("renderUI"); // Stop the timer

                if (open)
                {
                    if (!mpProfilerUI) mpProfilerUI = ProfilerUI::create(Profiler::instancePtr());
                    mpProfilerUI->render();
                    profiler.startEvent("renderUI");
                    profilerWindow.release();
                }

                profiler.setEnabled(open);
            }

            mpGui->render(getRenderContext(), gpDevice->getSwapChainFbo(), (float)mFrameRate.getLastFrameTime());
        }
    }

    void Sample::renderFrame()
    {
        if (gpDevice && gpDevice->isWindowOccluded()) return;

        // Check clock exit condition
        if (mClock.shouldExit()) postQuitMessage(0);

        mClock.tick();
        mFrameRate.newFrame();
        if (mVideoCapture.fixedTimeDelta) { mClock.setTime(mVideoCapture.currentTime); }

        {
            FALCOR_PROFILE("onFrameRender");

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

#if FALCOR_ENABLE_PROFILER
            Profiler::instance().endFrame();
#endif
            // Capture video frame after UI is rendered
            if (captureVideoUI) captureVideoFrame();
            if (mCaptureScreen) captureScreen();

            {
                FALCOR_PROFILE_CUSTOM("present", Profiler::Flags::Internal);
                gpDevice->present();
            }
        }

        mInputState.endFrame();

        Console::instance().flush();
    }

    std::filesystem::path Sample::captureScreen(const std::string explicitFilename, const std::filesystem::path explicitDirectory)
    {
        mCaptureScreen = false;

        std::string filename = explicitFilename.empty() ? getExecutableName() : explicitFilename;
        std::filesystem::path directory = explicitDirectory.empty() ? getExecutableDirectory() : explicitDirectory;

        std::filesystem::path path = findAvailableFilename(filename, directory, "png");
        Texture::SharedPtr pTexture;
        pTexture = gpDevice->getSwapChainFbo()->getColorTexture(0);
        pTexture->captureToFile(0, 0, path);
        return path;
    }

    void Sample::initUI()
    {
        float scaling = getDisplayScaleFactor();
        const auto& pSwapChainFbo = gpDevice->getSwapChainFbo();
        mpGui = Gui::create(uint32_t(pSwapChainFbo->getWidth()), uint32_t(pSwapChainFbo->getHeight()), scaling);
        TextRenderer::start();
    }

    void Sample::resizeSwapChain(uint32_t width, uint32_t height)
    {
        mpWindow->resize(width, height);
    }

    void Sample::initVideoCapture()
    {
        if (mVideoCapture.pUI == nullptr)
        {
            mVideoCapture.pUI = VideoEncoderUI::create([this]() {return startVideoCapture(); }, [this]() {endVideoCapture(); });
        }
        mVideoCapture.displayUI = true;
    }

    bool Sample::startVideoCapture()
    {
        // Create the Capture Object and Framebuffer.
        VideoEncoder::Desc desc;
        desc.flipY = false;
        desc.codec = mVideoCapture.pUI->getCodec();
        desc.path = mVideoCapture.pUI->getPath();
        const auto& pSwapChainFbo = gpDevice->getSwapChainFbo();
        desc.format = pSwapChainFbo->getColorTexture(0)->getFormat();
        desc.fps = mVideoCapture.pUI->getFPS();
        desc.height = pSwapChainFbo->getHeight();
        desc.width = pSwapChainFbo->getWidth();
        desc.bitrateMbps = mVideoCapture.pUI->getBitrate();
        desc.gopSize = mVideoCapture.pUI->getGopSize();

        mVideoCapture.pVideoCapture = VideoEncoder::create(desc);
        if (!mVideoCapture.pVideoCapture) return false;

        FALCOR_ASSERT(mVideoCapture.pVideoCapture);
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
        return true;
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
        c.showMessageBoxOnError = getShowMessageBoxOnError();
        c.timeScale = (float)mClock.getTimeScale();
        c.pauseTime = mClock.isPaused();
        c.showUI = mShowUI;
        return c;
    }

    void Sample::saveConfigToFile()
    {
        std::filesystem::path path;
        if (saveFileDialog(Scripting::kFileExtensionFilters, path))
        {
            SampleConfig c = getConfig();
            std::string s = "sampleConfig = " + ScriptBindings::repr(c) + "\n";
            std::ofstream(path) << s;
        }
    }

    void Sample::startScripting()
    {
        Scripting::start();
        auto bindFunc = [this](pybind11::module& m) { this->registerScriptBindings(m); };
        ScriptBindings::registerBinding(bindFunc);
    }

    void Sample::registerScriptBindings(pybind11::module& m)
    {
        using namespace pybind11::literals;

        ScriptBindings::SerializableStruct<SampleConfig> sampleConfig(m, "SampleConfig");
#define field(f_) field(#f_, &SampleConfig::f_)
        sampleConfig.field(windowDesc);
        sampleConfig.field(deviceDesc);
        sampleConfig.field(showMessageBoxOnError);
        sampleConfig.field(timeScale);
        sampleConfig.field(pauseTime);
        sampleConfig.field(showUI);
#undef field
        auto exit = [](int32_t errorCode) { postQuitMessage(errorCode); };
        m.def("exit", exit, "errorCode"_a = 0);

        auto renderFrame = [this]() {ProgressBar::close(); this->renderFrame(); };
        m.def("renderFrame", renderFrame);

        auto setWindowPos = [this](int32_t x, int32_t y) {getWindow()->setWindowPos(x, y); };
        m.def("setWindowPos", setWindowPos, "x"_a, "y"_a);

        auto resize = [this](uint32_t width, uint32_t height) {resizeSwapChain(width, height); };
        m.def("resizeSwapChain", resize, "width"_a, "height"_a);
    }
}

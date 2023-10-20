/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "SampleApp.h"
#include "Macros.h"
#include "Version.h"
#include "Core/Error.h"
#include "Core/Plugin.h"
#include "Core/AssetResolver.h"
#include "Core/Program/Program.h"
#include "Core/Program/ProgramManager.h"
#include "Core/Platform/ProgressBar.h"
#include "Utils/Threading.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/Console.h"
#include "Utils/Scripting/Scripting.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/UI/TextRenderer.h"
#include "Utils/Settings.h"
#include "Utils/StringUtils.h"

#include <imgui.h>

#include <fstream>

namespace Falcor
{
SampleApp::SampleApp(const SampleAppConfig& config)
{
    logInfo("Falcor {}", getLongVersionString());

    OSServices::start();
    Threading::start();

    mShowUI = config.showUI;
    mVsyncOn = config.windowDesc.enableVSync;
    mClock.setTimeScale(config.timeScale);
    if (config.pauseTime)
        mClock.pause();

    // Create GPU device
    mpDevice = make_ref<Device>(config.deviceDesc);

    if (!config.headless)
    {
        auto windowDesc = config.windowDesc;
        // Vulkan does not allow creating a swapchain on a minimized window.
        if (config.deviceDesc.type == Device::Type::Vulkan && windowDesc.mode == Window::WindowMode::Minimized)
            windowDesc.mode = Window::WindowMode::Normal;

        // Create the window
        mpWindow = Window::create(windowDesc, this);
        mpWindow->setWindowIcon(getRuntimeDirectory() / "data/framework/nvidia.ico");

        // Create swapchain
        Swapchain::Desc desc;
        desc.format = config.colorFormat;
        desc.width = mpWindow->getClientAreaSize().x;
        desc.height = mpWindow->getClientAreaSize().y;
        desc.imageCount = 3;
        desc.enableVSync = mVsyncOn;
        mpSwapchain = make_ref<Swapchain>(mpDevice, desc, mpWindow->getApiHandle());

        // Show the progress bar (unless window is minimized)
        if (windowDesc.mode != Window::WindowMode::Minimized)
            mProgressBar.show("Initializing Falcor");

        // When not running headless, we want to show message boxes on error by default.
        setErrorDiagnosticFlags(getErrorDiagnosticFlags() | ErrorDiagnosticFlags::ShowMessageBoxOnError);
    }

    // Create target frame buffer
    uint2 fboSize = mpWindow ? mpWindow->getClientAreaSize() : uint2(config.windowDesc.width, config.windowDesc.height);
    mpTargetFBO = Fbo::create2D(mpDevice, fboSize.x, fboSize.y, config.colorFormat, config.depthFormat);

    // Setup asset search paths.
    AssetResolver& resolver = AssetResolver::getDefaultResolver();
    resolver.addSearchPath(getProjectDirectory() / "media");
    for (auto& path : Settings::getGlobalSettings().getSearchDirectories("media"))
        resolver.addSearchPath(path);

    mpDevice->getProgramManager()->setGenerateDebugInfoEnabled(config.generateShaderDebugInfo);
    if (config.shaderPreciseFloat)
    {
        mpDevice->getProgramManager()->setForcedCompilerFlags(
            {SlangCompilerFlags::FloatingPointModePrecise, SlangCompilerFlags::FloatingPointModeFast}
        );
    }

    // Init the UI
    initUI();
    mpPixelZoom = std::make_unique<PixelZoom>(mpDevice, mpTargetFBO.get());

    PluginManager::instance().loadAllPlugins();
}

SampleApp::~SampleApp()
{
    mpPausedRenderOutput.reset();
    mpProfilerUI.reset();

    mpDevice->wait();

    Threading::shutdown();
    Scripting::shutdown();
    PluginManager::instance().releaseAllPlugins();
    mpGui.reset();
    mpTextRenderer.reset();
    mpTargetFBO.reset();
    mpPixelZoom.reset();

    mpSwapchain.reset();
    mpWindow.reset();
#if FALCOR_ENABLE_OBJECT_TRACKING
    Object::dumpAliveObjects();
#endif
    mpDevice.reset();
#ifdef _DEBUG
    Device::reportLiveObjects();
#endif

    OSServices::stop();
    Logger::shutdown();
}

const Settings& SampleApp::getSettings() const
{
    return Settings::getGlobalSettings();
}

Settings& SampleApp::getSettings()
{
    return Settings::getGlobalSettings();
}

int SampleApp::run()
{
    startScripting();
    runInternal();
    return mReturnCode;
}

void SampleApp::handleWindowSizeChange()
{
    FALCOR_ASSERT(mpDevice && mpWindow && mpSwapchain);

    // Tell the device to resize the swap chain
    auto newSize = mpWindow->getClientAreaSize();
    uint32_t width = newSize.x;
    uint32_t height = newSize.y;

    mpSwapchain->resize(width, height);

    resizeTargetFBO(width, height);
}

void SampleApp::handleRenderFrame()
{
    renderFrame();
}

void SampleApp::handleKeyboardEvent(const KeyboardEvent& keyEvent)
{
    if (mShowUI && mpGui->onKeyboardEvent(keyEvent))
        return;
    mInputState.onKeyEvent(keyEvent);
    if (onKeyEvent(keyEvent))
        return;

    // Checks if should toggle zoom
    mpPixelZoom->onKeyboardEvent(keyEvent);

    // Consume system messages first
    if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        if (keyEvent.hasModifier(Input::Modifier::Ctrl))
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
                getDevice()->getProfiler()->setEnabled(!getDevice()->getProfiler()->isEnabled());
                break;
#endif
            case Input::Key::V:
                // TODO(@skallweit) we'd need to recreate the swapchain here
                mVsyncOn = !mVsyncOn;
                mFrameRate.reset();
                mClock.setTime(0);
                break;
            case Input::Key::F2:
                toggleUI(!mShowUI);
                break;
            case Input::Key::F5:
            {
                HotReloadFlags reloaded = HotReloadFlags::None;
                if (mpDevice->getProgramManager()->reloadAllPrograms())
                    reloaded |= HotReloadFlags::Program;
                onHotReload(reloaded);
            }
            break;
            case Input::Key::Escape:
                mpWindow->shutdown();
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

void SampleApp::handleMouseEvent(const MouseEvent& mouseEvent)
{
    if (mShowUI && mpGui->onMouseEvent(mouseEvent))
        return;
    mInputState.onMouseEvent(mouseEvent);
    if (onMouseEvent(mouseEvent))
        return;
    if (mpPixelZoom->onMouseEvent(mouseEvent))
        return;
}

void SampleApp::handleGamepadEvent(const GamepadEvent& gamepadEvent)
{
    onGamepadEvent(gamepadEvent);
}

void SampleApp::handleGamepadState(const GamepadState& gamepadState)
{
    onGamepadState(gamepadState);
}

void SampleApp::handleDroppedFile(const std::filesystem::path& path)
{
    onDroppedFile(path);
}

void SampleApp::runInternal()
{
    // Load and run
    onLoad(getRenderContext());

    mProgressBar.close();

    mFrameRate.reset();

    // If a window was created, run the message loop, otherwise just run a render loop.
    if (mpWindow)
    {
        mpWindow->msgLoop();
    }
    else
    {
        while (!mShouldTerminate)
            handleRenderFrame();
    }

    onShutdown();
}

bool screenSizeUI(Gui::Widgets& widget, uint2& screenDims)
{
    static const uint2 resolutions[] = {
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
        for (uint32_t i = 0; i < count; i++)
        {
            list.push_back({i, std::to_string(resolutions[i].x) + "x" + std::to_string(resolutions[i].y)});
        }
        list[0] = {0, "Custom"};
        return list;
    };

    auto initDropDownVal = [](const uint2 resolutions[], uint32_t count, uint2 screenDims)
    {
        for (uint32_t i = 0; i < count; i++)
        {
            if (all(screenDims == resolutions[i]))
                return i;
        }
        return 0u;
    };

    static const Gui::DropdownList dropdownList = initDropDown(resolutions, (uint32_t)std::size(resolutions));
    uint32_t currentVal = initDropDownVal(resolutions, (uint32_t)std::size(resolutions), screenDims);

    widget.var("Screen Resolution", screenDims);
    if (widget.dropdown("Change Resolution", dropdownList, currentVal) && (currentVal != 0))
    {
        screenDims = resolutions[currentVal];
        return true;
    }
    return false;
}

std::string SampleApp::getKeyboardShortcutsStr()
{
    constexpr char help[] =
        "ESC - Quit\n"
        "F2 - Show/hide UI\n"
        "F3 - Capture current camera location\n"
        "F5 - Reload shaders\n"
        "F12 - Capture screenshot\n"
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

void SampleApp::renderGlobalUI(Gui* pGui)
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
        if (controlsGroup.var("Time", t, 0.f, FLT_MAX))
            mClock.setTime(double(t));
        if (controlsGroup.button("Reset"))
            mClock.setTime(0.0);
        bool timePaused = mClock.isPaused();
        if (controlsGroup.button(timePaused ? "Play" : "Pause", true))
            timePaused ? mClock.play() : mClock.pause();
        if (controlsGroup.button("Stop", true))
            mClock.stop();

        float scale = (float)mClock.getTimeScale();
        if (controlsGroup.var("Scale", scale, 0.f, FLT_MAX))
            mClock.setTimeScale(scale);
        controlsGroup.separator();

        if (controlsGroup.button(mRendererPaused ? "Resume Rendering" : "Pause Rendering"))
            mRendererPaused = !mRendererPaused;
        controlsGroup.tooltip(
            "Freeze the renderer and keep displaying the last rendered frame. The renderer will keep accepting mouse/keyboard/GUI "
            "messages. Changes in the UI will not be reflected in the displayed image until the renderer is unfrozen"
        );

        controlsGroup.separator();
        uint2 screenDims = mpWindow->getClientAreaSize();
        if (screenSizeUI(controlsGroup, screenDims))
            resizeFrameBuffer(screenDims.x, screenDims.y);

        controlsGroup.separator();

        mCaptureScreen = controlsGroup.button("Screen Capture");
        if (controlsGroup.button("Save Config"))
            saveConfigToFile();

        controlsGroup.release();
    }
}

void SampleApp::renderUI()
{
    RenderContext* pRenderContext = mpDevice->getRenderContext();
    Profiler* pProfiler = mpDevice->getProfiler();

    FALCOR_PROFILE(pRenderContext, "renderUI");

    if (mShowUI || pProfiler->isEnabled())
    {
        mpGui->beginFrame();

        if (mShowUI)
            onGuiRender(mpGui.get());

        if (pProfiler->isEnabled())
        {
            uint32_t y = mpTargetFBO->getHeight() - 360;

            bool open = pProfiler->isEnabled();
            Gui::Window profilerWindow(mpGui.get(), "Profiler", open, {800, 600}, {350, 80});
            pProfiler->endEvent(pRenderContext, "renderUI"); // Stop the timer

            if (open)
            {
                if (!mpProfilerUI)
                    mpProfilerUI = std::make_unique<ProfilerUI>(pProfiler);
                mpProfilerUI->render();
                pProfiler->startEvent(pRenderContext, "renderUI");
                profilerWindow.release();
            }

            pProfiler->setEnabled(open);
        }

        mpGui->render(pRenderContext, mpTargetFBO, (float)mFrameRate.getLastFrameTime());
    }
}

void SampleApp::renderFrame()
{
    FALCOR_ASSERT(mpDevice);

    // Skip rendering if window is occluded.
    if (mpSwapchain && mpSwapchain->isOccluded())
        return;

    // Check clock exit condition.
    if (mClock.shouldExit())
        shutdown();

    // Handle clock.
    mClock.tick();
    mFrameRate.newFrame();

    RenderContext* pRenderContext = mpDevice->getRenderContext();

    // Render a frame.
    // If the renderer is paused, create a copy of the rendered output and copy that back each frame.
    if (mRendererPaused && mpPausedRenderOutput)
    {
        pRenderContext->blit(mpPausedRenderOutput->getSRV(), mpTargetFBO->getColorTexture(0)->getRTV());
    }
    else
    {
        FALCOR_PROFILE(pRenderContext, "onFrameRender");
        onFrameRender(pRenderContext, mpTargetFBO);

        if (mRendererPaused)
        {
            auto srcTexture = mpTargetFBO->getColorTexture(0);
            mpPausedRenderOutput =
                mpDevice->createTexture2D(srcTexture->getWidth(), srcTexture->getHeight(), srcTexture->getFormat(), 1, 1);
            pRenderContext->copyResource(mpPausedRenderOutput.get(), srcTexture.get());
        }
        else
        {
            mpPausedRenderOutput = nullptr;
        }
    }

    // Render the UI.
    renderUI();

    if (mpPixelZoom)
        mpPixelZoom->render(pRenderContext, mpTargetFBO.get());

#if FALCOR_ENABLE_PROFILER
    mpDevice->getProfiler()->endFrame(pRenderContext);
#endif

    if (mCaptureScreen)
        captureScreen(mpTargetFBO->getColorTexture(0).get());

    // Copy framebuffer to swapchain image.
    if (mpSwapchain)
    {
        int imageIndex = mpSwapchain->acquireNextImage();
        FALCOR_ASSERT(imageIndex >= 0 && imageIndex < (int)mpSwapchain->getDesc().imageCount);
        const Texture* pSwapchainImage = mpSwapchain->getImage(imageIndex).get();
        pRenderContext->copyResource(pSwapchainImage, mpTargetFBO->getColorTexture(0).get());
        pRenderContext->resourceBarrier(pSwapchainImage, Resource::State::Present);
        pRenderContext->submit();
        mpSwapchain->present();
    }

    mpDevice->endFrame();

    mInputState.endFrame();

    mConsole.flush();
}

void SampleApp::resizeTargetFBO(uint32_t width, uint32_t height)
{
    // Resize target frame buffer.
    auto pPrevFBO = mpTargetFBO;
    mpTargetFBO = Fbo::create2D(mpDevice, width, height, pPrevFBO->getDesc());
    mpDevice->getRenderContext()->blit(pPrevFBO->getColorTexture(0)->getSRV(), mpTargetFBO->getRenderTargetView(0));

    // Tell the GUI the swap-chain size changed
    if (mpGui)
        mpGui->onWindowResize(width, height);

    // Resize the pixel zoom
    if (mpPixelZoom)
        mpPixelZoom->onResize(mpTargetFBO.get());

    // Call the user callback
    onResize(width, height);
}

void SampleApp::initUI()
{
    float scaling = getDisplayScaleFactor();
    mpGui = std::make_unique<Gui>(mpDevice, mpTargetFBO->getWidth(), mpTargetFBO->getHeight(), scaling);
    mpTextRenderer = std::make_unique<TextRenderer>(mpDevice);
}

void SampleApp::resizeFrameBuffer(uint32_t width, uint32_t height)
{
    if (mpWindow)
    {
        // If we have a window, resize it. This will result in a call
        // back to handleWindowSizeChange() which in turn will resize the frame buffer.
        mpWindow->resize(width, height);
    }
    else
    {
        // If we have no window, resize the frame buffer directly.
        resizeTargetFBO(width, height);
    }
}

void SampleApp::captureScreen(Texture* pTexture)
{
    mCaptureScreen = false;

    std::string filename = getExecutableName();
    std::filesystem::path directory = getRuntimeDirectory();
    std::filesystem::path path = findAvailableFilename(filename, directory, "png");
    pTexture->captureToFile(0, 0, path);
}

void SampleApp::shutdown(int returnCode)
{
    mShouldTerminate = true;
    mReturnCode = returnCode;
    if (mpWindow)
        mpWindow->shutdown();
}

SampleAppConfig SampleApp::getConfig() const
{
    SampleAppConfig c;
    c.deviceDesc = mpDevice->getDesc();
    c.windowDesc = mpWindow->getDesc();
    c.timeScale = (float)mClock.getTimeScale();
    c.pauseTime = mClock.isPaused();
    c.showUI = mShowUI;
    return c;
}

void SampleApp::saveConfigToFile()
{
    std::filesystem::path path;
    if (saveFileDialog(Scripting::kFileExtensionFilters, path))
    {
        SampleAppConfig c = getConfig();
        std::string s = "sampleConfig = " + ScriptBindings::repr(c) + "\n";
        std::ofstream(path) << s;
    }
}

void SampleApp::startScripting()
{
    auto bindFunc = [this](pybind11::module& m) { this->registerScriptBindings(m); };
    ScriptBindings::registerBinding(bindFunc);
    Scripting::start();
}

void SampleApp::registerScriptBindings(pybind11::module& m)
{
    using namespace pybind11::literals;

    auto exit = [this](int32_t errorCode) { this->shutdown(errorCode); };
    m.def("exit", exit, "errorCode"_a = 0);

    auto renderFrame = [this]()
    {
        mProgressBar.close();
        this->renderFrame();
    };
    m.def("renderFrame", renderFrame);

    auto setWindowPos = [this](int32_t x, int32_t y)
    {
        if (auto pWindow = getWindow())
            pWindow->setWindowPos(x, y);
    };
    m.def("setWindowPos", setWindowPos, "x"_a, "y"_a);

    auto resize = [this](uint32_t width, uint32_t height) { resizeFrameBuffer(width, height); };
    m.def("resizeSwapChain", resize, "width"_a, "height"_a); // PYTHONDEPRECATED
    m.def("resizeFrameBuffer", resize, "width"_a, "height"_a);
}
} // namespace Falcor

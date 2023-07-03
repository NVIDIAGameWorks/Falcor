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
#pragma once
#include "Window.h"
#include "Core/Macros.h"
#include "Core/HotReloadFlags.h"
#include "Core/Platform/ProgressBar.h"
#include "Core/API/Device.h"
#include "Core/API/Swapchain.h"
#include "Utils/Timing/FrameRate.h"
#include "Utils/Timing/ProfilerUI.h"
#include "Utils/UI/Gui.h"
#include "Utils/UI/PixelZoom.h"
#include "Utils/UI/InputState.h"
#include "Utils/Scripting/Console.h"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace pybind11
{
class module_;
using module = module_;
} // namespace pybind11
namespace Falcor
{
class Settings;
class TextRenderer;

/**
 * Sample application configuration.
 */
struct SampleAppConfig
{
    Device::Desc deviceDesc; ///< GPU device settings.
    Window::Desc windowDesc; ///< Window settings.

    ResourceFormat colorFormat = ResourceFormat::BGRA8UnormSrgb; ///< Color format of the frame buffer.
    ResourceFormat depthFormat = ResourceFormat::D32Float;       ///< Depth buffer format of the frame buffer.

    bool headless = false;  ///< Do not create a window and handle user input.
    float timeScale = 1.0f; ///< A scaling factor for the time elapsed between frames.
    bool pauseTime = false; ///< Control whether or not to start the clock when the sample start running.
    bool showUI = true;     ///< Show the UI.

    bool generateShaderDebugInfo = false;
    bool shaderPreciseFloat = false;
};

/**
 * Sample application base class.
 */
class FALCOR_API SampleApp : public Window::ICallbacks
{
public:
    SampleApp(const SampleAppConfig& config);
    virtual ~SampleApp();

    /**
     * Enters the main loop of the application.
     * @return Returns the return code set in shutdown().
     */
    int run();

    /**
     * Called once right after context creation.
     */
    virtual void onLoad(RenderContext* pRenderContext) {}

    /**
     * Called right before the context is destroyed.
     */
    virtual void onShutdown() {}

    /**
     * Called every time the main frame buffer is changed in size.
     * This typically happens upon resizing the main window.
     */
    virtual void onResize(uint32_t width, uint32_t height) {}

    /**
     * Called on each frame render.
     */
    virtual void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) {}

    /**
     * Called after onFrameRender().
     * It is highly recommended to use onGuiRender() exclusively for GUI handling. onGuiRender() will not be called when the GUI is hidden,
     * which should help reduce CPU overhead. You could also ignore this and render the GUI directly in your onFrameRender() function, but
     * that is discouraged.
     */
    virtual void onGuiRender(Gui* pGui){};

    /**
     * Called after Options in settings have been changed.
     * This seems to be the only reasonable way to handle all the possible options from:
     * Mogwai starts, then script runs, then scene loads, then rendering happens.
     * Mogwai starts and loads script, in which scene is loaded and rendering happens.
     * In all the cases, we want the Options to take effect before any window is shown,
     * which means we pretty much have to be told just after the Options have been set.
     */
    virtual void onOptionsChange() {}

    /**
     * Called upon hot reload (by pressing F5).
     * @param[in] reloaded Resources that have been reloaded.
     */
    virtual void onHotReload(HotReloadFlags reloaded) {}

    /**
     * Called every time a key event occurred.
     * @param[in] keyEvent The keyboard event.
     * @return true if the event was consumed by the callback, otherwise false.
     */
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) { return false; }

    /**
     * Called every time a mouse event occurred.
     * @param[in] mouseEvent The mouse event.
     * @return true if the event was consumed by the callback, otherwise false.
     */
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) { return false; }

    /**
     * Called every time a gamepad event occured.
     * @param[in] gamepadEvent The gamepad event.
     * @return true if the event was consumed by the callback, otherwise false.
     */
    virtual bool onGamepadEvent(const GamepadEvent& gamepadEvent) { return false; }

    /**
     * Called every time the gamepad state has changed.
     * @param[in] gamepadState The gamepad state.
     * @return true if the state was consumed by the callback, otherwise false.
     */
    virtual bool onGamepadState(const GamepadState& gamepadState) { return false; }

    /**
     * Called when a file is dropped into the window.
     */
    virtual void onDroppedFile(const std::filesystem::path& path) {}

    /**
     * Get the Settings object for Options and Attributes.
     */
    const Settings& getSettings() const;

    /**
     * Get the Settings object for Options and Attributes, accessible for writing.
     * Should only be done by input-parsers, whatever they might be.
     */
    Settings& getSettings();

    /**
     * Get the GPU device for this application.
     */
    const ref<Device>& getDevice() const { return mpDevice; }

    /**
     * Get the render-context for the current frame. This might change each frame.
     */
    RenderContext* getRenderContext() const { return mpDevice ? mpDevice->getRenderContext() : nullptr; }

    /**
     * Get the current FBO.
     */
    const ref<Fbo>& getTargetFbo() const { return mpTargetFBO; }

    /**
     * Get the window.
     */
    Window* getWindow() { return mpWindow.get(); }

    /**
     * Get the progress bar.
     */
    ProgressBar& getProgressBar() { return mProgressBar; }

    /**
     * Get the text renderer.
     */
    TextRenderer& getTextRenderer() { return *mpTextRenderer; }

    /**
     * Get the console.
     */
    Console& getConsole() { return mConsole; }

    /**
     * Get the global Clock object.
     */
    Clock& getGlobalClock() { return mClock; }
    const Clock& getGlobalClock() const { return mClock; }

    /**
     * Get the global FrameRate object.
     */
    FrameRate& getFrameRate() { return mFrameRate; }
    const FrameRate& getFrameRate() const { return mFrameRate; }

    /**
     * Resize the main frame buffer.
     */
    void resizeFrameBuffer(uint32_t width, uint32_t height);

    /**
     * Render a frame.
     */
    void renderFrame();

    /**
     * Get the global input state.
     */
    const InputState& getInputState() { return mInputState; }

    /**
     * Show/hide the UI.
     */
    void toggleUI(bool showUI) { mShowUI = showUI; }

    /**
     * Show/hide the UI.
     */
    bool isUiEnabled() { return mShowUI; }

    /**
     * Pause/resume the renderer. The GUI will still be rendered.
     */
    void pauseRenderer(bool pause) { mRendererPaused = pause; }

    /**
     * Check if the renderer running.
     */
    bool isRendererPaused() { return mRendererPaused; }

    /**
     * Shutdown the app.
     * @param returnCode Return code to be returned from run().
     */
    void shutdown(int returnCode = 0);

    /**
     * Get the current configuration.
     */
    SampleAppConfig getConfig() const;

    /**
     * Render the global UI. You'll can open a GUI window yourself before calling it.
     */
    void renderGlobalUI(Gui* pGui);

    /**
     * Set VSYNC.
     */
    void toggleVsync(bool on) { mVsyncOn = on; }

    /**
     * Get the VSYNC state.
     */
    bool isVsyncEnabled() { return mVsyncOn; }

    /**
     * Get the global shortcuts message.
     */
    static std::string getKeyboardShortcutsStr();

private:
    // Implementation of IWindow::Callbacks

    void handleWindowSizeChange() override;
    void handleRenderFrame() override;
    void handleKeyboardEvent(const KeyboardEvent& keyEvent) override;
    void handleMouseEvent(const MouseEvent& mouseEvent) override;
    void handleGamepadEvent(const GamepadEvent& gamepadEvent) override;
    void handleGamepadState(const GamepadState& gamepadState) override;
    void handleDroppedFile(const std::filesystem::path& path) override;

    // Private functions
    void resizeTargetFBO(uint32_t width, uint32_t height);
    void initUI();
    void saveConfigToFile();

    void captureScreen(Texture* pTexture);

    void renderUI();

    void runInternal();

    void startScripting();
    void registerScriptBindings(pybind11::module& m);

    ref<Device> mpDevice;              ///< GPU device.
    ref<Window> mpWindow;              ///< Main window (nullptr if headless).
    ref<Swapchain> mpSwapchain;        ///< Main swapchain (nullptr if headless).
    ref<Fbo> mpTargetFBO;              ///< FBO available to renderers.
    ref<Texture> mpPausedRenderOutput; ///< Contains the renderer output during pausing.

    ProgressBar mProgressBar;
    std::unique_ptr<Gui> mpGui;
    std::unique_ptr<TextRenderer> mpTextRenderer;
    InputState mInputState;
    std::unique_ptr<ProfilerUI> mpProfilerUI;
    std::unique_ptr<PixelZoom> mpPixelZoom;

    Console mConsole;
    FrameRate mFrameRate;
    Clock mClock;

    bool mShouldTerminate = false; ///< True if application should terminate.
    bool mRendererPaused = false;  ///< True if rendering is paused.
    bool mVsyncOn = false;
    bool mShowUI = true;
    bool mCaptureScreen = false;

    int mReturnCode = 0;

    SampleApp(const SampleApp&) = delete;
    SampleApp& operator=(const SampleApp&) = delete;
};
}; // namespace Falcor

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
#include "Core/Object.h"
#include "Core/API/Device.h"
#include "Core/API/Swapchain.h"
#include "Core/API/Formats.h"
#include "RenderGraph/RenderGraph.h"
#include "Scene/Scene.h"
#include "Scene/SceneBuilder.h"
#include "Utils/Image/ImageProcessing.h"
#include "Utils/Timing/FrameRate.h"
#include "Utils/UI/PythonUI.h"
#include "Utils/Timing/Clock.h"
#include <memory>
#include <filesystem>

namespace Falcor
{

class ProfilerUI;

/// Falcor testbed application class.
/// This is the main Falcor application available through the Python API.
class Testbed : public Object, private Window::ICallbacks
{
    FALCOR_OBJECT(Testbed)
public:
    struct Options
    {
        // Note: Empty constructor needed for clang due to the use of the nested struct constructor in the parent constructor.
        Options() {}
        ref<Device> pDevice = nullptr;
        /// Desc for creating a new device if an existing device isn't already available.
        Device::Desc deviceDesc;
        Window::Desc windowDesc;
        bool createWindow = false;

        /// Color format of the frame buffer.
        ResourceFormat colorFormat = ResourceFormat::BGRA8UnormSrgb;
        /// Depth buffer format of the frame buffer.
        ResourceFormat depthFormat = ResourceFormat::D32Float;
    };

    static ref<Testbed> create(const Options& options) { return make_ref<Testbed>(options); }

    Testbed(const Options& options = Options());
    virtual ~Testbed();

    const ref<Device>& getDevice() const { return mpDevice; }

    /// Run the main loop.
    /// This only returns if the application window is closed or the main loop is interrupted by calling interrupt().
    void run();

    /// Interrupt the main loop.
    void interrupt();

    /// Close the testbed.
    /// This results in `shouldClose` to return true on the next call.
    void close();

    /// Render a single frame.
    /// Note: This is called repeatadly when running the main loop.
    void frame();

    /// Resize the main frame buffer.
    void resizeFrameBuffer(uint32_t width, uint32_t height);

    /// Load a scene.
    void loadScene(const std::filesystem::path& path, SceneBuilder::Flags buildFlags = SceneBuilder::Flags::Default);

    /// Load a scene from a string.
    void loadSceneFromString(
        const std::string& sceneStr,
        const std::string extension = "pyscene",
        SceneBuilder::Flags buildFlags = SceneBuilder::Flags::Default
    );

    ref<Scene> getScene() const;
    Clock& getClock();

    /// Create a new render graph.
    ref<RenderGraph> createRenderGraph(const std::string& name = "");

    /// Load a render graph from a file.
    ref<RenderGraph> loadRenderGraph(const std::filesystem::path& path);

    /// Set the active render graph.
    void setRenderGraph(const ref<RenderGraph>& graph);

    /// Get the active render graph.
    const ref<RenderGraph>& getRenderGraph() const;

    /// Get the python UI screen.
    const ref<python_ui::Screen>& getScreen() const { return mpScreen; }

    /// Capture the output of a render graph and write it to a file.
    void captureOutput(const std::filesystem::path& path, uint32_t outputIndex = 0);

    /// Set the texture to be rendered on the main window (overrides render graph output).
    void setRenderTexture(const ref<Texture>& texture) { mpRenderTexture = texture; }

    /// Get the texture to be rendered on the main window.
    const ref<Texture>& getRenderTexture() const { return mpRenderTexture; }

    /// Set the UI visibility.
    void setShowUI(bool showUI) { mUI.showUI = showUI; }

    /// Get the UI visibility.
    bool getShowUI() const { return mUI.showUI; }

    /// Returns true if the application should terminate.
    /// This is true if the window was closed or escape was pressed.
    bool shouldClose() const { return mShouldClose || (mpWindow && mpWindow->shouldClose()); }

private:
    // Implementation of Window::ICallbacks

    void handleWindowSizeChange() override;
    void handleRenderFrame() override;
    void handleKeyboardEvent(const KeyboardEvent& keyEvent) override;
    void handleMouseEvent(const MouseEvent& mouseEvent) override;
    void handleGamepadEvent(const GamepadEvent& gamepadEvent) override;
    void handleGamepadState(const GamepadState& gamepadState) override;
    void handleDroppedFile(const std::filesystem::path& path) override;

    void internalInit(const Options& options);
    void internalShutdown();

    void resizeTargetFBO(uint32_t width, uint32_t height);

    void renderUI();

    ref<Device> mpDevice;
    ref<Window> mpWindow;
    ref<Swapchain> mpSwapchain;
    ref<Fbo> mpTargetFBO;
    std::unique_ptr<Gui> mpGui;
    ref<python_ui::Screen> mpScreen;
    std::unique_ptr<ProfilerUI> mpProfilerUI;

    ref<Scene> mpScene;
    ref<RenderGraph> mpRenderGraph;
    ref<Texture> mpRenderTexture;

    std::unique_ptr<ImageProcessing> mpImageProcessing;

    FrameRate mFrameRate;
    Clock mClock;

    bool mShouldInterrupt{false};
    bool mShouldClose{false};
    struct
    {
        bool showUI = true;
        bool showFPS = true;
    } mUI;
};

} // namespace Falcor

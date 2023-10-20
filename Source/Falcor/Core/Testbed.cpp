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
#include "Testbed.h"
#include "Core/ObjectPython.h"
#include "Core/AssetResolver.h"
#include "Core/Program/ProgramManager.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Threading.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/Timing/ProfilerUI.h"
#include "Utils/UI/Gui.h"
#include "Utils/UI/InputTypes.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include <imgui.h>

namespace Falcor
{

Testbed::Testbed(const Options& options)
{
    internalInit(options);
}

Testbed::~Testbed()
{
    internalShutdown();
}

void Testbed::run()
{
    mShouldInterrupt = false;

    while ((!mpWindow || !mpWindow->shouldClose()) && !mShouldInterrupt)
        frame();
}

void Testbed::interrupt()
{
    mShouldInterrupt = true;
}

void Testbed::close()
{
    mShouldClose = true;
}

void Testbed::frame()
{
    FALCOR_ASSERT(mpDevice);

    mClock.tick();
    mFrameRate.newFrame();

    if (mpWindow)
        mpWindow->pollForEvents();

    RenderContext* pRenderContext = mpDevice->getRenderContext();

    // Clear the frame buffer.
    const float4 clearColor(1, 0, 1, 1);
    pRenderContext->clearFbo(mpTargetFBO.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    // Compile the render graph.
    if (mpRenderGraph)
        mpRenderGraph->compile(pRenderContext);

    // Update the scene.
    if (mpScene)
    {
        Scene::UpdateFlags sceneUpdates = mpScene->update(pRenderContext, mClock.getTime());
        if (mpRenderGraph && sceneUpdates != Scene::UpdateFlags::None)
            mpRenderGraph->onSceneUpdates(pRenderContext, sceneUpdates);
    }

    // Execute the render graph.
    if (mpRenderGraph)
    {
        mpRenderGraph->getPassesDictionary()[kRenderPassRefreshFlags] = RenderPassRefreshFlags::None;
        mpRenderGraph->execute(pRenderContext);

        // Blit main graph output to frame buffer.
        if (mpRenderGraph->getOutputCount() > 0)
        {
            ref<Texture> pOutTex = mpRenderGraph->getOutput(0)->asTexture();
            FALCOR_ASSERT(pOutTex);
            pRenderContext->blit(pOutTex->getSRV(), mpTargetFBO->getRenderTargetView(0));
        }
    }

    // Blit the current render texture if set.
    if (mpRenderTexture)
    {
        pRenderContext->blit(mpRenderTexture->getSRV(), mpTargetFBO->getRenderTargetView(0));
    }

    renderUI();

#if FALCOR_ENABLE_PROFILER
    mpDevice->getProfiler()->endFrame(pRenderContext);
#endif

    // Copy framebuffer to swapchain image.
    if (mpSwapchain)
    {
        int imageIndex = mpSwapchain->acquireNextImage();
        FALCOR_ASSERT(imageIndex >= 0 && imageIndex < (int)mpSwapchain->getDesc().imageCount);
        Texture* pSwapchainImage = mpSwapchain->getImage(imageIndex).get();
        pRenderContext->copyResource(pSwapchainImage, mpTargetFBO->getColorTexture(0).get());
        pRenderContext->resourceBarrier(pSwapchainImage, Resource::State::Present);
        pRenderContext->submit();
        mpSwapchain->present();
    }

    mpDevice->endFrame();
}

void Testbed::resizeFrameBuffer(uint32_t width, uint32_t height)
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

void Testbed::loadScene(const std::filesystem::path& path, SceneBuilder::Flags buildFlags)
{
    mpScene = SceneBuilder(mpDevice, path, Settings(), buildFlags).getScene();

    if (mpRenderGraph)
        mpRenderGraph->setScene(mpScene);
}

void Testbed::loadSceneFromString(const std::string& scene, const std::string extension, SceneBuilder::Flags buildFlags)
{
    mpScene = SceneBuilder(mpDevice, scene.data(), scene.length(), extension, Settings(), buildFlags).getScene();

    if (mpRenderGraph)
        mpRenderGraph->setScene(mpScene);
}

ref<Scene> Testbed::getScene() const
{
    return mpScene;
}

Clock& Testbed::getClock()
{
    return mClock;
}

ref<RenderGraph> Testbed::createRenderGraph(const std::string& name)
{
    return RenderGraph::create(mpDevice, name);
}

ref<RenderGraph> Testbed::loadRenderGraph(const std::filesystem::path& path)
{
    return RenderGraph::createFromFile(mpDevice, path);
}

void Testbed::setRenderGraph(const ref<RenderGraph>& graph)
{
    mpRenderGraph = graph;

    if (mpRenderGraph)
    {
        mpRenderGraph->onResize(mpTargetFBO.get());
        mpRenderGraph->setScene(mpScene);
    }
}

const ref<RenderGraph>& Testbed::getRenderGraph() const
{
    return mpRenderGraph;
}

// Implementation of Window::ICallbacks

void Testbed::handleWindowSizeChange()
{
    FALCOR_ASSERT(mpDevice && mpWindow && mpSwapchain);

    // Tell the device to resize the swap chain
    auto newSize = mpWindow->getClientAreaSize();
    uint32_t width = newSize.x;
    uint32_t height = newSize.y;

    mpSwapchain->resize(width, height);

    resizeTargetFBO(width, height);
}

void Testbed::handleRenderFrame() {}

void Testbed::handleKeyboardEvent(const KeyboardEvent& keyEvent)
{
    if (mpGui->onKeyboardEvent(keyEvent))
        return;

    if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        switch (keyEvent.key)
        {
        case Input::Key::Escape:
            interrupt();
            close();
            break;
        case Input::Key::F2:
            mUI.showUI = !mUI.showUI;
            break;
        case Input::Key::F5:
            mpDevice->getProgramManager()->reloadAllPrograms();
            break;
        case Input::Key::P:
            mpDevice->getProfiler()->setEnabled(!mpDevice->getProfiler()->isEnabled());
            break;
        }
    }

    if (mpRenderGraph && mpRenderGraph->onKeyEvent(keyEvent))
        return;
    if (mpScene && mpScene->onKeyEvent(keyEvent))
        return;
}

void Testbed::handleMouseEvent(const MouseEvent& mouseEvent)
{
    if (mpGui->onMouseEvent(mouseEvent))
        return;
    if (mpRenderGraph && mpRenderGraph->onMouseEvent(mouseEvent))
        return;
    if (mpScene && mpScene->onMouseEvent(mouseEvent))
        return;
}

void Testbed::handleGamepadEvent(const GamepadEvent& gamepadEvent)
{
    if (mpScene && mpScene->onGamepadEvent(gamepadEvent))
        return;
}

void Testbed::handleGamepadState(const GamepadState& gamepadState)
{
    if (mpScene && mpScene->onGamepadState(gamepadState))
        return;
}

void Testbed::handleDroppedFile(const std::filesystem::path& path) {}

// Internal

void Testbed::internalInit(const Options& options)
{
    OSServices::start();
    Threading::start();

    // Setup asset search paths.
    AssetResolver& resolver = AssetResolver::getDefaultResolver();
    resolver.addSearchPath(getProjectDirectory() / "media");
    for (auto& path : Settings::getGlobalSettings().getSearchDirectories("media"))
        resolver.addSearchPath(path);

    // Create the device.
    if (options.pDevice)
        mpDevice = options.pDevice;
    else
        mpDevice = make_ref<Device>(options.deviceDesc);

    // Create the window & swapchain.
    if (options.createWindow)
    {
        mpWindow = Window::create(options.windowDesc, this);
        mpWindow->setWindowIcon(getRuntimeDirectory() / "data/framework/nvidia.ico");

        Swapchain::Desc desc;
        desc.format = options.colorFormat;
        desc.width = mpWindow->getClientAreaSize().x;
        desc.height = mpWindow->getClientAreaSize().y;
        desc.imageCount = 3;
        desc.enableVSync = options.windowDesc.enableVSync;
        mpSwapchain = make_ref<Swapchain>(mpDevice, desc, mpWindow->getApiHandle());
    }

    // Create target frame buffer
    uint2 fboSize = mpWindow ? mpWindow->getClientAreaSize() : uint2(options.windowDesc.width, options.windowDesc.height);
    mpTargetFBO = Fbo::create2D(mpDevice, fboSize.x, fboSize.y, options.colorFormat, options.depthFormat);

    // Create the GUI.
    mpGui = std::make_unique<Gui>(mpDevice, mpTargetFBO->getWidth(), mpTargetFBO->getHeight(), getDisplayScaleFactor());

    // Create python UI screen.
    mpScreen = make_ref<python_ui::Screen>();

    mFrameRate.reset();
}

void Testbed::internalShutdown()
{
    mpProfilerUI.reset();

    mpImageProcessing.reset();
    mpRenderGraph.reset();
    mpScene.reset();

    if (mpDevice)
        mpDevice->wait();

    Threading::shutdown();

    mpScreen.reset();
    mpGui.reset();
    mpTargetFBO.reset();

    mpSwapchain.reset();
    mpWindow.reset();
    mpDevice.reset();
#ifdef _DEBUG
    Device::reportLiveObjects();
#endif

    OSServices::stop();
}

void Testbed::resizeTargetFBO(uint32_t width, uint32_t height)
{
    // Resize target frame buffer.
    auto pPrevFBO = mpTargetFBO;
    mpTargetFBO = Fbo::create2D(mpDevice, width, height, pPrevFBO->getDesc());
    mpDevice->getRenderContext()->blit(pPrevFBO->getColorTexture(0)->getSRV(), mpTargetFBO->getRenderTargetView(0));

    if (mpGui)
        mpGui->onWindowResize(width, height);

    if (mpRenderGraph)
        mpRenderGraph->onResize(mpTargetFBO.get());

    if (mpScene)
        mpScene->setCameraAspectRatio(width / (float)height);
}

void Testbed::renderUI()
{
    RenderContext* pRenderContext = mpDevice->getRenderContext();
    Profiler* pProfiler = mpDevice->getProfiler();

    FALCOR_PROFILE(pRenderContext, "renderUI");

    mpGui->beginFrame();

    // Help screen.
    {
        if (!ImGui::IsPopupOpen("##Help") && ImGui::IsKeyPressed(ImGuiKey_F1))
            ImGui::OpenPopup("##Help");

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(50, 50));
        if (ImGui::BeginPopupModal("##Help", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration))
        {
            ImGui::Text(
                "Help\n"
                "\n"
                "ESC - Exit (or return to Python interpreter)\n"
                "F1  - Show this help screen\n"
                "F2  - Show/hide UI\n"
                "F5  - Reload shaders\n"
                "P   - Enable/disable profiler\n"
                "\n"
            );

            if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape))
                ImGui::CloseCurrentPopup();

            ImGui::EndPopup();
        }
        ImGui::PopStyleVar();
    }

    if (mUI.showUI)
    {
        // FPS display.
        if (mUI.showFPS)
        {
            Gui::Window w(
                mpGui.get(),
                "##FPS",
                {0, 0},
                {10, 10},
                Gui::WindowFlags::AllowMove | Gui::WindowFlags::AutoResize | Gui::WindowFlags::SetFocus
            );
            w.text(mFrameRate.getMsg());
        }

        if (mpRenderGraph)
        {
            Gui::Window w(mpGui.get(), "Render Graph", {300, 300}, {10, 50});
            mpRenderGraph->renderUI(pRenderContext, w);
        }

        if (mpScene)
        {
            Gui::Window w(mpGui.get(), "Scene", {300, 300}, {10, 360});
            mpScene->renderUI(w);
        }

        // Render Python UI.
        mpScreen->render();
    }

    // Profiler.
    {
        if (pProfiler->isEnabled())
        {
            bool open = pProfiler->isEnabled();
            Gui::Window profilerWindow(mpGui.get(), "Profiler", open, {800, 350}, {10, 10});
            pProfiler->endEvent(pRenderContext, "renderUI"); // Suspend renderUI profiler event

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
    }

    mpGui->render(pRenderContext, mpTargetFBO, (float)mFrameRate.getLastFrameTime());
}

void Testbed::captureOutput(const std::filesystem::path& path, uint32_t outputIndex)
{
    if (!mpImageProcessing)
        mpImageProcessing = std::make_unique<ImageProcessing>(mpDevice);

    RenderContext* pRenderContext = mpDevice->getRenderContext();

    const std::string outputName = mpRenderGraph->getOutputName(outputIndex);
    const ref<Texture> pOutput = mpRenderGraph->getOutput(outputName)->asTexture();
    if (!pOutput)
        FALCOR_THROW("Graph output {} is not a texture", outputName);

    const ResourceFormat format = pOutput->getFormat();
    const uint32_t channels = getFormatChannelCount(format);

    for (auto mask : mpRenderGraph->getOutputMasks(outputIndex))
    {
        // Determine output color channels and filename suffix.
        std::string suffix;
        uint32_t outputChannels = 0;

        switch (mask)
        {
        case TextureChannelFlags::Red:
            suffix = ".R";
            outputChannels = 1;
            break;
        case TextureChannelFlags::Green:
            suffix = ".G";
            outputChannels = 1;
            break;
        case TextureChannelFlags::Blue:
            suffix = ".B";
            outputChannels = 1;
            break;
        case TextureChannelFlags::Alpha:
            suffix = ".A";
            outputChannels = 1;
            break;
        case TextureChannelFlags::RGB: /* No suffix */
            outputChannels = 3;
            break;
        case TextureChannelFlags::RGBA:
            suffix = ".RGBA";
            outputChannels = 4;
            break;
        default:
            logWarning("Graph output {} mask {:#x} is not supported. Skipping.", outputName, (uint32_t)mask);
            continue;
        }

        // Copy relevant channels into new texture if necessary.
        ref<Texture> pTex = pOutput;
        if (outputChannels == 1 && channels > 1)
        {
            // Determine output format.
            ResourceFormat outputFormat = ResourceFormat::Unknown;
            uint bits = getNumChannelBits(format, mask);

            switch (getFormatType(format))
            {
            case FormatType::Unorm:
            case FormatType::UnormSrgb:
                if (bits == 8)
                    outputFormat = ResourceFormat::R8Unorm;
                else if (bits == 16)
                    outputFormat = ResourceFormat::R16Unorm;
                break;
            case FormatType::Snorm:
                if (bits == 8)
                    outputFormat = ResourceFormat::R8Snorm;
                else if (bits == 16)
                    outputFormat = ResourceFormat::R16Snorm;
                break;
            case FormatType::Uint:
                if (bits == 8)
                    outputFormat = ResourceFormat::R8Uint;
                else if (bits == 16)
                    outputFormat = ResourceFormat::R16Uint;
                else if (bits == 32)
                    outputFormat = ResourceFormat::R32Uint;
                break;
            case FormatType::Sint:
                if (bits == 8)
                    outputFormat = ResourceFormat::R8Int;
                else if (bits == 16)
                    outputFormat = ResourceFormat::R16Int;
                else if (bits == 32)
                    outputFormat = ResourceFormat::R32Int;
                break;
            case FormatType::Float:
                if (bits == 16)
                    outputFormat = ResourceFormat::R16Float;
                else if (bits == 32)
                    outputFormat = ResourceFormat::R32Float;
                break;
            }

            if (outputFormat == ResourceFormat::Unknown)
            {
                logWarning("Graph output {} mask {:#x} failed to determine output format. Skipping.", outputName, (uint32_t)mask);
                continue;
            }

            // If extracting a single R, G or B channel from an SRGB format we may lose some precision in the conversion
            // to a singel channel non-SRGB format of the same bit depth. Issue a warning for this case for now.
            // The alternative would be to convert to a higher-precision monochrome format like R32Float,
            // but then the output image will be in a floating-point format which may be undesirable too.
            if (is_set(mask, TextureChannelFlags::RGB) && isSrgbFormat(format))
            {
                logWarning(
                    "Graph output {} mask {:#x} extracting single RGB channel from SRGB format may lose precision.",
                    outputName,
                    (uint32_t)mask
                );
            }

            // Copy color channel into temporary texture.
            pTex = mpDevice->createTexture2D(
                pOutput->getWidth(),
                pOutput->getHeight(),
                outputFormat,
                1,
                1,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
            );
            mpImageProcessing->copyColorChannel(pRenderContext, pOutput->getSRV(0, 1, 0, 1), pTex->getUAV(), mask);
        }

        // Write output image.
        auto ext = Bitmap::getFileExtFromResourceFormat(pTex->getFormat());
        auto fileformat = Bitmap::getFormatFromFileExtension(ext);
        Bitmap::ExportFlags flags = Bitmap::ExportFlags::None;
        if (mask == TextureChannelFlags::RGBA)
            flags |= Bitmap::ExportFlags::ExportAlpha;

        pTex->captureToFile(0, 0, path, fileformat, flags, false /* async */);
    }
}

FALCOR_SCRIPT_BINDING(Testbed)
{
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Device)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(RenderGraph)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Clock)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Profiler)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Scene)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(SceneBuilder)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(python_ui);

    using namespace pybind11::literals;

    pybind11::class_<Testbed, ref<Testbed>> testbed(m, "Testbed");

    testbed.def(
        pybind11::init(
            [](uint32_t width,
               uint32_t height,
               bool create_window,
               Device::Type device_type,
               uint32_t gpu,
               bool enable_debug_layers,
               bool enable_aftermath,
               ref<Device> device)
            {
                Testbed::Options options;
                options.pDevice = device;
                options.windowDesc.width = width;
                options.windowDesc.height = height;
                options.createWindow = create_window;
                options.deviceDesc.type = device_type;
                options.deviceDesc.gpu = gpu;
                options.deviceDesc.enableDebugLayer = enable_debug_layers;
                options.deviceDesc.enableAftermath = enable_aftermath;
                return Testbed::create(options);
            }
        ),
        "width"_a = 1920,
        "height"_a = 1080,
        "create_window"_a = false,
        "device_type"_a = Device::Type::Default,
        "gpu"_a = 0,
        "enable_debug_layers"_a = false,
        "enable_aftermath"_a = false,
        "device"_a = nullptr
    );
    testbed.def("run", &Testbed::run);
    testbed.def("frame", &Testbed::frame);
    testbed.def("resize_frame_buffer", &Testbed::resizeFrameBuffer, "width"_a, "height"_a);
    testbed.def("load_scene", &Testbed::loadScene, "path"_a, "build_flags"_a = SceneBuilder::Flags::Default);
    testbed.def(
        "load_scene_from_string",
        &Testbed::loadSceneFromString,
        "scene"_a,
        "extension"_a = "pyscene",
        "build_flags"_a = SceneBuilder::Flags::Default
    );
    testbed.def("create_render_graph", &Testbed::createRenderGraph, "name"_a = "");
    testbed.def("load_render_graph", &Testbed::loadRenderGraph, "path"_a);
    testbed.def("capture_output", &Testbed::captureOutput, "path"_a, "output_index"_a = uint32_t(0)); // PYTHONDEPRECATED
    testbed.def_property_readonly("profiler", [](Testbed* pTestbed) { return pTestbed->getDevice()->getProfiler(); });

    testbed.def_property_readonly("device", &Testbed::getDevice);
    testbed.def_property_readonly("scene", &Testbed::getScene);
    testbed.def_property_readonly("clock", &Testbed::getClock); // PYTHONDEPRECATED
    testbed.def_property("render_graph", &Testbed::getRenderGraph, &Testbed::setRenderGraph);
    testbed.def_property("render_texture", &Testbed::getRenderTexture, &Testbed::setRenderTexture);
    testbed.def_property_readonly("screen", &Testbed::getScreen);
    testbed.def_property("show_ui", &Testbed::getShowUI, &Testbed::setShowUI);
    testbed.def_property_readonly("should_close", &Testbed::shouldClose);
}

} // namespace Falcor

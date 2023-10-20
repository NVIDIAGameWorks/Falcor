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
#include "Falcor.h"
#include "Mogwai.h"
#include "MogwaiSettings.h"
#include "GlobalState.h"
#include "Core/AssetResolver.h"
#include "Scene/Importer.h"
#include "RenderGraph/RenderGraphImportExport.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Scripting/Scripting.h"
#include "Utils/Timing/TimeReport.h"
#include "Utils/Settings.h"

#include <args.hxx>

#include <filesystem>
#include <algorithm>

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace Mogwai
{
    namespace
    {
        std::unique_ptr<std::map<std::string, Extension::CreateFunc>> gExtensions; // Map ensures ordering

        const std::string kEditorExecutableName = "RenderGraphEditor";
        const std::string kEditorSwitch = "--editor";
        const std::string kGraphFileSwitch = "--graph-file";
        const std::string kGraphNameSwitch = "--graph-name";

        const std::filesystem::path kAppDataPath = getAppDataDirectory() / "NVIDIA/Falcor/Mogwai.json";
    }

    size_t Renderer::DebugWindow::index = 0;

    Renderer::Renderer(const SampleAppConfig& config, const Options& options)
        : SampleApp(config)
        , mOptions(options)
        , mAppData(kAppDataPath)
    {
        setActivePythonRenderGraphDevice(getDevice());
    }

    Renderer::~Renderer()
    {
        setActivePythonRenderGraphDevice(nullptr);
    }

    void Renderer::extend(Extension::CreateFunc func, const std::string& name)
    {
        if (!gExtensions) gExtensions.reset(new std::map<std::string, Extension::CreateFunc>());
        if (gExtensions->find(name) != gExtensions->end())
        {
            FALCOR_THROW("Extension '{}' is already registered.", name);
        }
        (*gExtensions)[name] = func;
    }

    void Renderer::onShutdown()
    {
        resetEditor();
        getDevice()->wait(); // Need to do that because clearing the graphs will try to release some state objects which might be in use
        mGraphs.clear();
        if (mPipedOutput)
        {
#if FALCOR_WINDOWS
            _pclose(mPipedOutput);
#elif FALCOR_LINUX
            pclose(mPipedOutput);
#endif
        }
    }

    void Renderer::onLoad(RenderContext* pRenderContext)
    {
        // Load all plugins
        PluginManager::instance().loadAllPlugins();

        mpExtensions.push_back(MogwaiSettings::create(this));
        if (gExtensions)
        {
            for (auto& f : (*gExtensions)) mpExtensions.push_back(f.second(this));
            gExtensions.reset();
        }

        auto regBinding = [this](pybind11::module& m) {this->registerScriptBindings(m); };
        ScriptBindings::registerBinding(regBinding);

        // Load script provided via command line.
        if (!mOptions.scriptFile.empty())
        {
            if (mOptions.deferredLoad)
            {
                loadScriptDeferred(mOptions.scriptFile);
            }
            else
            {
                loadScript(mOptions.scriptFile);
            }
            // Add script to recent files only if not in silent mode (which is used during image tests).
            if (!mOptions.silentMode) mAppData.addRecentScript(mOptions.scriptFile);
        }

        // Load pyscene provided via command line.
        if (!mOptions.sceneFile.empty())
        {
            loadScene(mOptions.sceneFile);
            // Add scene to recent files only if not in silent mode (which is used during image tests).
            if (!mOptions.silentMode) mAppData.addRecentScene(mOptions.sceneFile);
        }
    }

    void Renderer::onOptionsChange()
    {
        for (auto& pe : mpExtensions)
            pe->onOptionsChange(getSettings().getOptions());
    }

    RenderGraph* Renderer::getActiveGraph() const
    {
        return mGraphs.size() ? mGraphs[mActiveGraph].pGraph.get() : nullptr;
    }

    void Renderer::onGuiRender(Gui* pGui)
    {
        for (auto& pe : mpExtensions)  pe->renderUI(pGui);
    }

    bool isInVector(const std::vector<std::string>& strVec, const std::string& str)
    {
        return std::find(strVec.begin(), strVec.end(), str) != strVec.end();
    }

    Gui::DropdownList createDropdownFromVec(const std::vector<std::string>& strVec, const std::string& currentLabel)
    {
        Gui::DropdownList dropdown;
        for (size_t i = 0; i < strVec.size(); i++) dropdown.push_back({ (uint32_t)i, strVec[i] });
        return dropdown;
    }

    void Renderer::addDebugWindow()
    {
        DebugWindow window;
        window.windowName = "Debug Window " + std::to_string(DebugWindow::index++);
        window.currentOutput = mGraphs[mActiveGraph].mainOutput;
        markOutput(window.currentOutput);
        mGraphs[mActiveGraph].debugWindows.push_back(window);
    }

    void Renderer::unmarkOutput(const std::string& name)
    {
        auto& graphData = mGraphs[mActiveGraph];
        // Skip the original outputs
        if (isInVector(graphData.originalOutputs, name)) return;

        // Decrease the reference counter
        auto& ref = graphData.graphOutputRefs.at(name);
        ref--;
        if (ref == 0)
        {
            graphData.graphOutputRefs.erase(name);
            graphData.pGraph->unmarkOutput(name);
        }
    }

    void Renderer::markOutput(const std::string& name)
    {
        auto& graphData = mGraphs[mActiveGraph];
        // Skip the original outputs
        if (isInVector(graphData.originalOutputs, name)) return;
        auto& refVec = mGraphs[mActiveGraph].graphOutputRefs;
        refVec[name]++;
        if (refVec[name] == 1) mGraphs[mActiveGraph].pGraph->markOutput(name);
    }

    void Renderer::renderOutputUI(Gui::Widgets& widget, const Gui::DropdownList& dropdown, std::string& selectedOutput)
    {
        uint32_t activeOut = -1;
        for (size_t i = 0; i < dropdown.size(); i++)
        {
            if (dropdown[i].label == selectedOutput)
            {
                activeOut = (uint32_t)i;
                break;
            }
        }

        // This can happen when `showAllOutputs` changes to false, and the chosen output is not an original output. We will force an output change
        bool forceOutputChange = activeOut == -1;
        if (forceOutputChange) activeOut = 0;

        if (widget.dropdown("Output", dropdown, activeOut) || forceOutputChange)
        {
            // Unmark old output, set new output, mark new output
            unmarkOutput(selectedOutput);
            selectedOutput = dropdown[activeOut].label;
            markOutput(selectedOutput);
        }
    }

    bool Renderer::renderDebugWindow(Gui::Widgets& widget, const Gui::DropdownList& dropdown, DebugWindow& data, const uint2& winSize)
    {
        // Get the current output, in case `renderOutputUI()` unmarks it
        ref<Texture> pTex = mGraphs[mActiveGraph].pGraph->getOutput(data.currentOutput)->asTexture();
        std::string label = data.currentOutput + "##" + mGraphs[mActiveGraph].pGraph->getName();
        if (!pTex)
        {
            logWarning("Invalid output resource. Is not a texture. Closing window.");
            return false;
        }

        uint2 debugSize = (uint2)(float2(winSize) * float2(0.4f, 0.55f));
        uint2 debugPos = winSize - debugSize;
        debugPos -= 10u;

        // Display the dropdown
        Gui::Window debugWindow(widget.gui(), data.windowName.c_str(), debugSize, debugPos);
        if (debugWindow.gui())
        {
            if (debugWindow.button("Save To File", true)) Bitmap::saveImageDialog(pTex.get());
            renderOutputUI(widget, dropdown, data.currentOutput);
            debugWindow.separator();

            debugWindow.image(label.c_str(), pTex.get());
            debugWindow.release();
            return true;
        }

        return false;
    }

    void Renderer::eraseDebugWindow(size_t id)
    {
        unmarkOutput(mGraphs[mActiveGraph].debugWindows[id].currentOutput);
        mGraphs[mActiveGraph].debugWindows.erase(mGraphs[mActiveGraph].debugWindows.begin() + id);
    }

    void Renderer::graphOutputsGui(Gui::Widgets& widget)
    {
        ref<RenderGraph> pGraph = mGraphs[mActiveGraph].pGraph;
        if (mGraphs[mActiveGraph].debugWindows.size()) mGraphs[mActiveGraph].showAllOutputs = true;
        auto strVec = mGraphs[mActiveGraph].showAllOutputs ? pGraph->getAvailableOutputs() : mGraphs[mActiveGraph].originalOutputs;
        Gui::DropdownList graphOuts = createDropdownFromVec(strVec, mGraphs[mActiveGraph].mainOutput);

        widget.checkbox("List All Outputs", mGraphs[mActiveGraph].showAllOutputs);
        widget.tooltip("Display every possible output in the render-graph, even if it wasn't explicitly marked as one. If there's a debug window open, you won't be able to uncheck this");

        if (graphOuts.size())
        {
            uint2 dims(getTargetFbo()->getWidth(), getTargetFbo()->getHeight());

            for (size_t i = 0; i < mGraphs[mActiveGraph].debugWindows.size();)
            {
                if (renderDebugWindow(widget, graphOuts, mGraphs[mActiveGraph].debugWindows[i], dims) == false)
                {
                    eraseDebugWindow(i);
                }
                else i++;
            }

            renderOutputUI(widget, graphOuts, mGraphs[mActiveGraph].mainOutput);

            // Render the debug windows *before* adding/removing debug windows
            if (widget.button("Show In Debug Window")) addDebugWindow();
            if (mGraphs[mActiveGraph].debugWindows.size())
            {
                if (widget.button("Close all debug windows"))
                {
                    while (mGraphs[mActiveGraph].debugWindows.size()) eraseDebugWindow(0);
                }
            }
        }
    }

    void Renderer::onDroppedFile(const std::filesystem::path& path)
    {
        std::string ext = getExtensionFromPath(path);
        if (ext == "py")
        {
            loadScript(path);
            mAppData.addRecentScript(path);
        }
        else if (std::any_of(Scene::getFileExtensionFilters().begin(), Scene::getFileExtensionFilters().end(), [&ext](FileDialogFilter f) {return f.ext == ext; }))
        {
            loadScene(path);
            mAppData.addRecentScene(path);
        }
        else
        {
            logWarning("RenderGraphViewer::onDroppedFile() - Unknown file extension '{}'", ext);
        }
    }

    void Renderer::editorFileChangeCB()
    {
        mEditorScript = readFile(mEditorTempPath);
    }

    void Renderer::openEditor()
    {
        bool unmarkOut = (isInVector(mGraphs[mActiveGraph].originalOutputs, mGraphs[mActiveGraph].mainOutput) == false);
        // If the current graph output is not an original output, unmark it
        if (unmarkOut) mGraphs[mActiveGraph].pGraph->unmarkOutput(mGraphs[mActiveGraph].mainOutput);

        mEditorTempPath = getTempFilePath();

        // Save the graph
        RenderGraphExporter::save(mGraphs[mActiveGraph].pGraph, mEditorTempPath);

        // Register an update callback
        monitorFileUpdates(mEditorTempPath, std::bind(&Renderer::editorFileChangeCB, this));

        // Run the process
        std::string commandLineArgs = kEditorSwitch + " " + kGraphFileSwitch + " " + mEditorTempPath.string() + " " + kGraphNameSwitch + " " + mGraphs[mActiveGraph].pGraph->getName();
        mEditorProcess = executeProcess(kEditorExecutableName, commandLineArgs);

        // Mark the output if it's required
        if (unmarkOut) mGraphs[mActiveGraph].pGraph->markOutput(mGraphs[mActiveGraph].mainOutput);
    }

    void Renderer::resetEditor()
    {
        if (mEditorProcess)
        {
            closeSharedFile(mEditorTempPath);
            std::filesystem::remove(mEditorTempPath);
            if (mEditorProcess != kInvalidProcessId)
            {
                terminateProcess(mEditorProcess);
                mEditorProcess = 0;
            }
        }
    }

    void Renderer::setActiveGraph(uint32_t active)
    {
        RenderGraph* pOld = getActiveGraph();
        mActiveGraph = active;
        RenderGraph* pNew = getActiveGraph();

        if (pOld != pNew)
        {
            for (auto& e : mpExtensions) e->activeGraphChanged(pNew, pOld);
        }
    }

    void Renderer::removeGraph(const ref<RenderGraph>& pGraph)
    {
        for (auto& e : mpExtensions) e->removeGraph(pGraph.get());
        size_t i = 0;
        for (; i < mGraphs.size(); i++) if (mGraphs[i].pGraph == pGraph) break;
        FALCOR_ASSERT(i < mGraphs.size());
        mGraphs.erase(mGraphs.begin() + i);
        if (mActiveGraph >= i && mActiveGraph > 0) mActiveGraph--;
        setActiveGraph(mActiveGraph);
    }

    void Renderer::removeGraph(const std::string& graphName)
    {
        auto pGraph = getGraph(graphName);
        FALCOR_CHECK(pGraph, "Can't find a graph named '{}'.", graphName);
        removeGraph(pGraph);
    }

    ref<RenderGraph> Renderer::getGraph(const std::string& graphName) const
    {
        for (const auto& g : mGraphs)
        {
            if (g.pGraph->getName() == graphName) return g.pGraph;
        }
        return nullptr;
    }

    void Renderer::removeActiveGraph()
    {
        if (mGraphs.size()) removeGraph(mGraphs[mActiveGraph].pGraph);
    }

    std::vector<std::string> Renderer::getGraphOutputs(const ref<RenderGraph>& pGraph)
    {
        std::vector<std::string> outputs;
        for (size_t i = 0; i < pGraph->getOutputCount(); i++)
        {
            outputs.push_back(pGraph->getOutputName(i));
        }
        return outputs;
    }

    void Renderer::initGraph(const ref<RenderGraph>& pGraph, GraphData* pData)
    {
        if (!pData)
        {
            mGraphs.push_back({});
            pData = &mGraphs.back();
        }
        GraphData& data = *pData;

        // Setup graph data.
        data = {};
        data.pGraph = pGraph;
        data.pGraph->onResize(getTargetFbo().get());
        data.pGraph->setScene(mpScene);
        if (data.pGraph->getOutputCount() != 0)
        {
            data.mainOutput = data.pGraph->getOutputName(0);
        }

        // Store the original outputs.
        data.originalOutputs = getGraphOutputs(pGraph);

        for (auto& e : mpExtensions) e->addGraph(pGraph.get());
    }

    void Renderer::loadScriptDialog()
    {
        std::filesystem::path path;
        if (openFileDialog(Scripting::kFileExtensionFilters, path))
        {
            loadScriptDeferred(path);
            mAppData.addRecentScript(path);
        }
    }

    void Renderer::loadScriptDeferred(const std::filesystem::path& path)
    {
        mScriptPath = path;
    }

    void Renderer::loadScript(const std::filesystem::path& path)
    {
        FALCOR_ASSERT(!path.empty());

        try
        {
            if (getProgressBar().isActive())
                getProgressBar().show("Loading Configuration");

            // Add script directory to search paths (add it to the front to make it highest priority).
            AssetResolver oldResolver = AssetResolver::getDefaultResolver();
            auto directory = std::filesystem::absolute(path).parent_path();
            AssetResolver::getDefaultResolver().addSearchPath(directory, SearchPathPriority::First);

            Scripting::runScriptFromFile(path);

            // Restore asset resolver.
            AssetResolver::getDefaultResolver() = oldResolver;
        }
        catch (const std::exception& e)
        {
            std::string msg = fmt::format("Error when loading configuration file: {}\n{}", path, e.what());
            if (is_set(getErrorDiagnosticFlags(), ErrorDiagnosticFlags::ShowMessageBoxOnError))
                reportErrorAndContinue(msg);
            else
                FALCOR_THROW(msg);
        }
    }

    void Renderer::saveConfigDialog()
    {
        std::filesystem::path path;
        if (saveFileDialog(Scripting::kFileExtensionFilters, path))
        {
            saveConfig(path);
            mAppData.addRecentScript(path);
        }
    }

    void Renderer::addGraph(const ref<RenderGraph>& pGraph)
    {
        FALCOR_CHECK(pGraph, "Can't add an empty graph");

        // If a graph with the same name already exists, remove it
        GraphData* pGraphData = nullptr;
        for (size_t i = 0; i < mGraphs.size(); i++)
        {
            if (mGraphs[i].pGraph->getName() == pGraph->getName())
            {
                logWarning("Replacing existing graph '{}' with new graph.", pGraph->getName());
                pGraphData = &mGraphs[i];
                break;
            }
        }
        initGraph(pGraph, pGraphData);
    }

    void Renderer::setActiveGraph(const ref<RenderGraph>& pGraph)
    {
        size_t index = 0;
        for (; index < mGraphs.size(); ++index)
            if (mGraphs[index].pGraph == pGraph)
                break;

        if (index == mGraphs.size())
            addGraph(pGraph);

        setActiveGraph((uint32_t)index);
    }

    void Renderer::loadSceneDialog()
    {
        std::filesystem::path path;
        if (openFileDialog(Scene::getFileExtensionFilters(), path))
        {
            loadScene(path);
            mAppData.addRecentScene(path);
        }
    }

    void Renderer::loadScene(std::filesystem::path path, SceneBuilder::Flags buildFlags)
    {
        if (mOptions.useSceneCache) buildFlags |= SceneBuilder::Flags::UseCache;
        if (mOptions.rebuildSceneCache) buildFlags |= SceneBuilder::Flags::RebuildCache;

        while (true)
        {
            try
            {
                TimeReport timeReport;
                setScene(SceneBuilder(getDevice(), path, getSettings(), buildFlags).getScene());
                timeReport.measure("Loading scene (total)");
                timeReport.printToLog();
                return;
            }
            catch (const ImporterError &e)
            {
                std::string msg = fmt::format("Failed to load scene.\n\nError in {}\n\n{}", e.path(), e.what());
                if (is_set(getErrorDiagnosticFlags(), ErrorDiagnosticFlags::ShowMessageBoxOnError))
                {
                    if (reportErrorAndAllowRetry(msg))
                        continue;
                    else
                        break;
                } else {
                    FALCOR_THROW(msg);
                }
            }
        }
    }

    void Renderer::unloadScene()
    {
        setScene(nullptr);
    }

    void Renderer::setScene(const ref<Scene>& pScene)
    {
        mpScene = pScene;

        if (mpScene)
        {
            const auto& pFbo = getTargetFbo();
            float ratio = float(pFbo->getWidth()) / float(pFbo->getHeight());
            mpScene->setCameraAspectRatio(ratio);

            if (mpSampler == nullptr)
            {
                // create common texture sampler
                Sampler::Desc desc;
                desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
                desc.setMaxAnisotropy(8);
                mpSampler = getDevice()->createSampler(desc);
            }
            mpScene->getMaterialSystem().setDefaultTextureSampler(mpSampler);
        }

        for (auto& g : mGraphs)
        {
            g.pGraph->setScene(mpScene);
            g.sceneUpdates = Scene::UpdateFlags::None;
        }
        getGlobalClock().setTime(0);
    }

    ref<Scene> Renderer::getScene() const
    {
        return mpScene;
    }

    void Renderer::applyEditorChanges()
    {
        if (!mEditorProcess) return;
        // If the editor was closed, reset the handles
        if ((mEditorProcess != kInvalidProcessId) && isProcessRunning(mEditorProcess) == false) resetEditor();

        if (mEditorScript.empty()) return;

        // Unmark the current output if it wasn't originally marked
        auto pActiveGraph = mGraphs[mActiveGraph].pGraph;
        bool hasUnmarkedOut = (isInVector(mGraphs[mActiveGraph].originalOutputs, mGraphs[mActiveGraph].mainOutput) == false);
        if (hasUnmarkedOut) pActiveGraph->unmarkOutput(mGraphs[mActiveGraph].mainOutput);

        // Run the scripting
        // TODO: Rendergraph scripts should be executed in an isolated scripting context.
        Scripting::getDefaultContext().setObject("g", pActiveGraph);
        Scripting::runScript(mEditorScript);

        // Update the list of marked outputs
        mGraphs[mActiveGraph].originalOutputs = getGraphOutputs(pActiveGraph);

        // If the output before the update was not initially marked but still exists, re-mark it.
        // If it no longer exists, mark a new output from the list of currently marked outputs.
        if (hasUnmarkedOut && isInVector(pActiveGraph->getAvailableOutputs(), mGraphs[mActiveGraph].mainOutput))
        {
            pActiveGraph->markOutput(mGraphs[mActiveGraph].mainOutput);
        }
        else if (isInVector(mGraphs[mActiveGraph].originalOutputs, mGraphs[mActiveGraph].mainOutput) == false)
        {
            mGraphs[mActiveGraph].mainOutput = mGraphs[mActiveGraph].originalOutputs[0];
        }

        mEditorScript.clear();
    }

    void Renderer::executeActiveGraph(RenderContext* pRenderContext)
    {
        if (mGraphs.empty()) return;

        auto& data = mGraphs[mActiveGraph];
        auto& pGraph = data.pGraph;

        // Notify active graph of any scene updates.
        if (data.sceneUpdates != Scene::UpdateFlags::None)
        {
            pGraph->onSceneUpdates(pRenderContext, data.sceneUpdates);
            data.sceneUpdates = Scene::UpdateFlags::None;
        }

        // Execute graph.
        pGraph->getPassesDictionary()[kRenderPassRefreshFlags] = RenderPassRefreshFlags::None;
        pGraph->execute(pRenderContext);
    }

    void Renderer::beginFrame(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
    {
        for (auto& pe : mpExtensions)  pe->beginFrame(pRenderContext, pTargetFbo);
    }

    void Renderer::endFrame(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
    {
        for (auto& pe : mpExtensions) pe->endFrame(pRenderContext, pTargetFbo);
    }

    void Renderer::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
    {
        if (!mScriptPath.empty())
        {
            auto path = mScriptPath;
            mScriptPath.clear();
            loadScript(path);
        }

        applyEditorChanges();

        if (mActiveGraph < mGraphs.size())
        {
            auto& pGraph = mGraphs[mActiveGraph].pGraph;
            pGraph->compile(pRenderContext);
        }

        beginFrame(pRenderContext, pTargetFbo);

        // Clear frame buffer.
        const float4 clearColor(0.38f, 0.52f, 0.10f, 1);
        pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

        if (mActiveGraph < mGraphs.size())
        {
            auto& pGraph = mGraphs[mActiveGraph].pGraph;

            // Update scene and camera.
            if (mpScene)
            {
                auto sceneUpdates = mpScene->update(pRenderContext, getGlobalClock().getTime());

                // Accumulate scene update flags for each graph.
                // The update flags are passed to the active graph, or accumulated until a graph becomes active to avoid missing updates.
                for (auto& g : mGraphs)
                {
                    g.sceneUpdates |= sceneUpdates;
                }
            }

            executeActiveGraph(pRenderContext);

            // Blit main graph output to frame buffer.
            if (mGraphs[mActiveGraph].mainOutput.size())
            {
                ref<Texture> pOutTex = pGraph->getOutput(mGraphs[mActiveGraph].mainOutput)->asTexture();
                FALCOR_ASSERT(pOutTex);
                pRenderContext->blit(pOutTex->getSRV(), pTargetFbo->getRenderTargetView(0));
            }

            if (getSettings().getOption("PipedOutput:enable", false))
            {
                // DEMO21 Opera -- this specific string should probably disappear
                static std::string defaultFFMPEGCmd("ffmpeg -r 30 -f rawvideo -pix_fmt rgba -s 1920x1080 -i - "
                    "-threads 0 -preset medium -y -pix_fmt yuv420p -crf 20 -vf colorchannelmixer=rr=0:rb=1:br=1:bb=0 output.mp4");
                if (!mPipedOutput)
                {
                    std::string ffmepgCmd = getSettings().getOption("PipedOutput:cmd", defaultFFMPEGCmd);
#if FALCOR_WINDOWS
                    mPipedOutput = _popen(ffmepgCmd.c_str(), "wb");
#elif FALCOR_LINUX
                    mPipedOutput = popen(ffmepgCmd.c_str(), "wb");
#endif

                    if (!mPipedOutput)
                        logError("Failed to create piped output with cmd `{}`. Piped output disabled.", ffmepgCmd);
                }

                if (mPipedOutput)
                {
                    Falcor::ref<Texture> framebufferTexture = pTargetFbo->getColorTexture(0);
                    uint32_t subresource = framebufferTexture->getSubresourceIndex(0, 0);
                    std::vector<uint8_t> framebufferData = pRenderContext->readTextureSubresource(framebufferTexture.get(), subresource);
                    fwrite(&framebufferData[0], 4 * pTargetFbo->getWidth() * pTargetFbo->getHeight(), 1, mPipedOutput);
                }
            }
        }

        endFrame(pRenderContext, pTargetFbo);
    }

    bool Renderer::onMouseEvent(const MouseEvent& mouseEvent)
    {
        for (auto& pe : mpExtensions)
        {
            if (pe->mouseEvent(mouseEvent)) return true;
        }

        if (mGraphs.size()) mGraphs[mActiveGraph].pGraph->onMouseEvent(mouseEvent);
        return mpScene ? mpScene->onMouseEvent(mouseEvent) : false;
    }

    bool Renderer::onKeyEvent(const KeyboardEvent& keyEvent)
    {
        for (auto& pe : mpExtensions)
        {
            if (pe->keyboardEvent(keyEvent)) return true;
        }
        if (mGraphs.size()) mGraphs[mActiveGraph].pGraph->onKeyEvent(keyEvent);
        if (mpScene && mpScene->onKeyEvent(keyEvent)) return true;
        // DEMO21 Opera
        if (mKeyCallback)
        {
            if (keyEvent.type == KeyboardEvent::Type::KeyPressed && mKeyCallback(true, (uint32_t)keyEvent.key)) return true;
            if (keyEvent.type == KeyboardEvent::Type::KeyReleased && mKeyCallback(false, (uint32_t)keyEvent.key)) return true;
        }
        return false;
    }

    bool Renderer::onGamepadEvent(const GamepadEvent& gamepadEvent)
    {
        for (auto& pe : mpExtensions)
        {
            if (pe->gamepadEvent(gamepadEvent)) return true;
        }

        return mpScene ? mpScene->onGamepadEvent(gamepadEvent) : false;
    }

    bool Renderer::onGamepadState(const GamepadState& gamepadState)
    {
        return mpScene ? mpScene->onGamepadState(gamepadState) : false;
    }

    void Renderer::onResize(uint32_t width, uint32_t height)
    {
        for (auto& g : mGraphs)
        {
            g.pGraph->onResize(getTargetFbo().get());
            ref<Scene> graphScene = g.pGraph->getScene();
            if (graphScene) graphScene->setCameraAspectRatio((float)width / (float)height);
        }
        if (mpScene) mpScene->setCameraAspectRatio((float)width / (float)height);
    }

    void Renderer::onHotReload(HotReloadFlags reloaded)
    {
        RenderGraph* pActiveGraph = getActiveGraph();
        if (pActiveGraph) pActiveGraph->onHotReload(reloaded);
    }

    size_t Renderer::findGraph(const std::string_view name)
    {
        for (size_t i = 0; i < mGraphs.size(); i++)
        {
            if (mGraphs[i].pGraph->getName() == name) return i;
        };
        return -1;
    }

    std::string Renderer::getVersionString()
    {
        return "Mogwai " + std::to_string(kMajorVersion) + "." + std::to_string(kMinorVersion);
    }
}

int runMain(int argc, char** argv)
{
    args::ArgumentParser parser("Mogwai render application.");
    parser.helpParams.programName = "Mogwai";
    args::HelpFlag helpFlag(parser, "help", "Display this help menu.", {'h', "help"});
    args::ValueFlag<std::string> deviceTypeFlag(parser, "d3d12|vulkan", "Graphics device type.", {'d', "device-type"});
    args::Flag listGPUsFlag(parser, "", "List available GPUs", {"list-gpus"});
    args::ValueFlag<uint32_t> gpuFlag(parser, "index", "Select specific GPU to use", {"gpu"});
    args::Flag headlessFlag(parser, "", "Start without opening a window and handling user input.", {"headless"});
    args::ValueFlag<std::string> scriptFlag(parser, "path", "Python script file to run.", {'s', "script"});
    args::Flag deferredFlag(parser, "deferred", "The script is loaded deferred.", {"deferred"});
    args::ValueFlag<std::string> sceneFlag(parser, "path", "Scene file (for example, a .pyscene file) to open.", { 'S', "scene" });
    args::ValueFlag<std::string> shaderCacheFlag(parser, "shadercache", "Path to the GFX shader cache.", { "shadercache" });
    args::ValueFlag<std::string> logfileFlag(parser, "path", "File to write log into.", {'l', "logfile"});
    args::ValueFlag<int32_t> verbosityFlag(parser, "verbosity", "Logging verbosity (0=disabled, 1=fatal errors, 2=errors, 3=warnings, 4=infos, 5=debugging)", { 'v', "verbosity" }, 4);
    args::Flag silentFlag(parser, "", "Start without opening a window and handling user input (deprecated: use --headless).", {"silent"});
    args::Flag fullscreenFlag(parser, "", "Start in fullscreen mode instead of windowed.", {"fullscreen"});
    args::ValueFlag<uint32_t> widthFlag(parser, "pixels", "Initial window width.", {"width"});
    args::ValueFlag<uint32_t> heightFlag(parser, "pixels", "Initial window height.", {"height"});
    args::Flag useSceneCacheFlag(parser, "", "Use scene cache to improve scene load times.", {'c', "use-cache"});
    args::Flag rebuildSceneCacheFlag(parser, "", "Rebuild the scene cache.", {"rebuild-cache"});
    args::Flag generateShaderDebugInfoFlag(parser, "", "Generate shader debug info.", {"debug-shaders"});
    args::Flag enableDebugLayerFlag(parser, "", "Enable debug layer (enabled by default in Debug build).", {"enable-debug-layer"});
    args::Flag preciseProgramFlag(parser, "", "Force all slang programs to run in precise mode", { "precise" });

    args::CompletionFlag completionFlag(parser, {"complete"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Completion& e)
    {
        std::cout << e.what();
        return 0;
    }
    catch (const args::Help&)
    {
        std::cout << parser;
        return 0;
    }
    catch (const args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (const args::RequiredError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    int32_t verbosity = args::get(verbosityFlag);

    if (verbosity < 0 || verbosity >= (int32_t)Logger::Level::Count)
    {
        std::cerr << argv[0] << ": invalid verbosity level " << verbosity << std::endl;
        return 1;
    }

    Logger::setVerbosity((Logger::Level)verbosity);

    if (logfileFlag)
    {
        std::string logfile = args::get(logfileFlag);
        Logger::setLogFilePath(logfile);
    }

    SampleAppConfig config;
    if (deviceTypeFlag)
    {
        if (args::get(deviceTypeFlag) == "d3d12")
            config.deviceDesc.type = Device::Type::D3D12;
        else if (args::get(deviceTypeFlag) == "vulkan")
            config.deviceDesc.type = Device::Type::Vulkan;
        else
        {
            std::cerr << "Invalid device type, use 'd3d12' or 'vulkan'" << std::endl;
            return 1;
        }
    }
    if (listGPUsFlag)
    {
        const auto gpus = Device::getGPUs(config.deviceDesc.type);
        for (size_t i = 0; i < gpus.size(); ++i)
            fmt::print("GPU {}: {}\n", i, gpus[i].name);
        return 0;
    }
    if (gpuFlag)
        config.deviceDesc.gpu = args::get(gpuFlag);
    if (headlessFlag)
        config.headless = true;
    if (shaderCacheFlag)
        config.deviceDesc.shaderCachePath = args::get(shaderCacheFlag);
    if (enableDebugLayerFlag)
        config.deviceDesc.enableDebugLayer = true;
    if (generateShaderDebugInfoFlag)
        config.generateShaderDebugInfo = true;
    if (preciseProgramFlag)
        config.shaderPreciseFloat = true;

    config.windowDesc.title = "Mogwai";
    if (widthFlag)
        config.windowDesc.width = args::get(widthFlag);
    if (heightFlag)
        config.windowDesc.height = args::get(heightFlag);
    if (fullscreenFlag)
        config.windowDesc.mode = Window::WindowMode::Fullscreen;
    if (silentFlag)
    {
        logWarning("The --silent flag is deprecated. Use --headless instead.");
        config.headless = true;
    }

    Mogwai::Renderer::Options options;
    if (scriptFlag) options.scriptFile = args::get(scriptFlag);
    if (deferredFlag) options.deferredLoad = true;
    if (sceneFlag) options.sceneFile = args::get(sceneFlag);
    if (silentFlag) options.silentMode = true;
    if (useSceneCacheFlag) options.useSceneCache = true;
    if (rebuildSceneCacheFlag) options.rebuildSceneCache = true;

    Mogwai::Renderer renderer(config, options);
    return renderer.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}

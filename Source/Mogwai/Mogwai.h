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
#pragma once
#include "Falcor.h"
#include "AppData.h"
#include "RenderGraph/RenderGraph.h"

namespace Falcor
{
    class Properties;
}

using namespace Falcor;

namespace Mogwai
{
    class Renderer;
    class Extension
    {
    public:
        using UniquePtr = std::unique_ptr<Extension>;
        virtual ~Extension() = default;

        using CreateFunc = UniquePtr(*)(Renderer* pRenderer);

        virtual const std::string& getName() const { return mName; }
        virtual void beginFrame(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo) {};
        virtual void endFrame(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo) {};
        virtual bool hasWindow() const { return false; }
        virtual bool isWindowShown() const { return false; }
        virtual void toggleWindow() {}
        virtual void renderUI(Gui* pGui) {};
        virtual bool mouseEvent(const MouseEvent& e) { return false; }
        virtual bool keyboardEvent(const KeyboardEvent& e) { return false; }
        virtual bool gamepadEvent(const GamepadEvent& e) { return false; }
        virtual void registerScriptBindings(pybind11::module& m) {};
        virtual std::string getScriptVar() const { return {}; }
        virtual std::string getScript(const std::string& var) const { return {}; }
        virtual void addGraph(RenderGraph* pGraph) {};
        virtual void setActiveGraph(RenderGraph* pGraph) {};
        virtual void removeGraph(RenderGraph* pGraph) {};
        virtual void activeGraphChanged(RenderGraph* pNewGraph, RenderGraph* pPrevGraph) {};
        virtual void onOptionsChange(const Properties& settings){}

    protected:
        Extension(Renderer* pRenderer, const std::string& name) : mpRenderer(pRenderer), mName(name) {}

        Renderer* mpRenderer;
        std::string mName;
    };

    class Renderer : public IRenderer
    {
    public:
        struct Options
        {
            std::string scriptFile;
            bool deferredLoad = false;
            std::string sceneFile;
            bool silentMode = false;
            bool useSceneCache = false;
            bool rebuildSceneCache = false;
            bool generateShaderDebugInfo = false;
        };

        using KeyCallback = std::function<bool(bool pressed, uint32_t key)>;

        Renderer(const Options& options);

        void onLoad(RenderContext* pRenderContext) override;
        void onOptionsChange() override;
        void onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
        void onResizeSwapChain(uint32_t width, uint32_t height) override;
        bool onKeyEvent(const KeyboardEvent& e) override;
        bool onMouseEvent(const MouseEvent& e) override;
        bool onGamepadEvent(const GamepadEvent& gamepadEvent) override;
        bool onGamepadState(const GamepadState& gamepadState) override;
        void onGuiRender(Gui* pGui) override;
        void onHotReload(HotReloadFlags reloaded) override;
        void onShutdown() override;
        void onDroppedFile(const std::filesystem::path& path) override;
        void loadScriptDialog();
        void loadScriptDeferred(const std::filesystem::path& path);
        void loadScript(const std::filesystem::path& path);
        void saveConfigDialog();
        void saveConfig(const std::filesystem::path& path) const;
        static std::string getVersionString();

        static void extend(Extension::CreateFunc func, const std::string& name);
        const std::vector<Extension::UniquePtr>& getExtensions() const { return mpExtensions; }

        static constexpr uint32_t kMajorVersion = 0;
        static constexpr uint32_t kMinorVersion = 1;

        AppData& getAppData() { return mAppData; }

        RenderGraph* getActiveGraph() const;

        uint32_t getActiveGraphIndex() const { return mActiveGraph; }

        KeyCallback getKeyCallback() const { return mKeyCallback; }
        void setKeyCallback(KeyCallback keyCallback) { mKeyCallback = keyCallback; }

//    private: // MOGWAI
        friend class Extension;

        Options mOptions;

        std::vector<Extension::UniquePtr> mpExtensions;

        struct DebugWindow
        {
            std::string windowName;
            std::string currentOutput;
            static size_t index;
        };

        struct GraphData
        {
            RenderGraph::SharedPtr pGraph;
            std::string mainOutput;
            bool showAllOutputs = false;
            std::vector<std::string> originalOutputs;
            std::vector<DebugWindow> debugWindows;
            std::unordered_map<std::string, uint32_t> graphOutputRefs;
        };

        Scene::SharedPtr mpScene;

        void addGraph(const RenderGraph::SharedPtr& pGraph);
        void setActiveGraph(const RenderGraph::SharedPtr& pGraph);
        void removeGraph(const RenderGraph::SharedPtr& pGraph);
        void removeGraph(const std::string& graphName);
        RenderGraph::SharedPtr getGraph(const std::string& graphName) const;
        void initGraph(const RenderGraph::SharedPtr& pGraph, GraphData* pData);

        void removeActiveGraph();
        void loadSceneDialog();
        void loadScene(std::filesystem::path path, SceneBuilder::Flags buildFlags = SceneBuilder::Flags::Default);
        void unloadScene();
        void setScene(const Scene::SharedPtr& pScene);
        Scene::SharedPtr getScene() const;
        void executeActiveGraph(RenderContext* pRenderContext);
        void beginFrame(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo);
        void endFrame(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo);

        std::vector<std::string> getGraphOutputs(const RenderGraph::SharedPtr& pGraph);
        void graphOutputsGui(Gui::Widgets& widget);
        bool renderDebugWindow(Gui::Widgets& widget, const Gui::DropdownList& dropdown, DebugWindow& data, const uint2& winSize); // Returns false if the window was closed
        void renderOutputUI(Gui::Widgets& widget, const Gui::DropdownList& dropdown, std::string& selectedOutput);
        void addDebugWindow();
        void eraseDebugWindow(size_t id);
        void unmarkOutput(const std::string& name);
        void markOutput(const std::string& name);
        size_t findGraph(std::string_view name);

        AppData mAppData;

        std::vector<GraphData> mGraphs;
        uint32_t mActiveGraph = 0;
        Sampler::SharedPtr mpSampler = nullptr;
        std::filesystem::path mScriptPath;

        // Editor stuff
        void openEditor();
        void resetEditor();
        void editorFileChangeCB();
        void applyEditorChanges();
        void setActiveGraph(uint32_t active);

        static const size_t kInvalidProcessId = -1; // We use this to know that the editor was launching the viewer
        size_t mEditorProcess = 0;
        std::filesystem::path mEditorTempPath;
        std::string mEditorScript;

        KeyCallback mKeyCallback;
        FILE*       mPipedOutput = nullptr;

        // Scripting
        void registerScriptBindings(pybind11::module& m);

        void handleGamepadInput(float deltaTimeSeconds);
    };

#define MOGWAI_EXTENSION(Name)                         \
    struct ExtendRenderer##Name {                      \
        ExtendRenderer##Name()                         \
        {                                              \
            Renderer::extend(Name::create, #Name);     \
        }                                              \
    } gRendererExtensions##Name;
}

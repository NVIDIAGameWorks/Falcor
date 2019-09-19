/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "Falcor.h"
#include "FalcorExperimental.h"

using namespace Falcor;

namespace Mogwai
{
    class Renderer;
    class Extension
    {
    public:
        class Bindings
        {
        public:
            ScriptBindings::Module& getModule() { return mModule; }
            ScriptBindings::Class<Renderer>& getMogwaiClass() { return mMogwai; }
            template<typename T>
            void addGlobalObject(const std::string& name, const T& obj, const std::string& desc)
            {
                if (mGlobalObjects.find(name) != mGlobalObjects.end()) throw std::exception(("Object `" + name + "` already exists").c_str());
                Scripting::getGlobalContext().setObject(name, obj);
                mGlobalObjects[name] = desc;
            }

        private:
            Bindings(ScriptBindings::Module& m, ScriptBindings::Class<Renderer>& c) : mModule(m), mMogwai(c) {}
            friend class Renderer;
            std::unordered_map<std::string, std::string> mGlobalObjects;
            ScriptBindings::Module& mModule;
            ScriptBindings::Class<Renderer>& mMogwai;
        };

        using UniquePtr = std::unique_ptr<Extension>;
        virtual ~Extension() = default;

        using CreateFunc = UniquePtr(*)(Renderer* pRenderer);

        virtual void beginFrame(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo) {};
        virtual void endFrame(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo) {};
        virtual void renderUI(Gui* pGui) {};
        virtual bool mouseEvent(const MouseEvent& e) { return false; }
        virtual bool keyboardEvent(const KeyboardEvent& e) { return false; }
        virtual void scriptBindings(Bindings& bindings) {};
        virtual std::string getScript() { return {}; }
        virtual void addGraph(RenderGraph* pGraph) {};
        virtual void removeGraph(RenderGraph* pGraph) {};
        virtual void activeGraphChanged(RenderGraph* pNewGraph, RenderGraph* pPrevGraph) {};
    };

    class Renderer : public IRenderer
    {
    public:
        void onLoad(RenderContext* pRenderContext) override;
        void onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
        void onResizeSwapChain(uint32_t width, uint32_t height) override;
        bool onKeyEvent(const KeyboardEvent& e) override;
        bool onMouseEvent(const MouseEvent& e) override;
        void onGuiRender(Gui* pGui) override;
        void onDataReload() override;
        void onShutdown() override;
        void onDroppedFile(const std::string& filename) override;
        void loadScriptDialog();
        void loadScript(const std::string& filename);
        void dumpConfig(std::string filename = {}) const;
        static std::string getVersionString();

        static void extend(Extension::CreateFunc func, const std::string& name);

        static constexpr uint32_t kMajorVersion = 0;
        static constexpr uint32_t kMinorVersion = 1;

        RenderGraph* getActiveGraph() const;
//    private: // MOGWAI
        friend class Extension;
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
        void removeGraph(const RenderGraph::SharedPtr& pGraph);
        void removeGraph(const std::string& graphName);
        RenderGraph::SharedPtr getGraph(const std::string& graphName) const;
        void initGraph(const RenderGraph::SharedPtr& pGraph, GraphData* pData);

        void removeActiveGraph();
        void loadScene(std::string filename = "");
        Scene::SharedPtr getScene() const;
        void executeActiveGraph(RenderContext* pRenderContext);
        void startFrame(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo);

        std::vector<std::string> getGraphOutputs(const RenderGraph::SharedPtr& pGraph);
        void graphOutputsGui(Gui::Widgets& widget);
        bool renderDebugWindow(Gui::Widgets& widget, const Gui::DropdownList& dropdown, DebugWindow& data, const uvec2& winSize); // Returns false if the window was closed
        void renderOutputUI(Gui::Widgets& widget, const Gui::DropdownList& dropdown, std::string& selectedOutput);
        void addDebugWindow();
        void eraseDebugWindow(size_t id);
        void unmarkOutput(const std::string& name);
        void markOutput(const std::string& name);
        size_t findGraph(std::string_view name);

        std::vector<GraphData> mGraphs;
        uint32_t mActiveGraph = 0;
        Sampler::SharedPtr mpSampler = nullptr;
        std::string mScriptFilename;

        // Editor stuff
        void openEditor();
        void resetEditor();
        void editorFileChangeCB();
        void applyEditorChanges();
        void setActiveGraph(uint32_t active);

        static const size_t kInvalidProcessId = -1; // We use this to know that the editor was launching the viewer
        size_t mEditorProcess = 0;
        std::string mEditorTempFile;
        std::string mEditorScript;

        // Scripting
        void registerScriptBindings(ScriptBindings::Module& m);
        std::string mGlobalHelpMessage;
    };

#define MOGWAI_EXTENSION(Name)                         \
    struct ExtendRenderer##Name {                      \
        ExtendRenderer##Name()                         \
        {                                              \
            Renderer::extend(Name::create, #Name);     \
        }                                              \
    } gRendererExtensions##Name;

    constexpr char kRendererVar[] = "m"; // MOGWAI do we want to expose it to all the extensions?

    inline std::string filenameString(const std::string& s, bool stripDataDirs = true)
    {
        std::string filename = stripDataDirs ? stripDataDirectories(s) : s;
        std::replace(filename.begin(), filename.end(), '\\', '/');
        return filename;
    }
}

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
#include "RenderGraphEditor.h"
#include "RenderGraph/RenderGraph.h"
#include "RenderGraph/RenderGraphImportExport.h"
#include "RenderGraph/RenderPassLibrary.h"

#include <args.hxx>
#include <imgui.h>
#include <imgui_internal.h>

#include <fstream>
#include <filesystem>

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace
{
    const std::string kViewerExecutableName = "Mogwai";
    const std::string kScriptSwitch = "--script";
    const std::string kDefaultPassIcon = "DefaultPassIcon.png";
}

RenderGraphEditor::RenderGraphEditor(const Options& options)
    : mOptions(options)
    , mCurrentGraphIndex(0)
{
    mNextGraphString.resize(255, 0);
    mCurrentGraphOutput = "";
    mGraphOutputEditString = mCurrentGraphOutput;
    mGraphOutputEditString.resize(255, 0);
}

RenderGraphEditor::~RenderGraphEditor()
{
    if (mViewerProcess)
    {
        terminateProcess(mViewerProcess);
        mViewerProcess = 0;
    }
}

void RenderGraphEditor::onLoad(RenderContext* pRenderContext)
{
    mpDefaultIconTex = Texture::createFromFile(kDefaultPassIcon, false, false);
    if (!mpDefaultIconTex) throw RuntimeError("Failed to load icon");

    loadAllPassLibraries();

    if (!mOptions.graphFile.empty())
    {
        mViewerRunning = true;
        loadGraphsFromFile(mOptions.graphFile, mOptions.graphName);

        if (mOptions.runFromMogwai) mUpdateFilePath = mOptions.graphFile;
    }
    else createNewGraph("DefaultRenderGraph");
}

void RenderGraphEditor::onDroppedFile(const std::filesystem::path& path)
{
    if (hasExtension(path, "py"))
    {
        if (mViewerRunning)
        {
            msgBox("Viewer is running. Please close the viewer before loading a graph file.", MsgBoxType::Ok);
        }
        else loadGraphsFromFile(path);
    }
}

void setUpDockSpace(uint32_t width, uint32_t height)
{
    ImGuiID dockspace_id = ImGui::GetID("RenderGraphEditor");

    if (ImGui::DockBuilderGetNode(dockspace_id) == NULL)
    {
        ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace); // Add empty node
        ImGui::DockBuilderSetNodeSize(dockspace_id, ImVec2((float) width, (float) height));

        ImGuiID dock_id_graph_editor = dockspace_id; // This variable tracks the central node
        ImGuiID dock_id_render_ui = ImGui::DockBuilderSplitNode(dock_id_graph_editor, ImGuiDir_Right, 0.20f, NULL, &dock_id_graph_editor);
        ImGuiID dock_id_editor_settings = ImGui::DockBuilderSplitNode(dock_id_graph_editor, ImGuiDir_Down, 0.25f, NULL, &dock_id_graph_editor);
        ImGuiID dock_id_render_passes = ImGui::DockBuilderSplitNode(dock_id_editor_settings, ImGuiDir_Right, 0.75f, NULL, &dock_id_editor_settings);

        ImGui::DockBuilderDockWindow("Graph Editor", dock_id_graph_editor);
        ImGui::DockBuilderDockWindow("Render UI", dock_id_render_ui);
        ImGui::DockBuilderDockWindow("Graph Editor Settings", dock_id_editor_settings);
        ImGui::DockBuilderDockWindow("Render Passes", dock_id_render_passes);
        ImGui::DockBuilderFinish(dockspace_id);
    }

    // create the main dockspace over the entire editor window
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::SetNextWindowBgAlpha(0.0f);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Render Graph Editor", NULL, window_flags);
    ImGui::PopStyleVar(3);

    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
    ImGui::End();
}

// some of this will need to be moved to render graph ui
void RenderGraphEditor::onGuiRender(Gui* pGui)
{
    RenderContext* pRenderContext = gpFramework->getRenderContext();

    uint32_t screenHeight = mWindowSize.y;
    uint32_t screenWidth = mWindowSize.x;

    setUpDockSpace(screenWidth, screenHeight);

    Gui::MainMenu menu(pGui);
    Gui::Menu::Dropdown fileMenu = menu.dropdown("File");
    if (!mShowCreateGraphWindow && fileMenu.item("Create New Graph"))
    {
        mShowCreateGraphWindow = true;
    }

    if (fileMenu.item("Load File"))
    {
        if (mViewerRunning)
        {
            msgBox("Viewer is running. Please close the viewer before loading a graph file.", MsgBoxType::Ok);
        }
        else
        {
            std::filesystem::path path;
            if (openFileDialog({}, path)) loadGraphsFromFile(path);
        }
    }

    if (fileMenu.item("Save To File"))
    {
        bool saveGraph = true;
        std::string log;

        try
        {
            std::string s;
            mpGraphs[mCurrentGraphIndex]->compile(pRenderContext, s);
        }
        catch (const std::exception&)
        {
            MsgBoxButton msgBoxButton = msgBox("Attempting to save invalid graph.\nGraph may not execute correctly when loaded\nAre you sure you want to save the graph?"
                , MsgBoxType::OkCancel);
            saveGraph = !(msgBoxButton == MsgBoxButton::Cancel);
        }

        if (saveGraph)
        {
            std::filesystem::path path = mOpenGraphNames[mCurrentGraphIndex].label + ".py";
            if (saveFileDialog(RenderGraph::kFileExtensionFilters, path)) serializeRenderGraph(path);
        }
    }

    fileMenu.release();

    Gui::Menu::Dropdown windowMenu = menu.dropdown("Window");
    windowMenu.item("Debug Window", mShowDebugWindow);
    windowMenu.release();

    menu.release();

    // sub window for listing available window passes
    Gui::Window passWindow(pGui, "Render Passes", { screenWidth * 3 / 5, screenHeight / 4 - 20 }, { screenWidth / 5, screenHeight * 3 / 4 + 20 });
    if (mResetGuiWindows)
    {
        passWindow.windowSize(screenWidth * 3 / 5, screenHeight / 4 - 20);
        passWindow.windowPos(screenWidth / 5, screenHeight * 3 / 4 + 20);
    }

    passWindow.columns(5);
    auto renderPasses = RenderPassLibrary::instance().enumerateClasses();
    for (size_t i = 0; i < renderPasses.size(); i++)
    {
        const auto& pass = renderPasses[i];
        passWindow.rect({ 148.0f, 64.0f }, pGui->pickUniqueColor(pass.info.type), false);
        passWindow.image(("RenderPass##" + std::to_string(i)).c_str(), mpDefaultIconTex, { 148.0f, 44.0f });
        passWindow.dragDropSource(pass.info.type.c_str(), "RenderPassType", pass.info.type);
        passWindow.text(pass.info.type);
        passWindow.tooltip(pass.info.desc, true);
        passWindow.nextColumn();
    }

    passWindow.release();

    Gui::Window renderWindow(pGui, "Render UI", { screenWidth * 1 / 5, screenHeight - 20 }, { screenWidth * 4 / 5, 20 });
    if (mResetGuiWindows)
    {
        renderWindow.windowSize(screenWidth * 1 / 5, screenHeight - 20);
        renderWindow.windowPos(screenWidth * 4 / 5, 20);
    }
    renderWindow.release();

    // push a sub GUI window for the node editor
    Gui::Window editorWindow(pGui, "Graph Editor", { screenWidth * 4 / 5, screenHeight * 3 / 4 }, { 0, 20 }, Gui::WindowFlags::SetFocus | Gui::WindowFlags::AllowMove);
    if (mResetGuiWindows)
    {
        editorWindow.windowSize(screenWidth * 4 / 5, screenHeight * 3 / 4);
        editorWindow.windowPos(0, 20);
    }
    mRenderGraphUIs[mCurrentGraphIndex].renderUI(pRenderContext, pGui);
    editorWindow.release();

    for (auto& renderGraphUI : mRenderGraphUIs)
    {
        mCurrentLog += renderGraphUI.getCurrentLog();
        renderGraphUI.clearCurrentLog();
    }

    Gui::Window settingsWindow(pGui, "Graph Editor Settings", { screenWidth / 5, screenHeight / 4 - 20 }, { 0, screenHeight * 3 / 4 + 20 });
    if (mResetGuiWindows)
    {
        settingsWindow.windowSize(screenWidth / 5, screenHeight / 4 - 20);
        settingsWindow.windowPos(0, screenHeight * 3 / 4 + 20);
    }

    uint32_t selection = static_cast<uint32_t>(mCurrentGraphIndex);
    if (mOpenGraphNames.size() && settingsWindow.dropdown("Open Graph", mOpenGraphNames, selection))
    {
        // Display graph
        mCurrentGraphIndex = selection;
    }

    if (!mUpdateFilePath.empty())
    {
        mRenderGraphUIs[mCurrentGraphIndex].writeUpdateScriptToFile(pRenderContext, mUpdateFilePath, (float)gpFramework->getFrameRate().getLastFrameTime());
    }

    if (mViewerRunning && mViewerProcess)
    {
        if (!isProcessRunning(mViewerProcess))
        {
            terminateProcess(mViewerProcess);
            mViewerProcess = 0;
            mViewerRunning = false;
            mUpdateFilePath.clear();
        }
    }

    // validate the graph and output the current status to the console
    if (settingsWindow.button("Validate Graph"))
    {
        std::string s;
        bool valid = mpGraphs[mCurrentGraphIndex]->compile(pRenderContext, s);
        if (valid) s += "The graph is valid";
        else s += std::string("The graph is invalid.");
        msgBox(s);
        mCurrentLog += s;
    }

    auto pScene = mpGraphs[mCurrentGraphIndex]->getScene();
//     if (pScene)
//     {
//         settingsWindow.text((std::string("Graph Sets Scene: ") + pScene->getFilename()).c_str());
//     }

    std::vector<std::string> graphOutputString{ mGraphOutputEditString };
    if (settingsWindow.multiTextbox("Add Output", { "GraphOutput" }, graphOutputString))
    {
        if (mCurrentGraphOutput != mGraphOutputEditString)
        {
            if (mCurrentGraphOutput.size())
            {
                mpGraphs[mCurrentGraphIndex]->unmarkOutput(mCurrentGraphOutput);
            }

            mCurrentGraphOutput = graphOutputString[0];
            mRenderGraphUIs[mCurrentGraphIndex].addOutput(mCurrentGraphOutput);
        }
    }
    mGraphOutputEditString = graphOutputString[0];

    mRenderGraphUIs[mCurrentGraphIndex].setRecordUpdates(mViewerRunning);
    if (!mViewerRunning && settingsWindow.button("Open in Mogwai"))
    {
        std::string log;
        bool openViewer = true;
        try
        {
            std::string s;
            mpGraphs[mCurrentGraphIndex]->compile(pRenderContext, s);
        }
        catch (const std::exception& e)
        {
            openViewer = msgBox(std::string("Graph is invalid :\n ") + e.what() + "\n Are you sure you want to attempt preview?", MsgBoxType::OkCancel) == MsgBoxButton::Ok;
        }

        if (openViewer)
        {
            mUpdateFilePath = getTempFilePath();
            RenderGraphExporter::save(mpGraphs[mCurrentGraphIndex], mUpdateFilePath);

            // load application for the editor given it the name of the mapped file
            std::string commandLineArgs = kScriptSwitch + " " + mUpdateFilePath.string();
            mViewerProcess = executeProcess(kViewerExecutableName, commandLineArgs);
            FALCOR_ASSERT(mViewerProcess);
            mViewerRunning = true;
        }
    }

    settingsWindow.release();

    if (mShowDebugWindow)
    {
        Gui::Window debugWindow(pGui, "output", { screenWidth / 4, screenHeight / 4 - 20 }, { screenWidth * 3 / 4, screenHeight * 3 / 4 + 20 });
        if (mResetGuiWindows)
        {
            debugWindow.windowSize(screenWidth / 4, screenHeight / 4 - 20);
            debugWindow.windowPos(screenWidth * 3 / 4, screenHeight * 3 / 4 + 20);
        }
        renderLogWindow(debugWindow);
        debugWindow.release();
    }

    // pop up window for naming a new render graph
    if (mShowCreateGraphWindow)
    {
        Gui::Window createWindow(pGui, "CreateNewGraph", { 256, 128 }, { screenWidth / 2 - 128, screenHeight / 2 - 64 });
        createWindow.textbox("Graph Name", mNextGraphString);

        if (createWindow.button("Create Graph") && mNextGraphString[0])
        {
            createNewGraph(mNextGraphString);
            mNextGraphString.clear();
            mNextGraphString.resize(255, 0);
            mShowCreateGraphWindow = false;
        }

        if (createWindow.button("Cancel", true))
        {
            mNextGraphString.clear();
            mNextGraphString.resize(255, 0);
            mShowCreateGraphWindow = false;
        }

        createWindow.release();
    }

    mResetGuiWindows = false;
}

void RenderGraphEditor::loadAllPassLibraries()
{
#if FALCOR_WINDOWS
    static const std::string kLibraryExtension = "dll";
#elif FALCOR_LINUX
    static const std::string kLibraryExtension = "so";
#endif

    auto executableDirectory = getExecutableDirectory();

    // iterate through and find all render pass libraries
    for (auto& it : std::filesystem::directory_iterator(executableDirectory))
    {
        const auto& path = it.path();
        if (hasExtension(path, kLibraryExtension))
        {
            // check for addPasses()
            SharedLibraryHandle l = loadSharedLibrary(path);
            if (auto pGetPass = (RenderPassLibrary::LibraryFunc)getProcAddress(l, "getPasses"))
            {
                releaseSharedLibrary(l);
                RenderPassLibrary::instance().loadLibrary(path.filename().string());
            }
        }
    }
}

void RenderGraphEditor::renderLogWindow(Gui::Widgets& widget)
{
    // window for displaying log from render graph validation
    widget.text(mCurrentLog);
}

void RenderGraphEditor::serializeRenderGraph(const std::filesystem::path& path)
{
    RenderGraphExporter::save(mpGraphs[mCurrentGraphIndex], path);
}

void RenderGraphEditor::deserializeRenderGraph(const std::filesystem::path& path)
{
    mpGraphs[mCurrentGraphIndex] = RenderGraphImporter::import(path.string());
    if (mRenderGraphUIs.size() < mCurrentGraphIndex)
    {
        mRenderGraphUIs[mCurrentGraphIndex].setToRebuild();
    }
}

void RenderGraphEditor::loadGraphsFromFile(const std::filesystem::path& path, const std::string& graphName)
{
    FALCOR_ASSERT(!path.empty());

    // behavior is load each graph defined within the file as a separate editor ui
    std::vector <RenderGraph::SharedPtr> newGraphs;
    if (graphName.size())
    {
        auto pGraph = RenderGraphImporter::import(graphName, path);
        if (pGraph) newGraphs.push_back(pGraph);
    }
    else
    {
        newGraphs = RenderGraphImporter::importAllGraphs(path);
    }

    for (const auto& pGraph : newGraphs)
    {
        const std::string& name = pGraph->getName();
        auto nameToIndexIt = mGraphNamesToIndex.find(name);
        if (nameToIndexIt != mGraphNamesToIndex.end())
        {
            MsgBoxButton button = msgBox("Warning! Graph is already open. Update graph from file?", MsgBoxType::YesNo);
            if (button == MsgBoxButton::Yes)
            {
                mCurrentGraphIndex = nameToIndexIt->second;
                mpGraphs[mCurrentGraphIndex]->update(pGraph);
                mRenderGraphUIs[mCurrentGraphIndex].reset();
                continue;
            }
        }
        else
        {
            mCurrentGraphIndex = mpGraphs.size();
            mpGraphs.push_back(pGraph);
            mRenderGraphUIs.push_back(RenderGraphUI(mpGraphs[mCurrentGraphIndex], name));

            Gui::DropdownValue nextGraphID;
            mGraphNamesToIndex.insert(std::make_pair(name, static_cast<uint32_t>(mCurrentGraphIndex)));
            nextGraphID.value = static_cast<int32_t>(mOpenGraphNames.size());
            nextGraphID.label = name;
            mOpenGraphNames.push_back(nextGraphID);
        }
    }
}

void RenderGraphEditor::createNewGraph(const std::string& renderGraphName)
{
    std::string graphName = renderGraphName;
    auto nameToIndexIt = mGraphNamesToIndex.find(graphName);
    RenderGraph::SharedPtr newGraph = RenderGraph::create();

    std::string tempGraphName = graphName;
    while (mGraphNamesToIndex.find(tempGraphName) != mGraphNamesToIndex.end())
    {
        tempGraphName.append("_");
    }
    // Matt TODO can we put the GUI dropdown code in a shared function shared with 'loadFromFile'?
    graphName = tempGraphName;
    newGraph->setName(graphName);
    mCurrentGraphIndex = mpGraphs.size();
    mpGraphs.push_back(newGraph);
    mRenderGraphUIs.push_back(RenderGraphUI(newGraph, graphName));

    Gui::DropdownValue nextGraphID;
    mGraphNamesToIndex.insert(std::make_pair(graphName, static_cast<uint32_t>(mCurrentGraphIndex) ));
    nextGraphID.value = static_cast<int32_t>(mOpenGraphNames.size());
    nextGraphID.label = graphName;
    mOpenGraphNames.push_back(nextGraphID);
}

void RenderGraphEditor::onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const float4 clearColor(0.25, 0.25, 0.25 , 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    mRenderGraphUIs[mCurrentGraphIndex].updateGraph(pRenderContext);
}

void RenderGraphEditor::onResizeSwapChain(uint32_t width, uint32_t height)
{
    for(auto pG : mpGraphs) pG->onResize(gpFramework->getTargetFbo().get());
    mWindowSize = {width, height};
    mResetGuiWindows = true;
}

#if FALCOR_WINDOWS
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    args::ArgumentParser parser("Render graph editor.");
    parser.helpParams.programName = "RenderGraphEditor";
    args::HelpFlag helpFlag(parser, "help", "Display this help menu.", {'h', "help"});
    args::ValueFlag<std::string> graphFileFlag(parser, "path", "Graph file to edit.", {"graph-file"});
    args::ValueFlag<std::string> graphNameFlag(parser, "name", "Graph name to edit.", {"graph-name"});
    args::Flag editorFlag(parser, "", "Run in editor mode (started from Mogwai).", {"editor"});
    args::CompletionFlag completionFlag(parser, {"complete"});

    try
    {
#if FALCOR_WINDOWS
        parser.ParseCLI(__argc, __argv);
#else
        parser.ParseCLI(argc, argv);
#endif
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

    RenderGraphEditor::Options options;

    if (graphFileFlag) options.graphFile = args::get(graphFileFlag);
    if (graphNameFlag) options.graphName = args::get(graphNameFlag);
    if (editorFlag) options.runFromMogwai = true;

    RenderGraphEditor::UniquePtr pEditor = std::make_unique<RenderGraphEditor>(options);
    SampleConfig config;
    config.windowDesc.title = "Render Graph Editor";
    config.windowDesc.resizableWindow = true;
    Sample::run(config, pEditor);
    return 0;
}

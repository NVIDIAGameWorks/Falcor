/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "MogwaiSettings.h"
#include <iomanip>
#include <sstream>

namespace Mogwai
{
    namespace
    {
        void shortcuts()
        {
            std::string s;
            s += " 'F1'   - Show the help message\n";
            s += " 'F9'   - Show/Hide the time\n";
            s += " 'F10'  - Show/Hide the FPS\n";
            s += " 'F11'  - Toggle Main Menu Auto-Hide\n";
            s += "\n" + gpFramework->getKeyboardShortcutsStr();
            msgBox(s);
        }

        void about()
        {
            std::string s = Renderer::getVersionString() + "\n";
            s += "Powered by Falcor ";
            s += FALCOR_VERSION_STRING;
            msgBox(s);
        }

        void showFps(Gui* pGui)
        {
            Gui::Window w(pGui, "##FPS", { 0, 0 }, { 10, 25 }, Gui::WindowFlags::AllowMove | Gui::WindowFlags::AutoResize | Gui::WindowFlags::SetFocus);
            std::string msg = gpFramework->getFrameRate().getMsg(gpFramework->isVsyncEnabled());
            w.text(msg);
        }

        void winSizeUI(Gui::Window& w)
        {
            static const uint2 resolutions[] =
            {
                {1280, 720},
                {1920, 1080},
                {1920, 1200},
                {2560, 1440},
                {3840, 2160},
            };

            constexpr uint32_t kCustomIndex = uint32_t(-1);

            static const auto initDropDown = [=](const uint2 resolutions[], uint32_t count) -> Gui::DropdownList
            {
                Gui::DropdownList list;
                for (uint32_t i = 0; i < count; i++)
                {
                    list.push_back({ i, std::to_string(resolutions[i].x) + "x" + std::to_string(resolutions[i].y) });
                }
                list.push_back({ kCustomIndex, "Custom" });
                return list;
            };

            auto initDropDownVal = [=](const uint2 resolutions[], uint32_t count, uint2 screenDims)
            {
                for (uint32_t i = 0; i < count; i++)
                {
                    if (screenDims == resolutions[i]) return i;
                }
                return kCustomIndex;
            };

            uint2 currentRes = gpFramework->getWindow()->getClientAreaSize();
            static const Gui::DropdownList dropdownList = initDropDown(resolutions, arraysize(resolutions));
            uint32_t currentVal = initDropDownVal(resolutions, arraysize(resolutions), currentRes);
            w.text("Window Size");
            w.tooltip("The Window Size refers to the renderable area size (Swap-Chain dimensions)");

            bool dropdownChanged = w.dropdown("##resdd", dropdownList, currentVal);
            static uint2 customSize;
            static bool forceCustom = false;

            if (dropdownChanged)
            {
                if (currentVal == kCustomIndex)
                {
                    forceCustom = true;
                }
                else
                {
                    customSize = {};
                    gpFramework->resizeSwapChain(resolutions[currentVal].x, resolutions[currentVal].y);
                }
            }

            if (currentVal == kCustomIndex || forceCustom)
            {
                if (customSize.x == 0) customSize = currentRes;

                w.var("##custres", customSize);
                if (w.button("Apply##custres", true))
                {
                    gpFramework->resizeSwapChain(customSize.x, customSize.y);
                    forceCustom = false;
                }
                if (w.button("Cancel##custres", true))
                {
                    customSize = currentRes;
                    forceCustom = false;
                }
            }
        }
    }

    void MogwaiSettings::windowSettings(Gui* pGui)
    {
        Gui::Window w(pGui, "Window", mShowWinSize, { 0, 0 }, { 350, 300 }, Gui::WindowFlags::AllowMove | Gui::WindowFlags::AutoResize | Gui::WindowFlags::ShowTitleBar | Gui::WindowFlags::CloseButton);
        winSizeUI(w);
    }

    void MogwaiSettings::timeSettings(Gui* pGui)
    {
        Gui::Window w(pGui, "Time", mShowTime, { 0, 0 }, { 350, 25 }, Gui::WindowFlags::AllowMove | Gui::WindowFlags::AutoResize | Gui::WindowFlags::ShowTitleBar | Gui::WindowFlags::CloseButton);

        Clock& clock = gpFramework->getGlobalClock();
        clock.renderUI(w);
        w.separator(2);

        double exitTime = clock.getExitTime();
        uint64_t exitFrame = clock.getExitFrame();

        if (exitTime || exitTime)
        {
            std::stringstream s;
            s << "Exiting in ";
            if (exitTime)  s << std::fixed << std::setprecision(2) << (exitTime - clock.getTime()) << " seconds";
            if (exitFrame) s << (exitFrame - clock.getFrame()) << " frames";
            w.text(s.str());
        }
    }

    void MogwaiSettings::graphs(Gui* pGui)
    {
        if (!mShowGraphUI || mpRenderer->mGraphs.empty()) return;

        Gui::Window w(pGui, "Graphs", mShowGraphUI, { 300, 400 }, { 10, 80 }, Gui::WindowFlags::Default);
        if (!mShowGraphUI) return;

        if (mpRenderer->mEditorProcess == 0)
        {
            Gui::DropdownList graphList;
            for (size_t i = 0; i < mpRenderer->mGraphs.size(); i++) graphList.push_back({ (uint32_t)i, mpRenderer->mGraphs[i].pGraph->getName() });
            uint32_t activeGraph = mpRenderer->mActiveGraph;
            if (w.dropdown("Active Graph", graphList, activeGraph))
            {
                mpRenderer->setActiveGraph(activeGraph);
            }

            if (w.button("Edit")) mpRenderer->openEditor();
            if (w.button("Remove", true))
            {
                mpRenderer->removeActiveGraph();
                if (mpRenderer->mGraphs.empty()) return;
            }
            w.separator();
        }

        // Active graph output
        mpRenderer->graphOutputsGui(w); // MOGWAI shouldn't be here

        // Graph UI
        w.separator();
        Gui::Group graphGroup(pGui, (mpRenderer->mGraphs[mpRenderer->mActiveGraph].pGraph->getName() + "##Graph").c_str());
        mpRenderer->mGraphs[mpRenderer->mActiveGraph].pGraph->renderUI(graphGroup);
    }

    void MogwaiSettings::mainMenu(Gui* pGui)
    {
        if (mAutoHideMenu && mMousePosition.y >= 20) return;

        auto m = Gui::MainMenu(pGui);

        {
            auto file = m.dropdown("File");
            if (file.item("Load Script", "Ctrl+O")) mpRenderer->loadScriptDialog();
            if (file.item("Save Config")) mpRenderer->saveConfigDialog();
            if (file.item("Load Scene", "Ctrl+Shift+O")) mpRenderer->loadSceneDialog();
            // if (file.item("Reset Scene")) mpRenderer->setScene(nullptr);
            file.separator();
            if (file.item("Reload Render-Passes", "F5")) RenderPassLibrary::instance().reloadLibraries(gpFramework->getRenderContext());
            file.separator();

            {
                auto &appData = mpRenderer->getAppData();
                auto recentScripts = file.menu("Recent Scripts");
                for (auto path : appData.getRecentScripts())
                {
                    if (recentScripts.item(path))
                    {
                        mpRenderer->loadScriptDeferred(path);
                        appData.addRecentScript(path);
                    }
                }
            }

            {
                auto &appData = mpRenderer->getAppData();
                auto recentScenes = file.menu("Recent Scenes");
                for (auto path : appData.getRecentScenes())
                {
                    if (recentScenes.item(path))
                    {
                        mpRenderer->loadScene(path);
                        appData.addRecentScene(path);
                    }
                }
            }
        }

        {
            auto view = m.dropdown("View");
            view.item("Graph UI", mShowGraphUI, "F6");
            view.item("Auto Hide", mAutoHideMenu, "F11");
            view.item("FPS", mShowFps, "F10");
            view.item("Time", mShowTime, "F9");
            view.item("Window Size", mShowWinSize);
            view.separator();
            for (const auto& ext : mpRenderer->getExtensions())
            {
                bool show = ext->isWindowShown();
                if (ext->hasWindow() && view.item(ext->getName(), show))
                {
                    ext->toggleWindow();
                }
            }
            view.separator();
            view.item("Console", mShowConsole, "`");
        }

        {
            auto help = m.dropdown("Help");
            if (help.item("Shortcuts")) shortcuts();
            if (help.item("About")) about();
        }
    }

    void MogwaiSettings::renderUI(Gui* pGui__)
    {
        Gui* pGui = (Gui*)pGui__;
        mainMenu(pGui);
        graphs(pGui);
        if (mShowFps) showFps(pGui);
        if (mShowTime) timeSettings(pGui);
        if (mShowWinSize) windowSettings(pGui);
        Console::instance().render(pGui__, mShowConsole);
    }

    bool MogwaiSettings::mouseEvent(const MouseEvent& e)
    {
        if (e.type == MouseEvent::Type::Move) mMousePosition = e.screenPos;
        return false;
    }

    bool noMods(InputModifiers m)
    {
        return !(m.isAltDown || m.isCtrlDown || m.isShiftDown);
    }

    bool MogwaiSettings::keyboardEvent(const KeyboardEvent& e)
    {
        if (e.type == KeyboardEvent::Type::KeyPressed)
        {
            if (e.mods.isAltDown) return false;

            // Regular keystrokes
            if (noMods(e.mods))
            {
                switch (e.key)
                {
                case KeyboardEvent::Key::F1:
                    shortcuts();
                    break;
                case KeyboardEvent::Key::F10:
                    mShowFps = !mShowFps;
                    break;
                case KeyboardEvent::Key::F11:
                    mAutoHideMenu = !mAutoHideMenu;
                    break;
                case KeyboardEvent::Key::F6:
                    mShowGraphUI = !mShowGraphUI;
                    break;
                case KeyboardEvent::Key::F9:
                    mShowTime = !mShowTime;
                    break;
                default:
                    return false;
                }
                return true;
            }
            else if (e.mods.isCtrlDown)
            {
                if (e.key == KeyboardEvent::Key::O)
                {
                    e.mods.isShiftDown ? mpRenderer->loadSceneDialog() : mpRenderer->loadScriptDialog();
                    return true;
                }
                else return false;
            }
        }
        return false;
    }

    MogwaiSettings::UniquePtr MogwaiSettings::create(Renderer* pRenderer)
    {
        return UniquePtr(new MogwaiSettings(pRenderer));
    }
}

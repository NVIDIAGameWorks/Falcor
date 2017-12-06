/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/Renderer/Renderer.h"

namespace Falcor
{
    /** A more complex "Sample" that handles virtually all window / GUI stuff internally, so writing a prototype
        is as simple as creating a derived Renderer class (see Renderer/Renderer.h) that defines initialization
        and rendering callbacks (and optionally a handful of other callback).  You can then swap between
        different prototype renderers via a dropdown in the GUI and (ideally) the MultiRendererSample
        encapsulates state (pushes/pops/cleans) so that even buggy prototypes don't screw up code running in
        other Renderers.  (It's also easy to do side-by-side comparisons:  just write a Renderer that takes
        in two other Renderers and displays them side by side.)

        Adds a "Load Scene" button / function that loads Falcor scene files for you.

        There are two methods users of MultiRendererSample object may need: addRenderer() and loadScene().
    */
    class MultiRendererSample : public Sample
    {
    public:
        /** Add a renderer to the list selectable in this application.  First one added runs by default on app load.
            \param[in] pRenderer Shared pointer to a renderer that will be selectable in the application.
            \return An integer identifier specifying location in internal list of renderers.
        */
        uint32_t addRenderer(Renderer::SharedPtr pRenderer);

        /** An overloadable method that loads and can do custom processing of a newly-loaded scene.  
            \param[in] filename The file of the scene to load, suitable for a direct call to Scene::loadFromFile().
            \return A Scene::SharedPtr representing the loaded scene with all custom preprocessing done.
        */
        virtual Scene::SharedPtr loadScene(const std::string& filename);

    protected:
        GraphicsState::SharedPtr             mpState;                ///< Somme common graphics state
        SceneRenderer::SharedPtr             mpSceneRenderer;        ///< A scene renderer for user-loaded scenes
        std::vector< Renderer::SharedPtr >   mRenderer;              ///< A list of the renderers available for the user to switch between
        uint32_t                             mCurRenderer = 0;       ///< Which renderer are we currently displaying?
        uint32_t                             mSelectedRenderer = 0;  ///< New UI-selection, before we finally switch away from mCurRenderer
        Gui::DropdownList                    mRendererList;          ///< The list of names to display in the UI to switch between renderers

    private:
        /** Overloaded methods from the base Sample class.
        */
        void onLoad() override;
        void onFrameRender() override;
        void onShutdown() override;
        void onResizeSwapChain() override;
        bool onKeyEvent(const KeyboardEvent& keyEvent) override;
        bool onMouseEvent(const MouseEvent& mouseEvent) override;
        void onGuiRender() override;

        /** Called in a couple places to switch between one renderer (fromId) to another (toId)
        */
        void switchRenderers(uint32_t fromId, uint32_t toId);

        /** This is used rather than mpGui->pushWindow() to pass in custom flags to ImGui::Begin()
        */
        void customGuiWindow(const char label[], uint32_t width, uint32_t height, uint32_t x, uint32_t y);

        /** Called when the user clicks on the 'Load Scene' button.
        */
        void internalSceneLoader(const std::string& filename);
    };


} // End namespace Falcor

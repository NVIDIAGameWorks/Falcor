/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "Framework.h"
#include "API/RenderContext.h"

namespace Falcor
{
    /** Magnifies a region of the screen to assist with inspecting details
    */
    class PixelZoom
    {
    public:
        using SharedPtr = std::shared_ptr<PixelZoom>;

        /** Create a new object
            \param[in] pBackBuffer Pointer to the back buffer FBO
        */
        static SharedPtr create(const Fbo* pBackbuffer);

        /** Does zoom operation if mShouldZoom is true (if ctrl+alt pressed this frame)
            \param pCtx Pointer to the render context
            \param backbuffer Pointer to the swap chain FBO
        */
        void render(RenderContext* pCtx, Fbo* backBuffer);

        /** Stores data about mouse needed for zooming
            \param me the mouse event
        */
        bool onMouseEvent(const MouseEvent& me);

        /** Checks if it should zoom
            \param ke Keyboard event
        */
        bool onKeyboardEvent(const KeyboardEvent& ke);

        /** Handle resize events
        */
        void onResizeSwapChain(const Fbo* pBackbuffer);

    private:
        int32_t mSrcZoomSize = 5;
        const uint32_t mDstZoomSize = 200;
        const uint32_t mZoomCoefficient = 4;

        Fbo::SharedPtr mpSrcBlitFbo;
        Fbo::SharedPtr mpDstBlitFbo;
        glm::vec2 mMousePos;
        bool mShouldZoom;
    };
}

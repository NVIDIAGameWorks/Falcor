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
#pragma once
#include "Core/API/FBO.h"

namespace Falcor
{
    class RenderContext;

    /** Class that renders text into the screen.
    */
    class dlldecl TextRenderer
    {
    public:
        enum class Flags
        {
            None     = 0x0,
            Shadowed = 0x1
        };

        /** Initialize the text-renderer
            This class is not thread-safe!
        */
        static void start();

        /** End batching. This will cause the render queue to flush and display the message to the screen.
        */
        static void shutdown();

        /** Render text
            \param[in] pRenderContext A render-context which will be used to dispatch the draw
            \param[in] text The text to draw. It can include newlines, tabs, carriage returns and regular ASCII characters.
            \param[in] pDstFbo The target FBO
            \param[in] pos Text position
        */
        static void render(RenderContext* pRenderContext, const std::string& text, const Fbo::SharedPtr& pDstFbo, float2 pos);

        /** Returns the color of the text being rendered
            \return current color The text color
        */
        static const float3& getColor();

        /** Set the color of the text being rendered
            \param[in] color The text color
        */
        static void setColor(const float3& color);

        /** Get the active flags
        */
        static Flags getFlags();

        /** Set the flags
        */
        static void setFlags(Flags f);
    private:
        TextRenderer() = default;
    };

    enum_class_operators(TextRenderer::Flags);
}

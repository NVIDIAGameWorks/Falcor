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
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include <string>
#include "Font.h"
#include "API/VAO.h"
#include "API/Buffer.h"
#include "Graphics/GraphicsState.h"
#include "Graphics/Program/ProgramVars.h"
#include "API/RenderContext.h"

namespace Falcor
{
    class RenderContext;

    /** Class that renders text into the screen.
        This class batches messages before drawing them for more efficient rendering. In order to do that, you have to enclose TextRenderer#renderLine() calls between TextRenderer#begin() and TextRenderer#end() calls.
    */
    class TextRenderer
    {
    public:
        using UniquePtr = std::unique_ptr<TextRenderer>;
        using UniqueConstPtr = std::unique_ptr<const TextRenderer>;

        ~TextRenderer();

        /** create a new object
        */
        static UniquePtr create();

        /** Start batching messages
            \param[in] pRenderContext The rendering context
            \param[in] startPos The screen-space position, from the top-left, of the first letter to draw
        */
        void begin(const RenderContext::SharedPtr& pRenderContext, const glm::vec2& startPos);

        /** End batching. This will cause the render queue to flush and display the message to the screen.
        */
        void end();

        /** Render a line. After the function is called, an implicit newline is inserted into the message.
            \param[in] line The line to draw. It can include newlines, tabs, carriage returns and regular ASCII characters.
        */
        void renderLine(const std::string& line);
                
        /** Returns the color of the text being rendered
            \return current color The text color
        */
        const glm::vec3& getTextColor() const { return mTextColor; }

        /** Set the color of the text being rendered
            \param[in] color The text color
        */
        void setTextColor(const glm::vec3& color) { mTextColor = color; }

    private:
        TextRenderer();

        struct Vertex
        {
            glm::vec2 screenPos;
            glm::vec2 texCoord;
        };

        RenderContext::SharedPtr mpRenderContext = nullptr;

        glm::vec2 mCurPos = {0, 0};
        glm::vec2 mStartPos = {0, 0};
        glm::vec3 mTextColor = glm::vec3(1, 1, 1);

        Font::UniquePtr mpFont;
        Buffer::SharedPtr mpVertexBuffer;

        GraphicsState::SharedPtr mpPipelineState;
        GraphicsVars::SharedPtr mpProgramVars;

        uint32_t mCurrentVertexID = 0;

        void createVertexBuffer();
        static const auto kMaxBatchSize = 1000;

        void flush();
        Vertex* mpBufferData = nullptr;

        struct  
        {
            size_t vpTransform;
            size_t fontColor;
        } mVarOffsets;
    };
}
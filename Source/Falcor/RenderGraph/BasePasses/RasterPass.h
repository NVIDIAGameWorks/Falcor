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
#include "Core\Program\ShaderVar.h"
#include "BaseGraphicsPass.h"

namespace Falcor
{
    class dlldecl RasterPass : public BaseGraphicsPass, public std::enable_shared_from_this<RasterPass>
    {
    public:
        using SharedPtr = ParameterBlockSharedPtr<RasterPass>;

        /** Create a new object.
            \param[in] filename Program filename.
            \param[in] vsEntry Vertex shader entry point. If this string is empty (""), it will use a default vertex shader which transforms and outputs all default vertex attributes.
            \param[in] psEntry Pixel shader entry point
            \param[in] programDefines Optional list of macro definitions to set into the program. The macro definitions will be set on all shader stages.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(const std::string& filename, const std::string& vsEntry, const std::string& psEntry, const Program::DefineList& defines = Program::DefineList());

        /** Create a new object.
            \param[in] progDesc The program description.
            \param[in] programDefines Optional list of macro definitions to set into the program. The macro definitions will be set on all shader stages.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(const Program::Desc& desc, const Program::DefineList& defines = Program::DefineList());

        /** Ordered draw call.
            \param[in] vertexCount Number of vertices to draw
            \param[in] startVertexLocation The location of the first vertex to read from the vertex buffers (offset in vertices)
        */
        void draw(RenderContext* pContext, uint32_t vertexCount, uint32_t startVertexLocation);

        /** Indexed draw call.
            \param[in] indexCount Number of indices to draw
            \param[in] startIndexLocation The location of the first index to read from the index buffer (offset in indices)
            \param[in] baseVertexLocation A value which is added to each index before reading a vertex from the vertex buffer
        */
        void drawIndexed(RenderContext* pContext, uint32_t indexCount, uint32_t startIndexLocation, int32_t baseVertexLocation);

    protected:
        RasterPass(const Program::Desc& progDesc, const Program::DefineList& programDefines);
    };
}

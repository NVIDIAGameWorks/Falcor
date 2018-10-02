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
#include <map>
#include <string>
#include "Graphics/Program/Program.h"
#include "API/VAO.h"
#include "API/DepthStencilState.h"
#include "API/Buffer.h"
#include "Graphics/Program/ProgramVersion.h"
#include "Graphics/GraphicsState.h"

namespace Falcor
{
    class RenderContext;

    struct FullScreenPassData
    {
        Buffer::SharedPtr pVertexBuffer;
        Vao::SharedPtr    pVao;
        uint64_t objectCount = 0;
    };

    dlldecl FullScreenPassData gFullScreenData;

    /** Helper class to simplify full-screen passes
    */
    class FullScreenPass
    {
    public:
        using UniquePtr = std::unique_ptr<FullScreenPass>;
        using UniqueConstPtr = std::unique_ptr<const FullScreenPass>;

        ~FullScreenPass()
        {
            assert(gFullScreenData.objectCount > 0);

            gFullScreenData.objectCount--;
            if (gFullScreenData.objectCount == 0)
            {
                gFullScreenData.pVao = nullptr;
                gFullScreenData.pVertexBuffer = nullptr;
            }
        }


        /** Create a new object.
            \param[in] psFile Pixel shader filename. Can also be an absolute path or a relative path from a data directory.
            \param[in] shaderDefines Optional. A list of macro definitions to be patched into the shaders.
            \param[in] disableDepth Optional. Disable depth test (and therefore depth writes).  This is the common case; however, e.g. writing depth in fullscreen passes can sometimes be useful.
            \param[in] disableStencil Optional. As DisableDepth for stencil.
            \param[in] viewportMask Optional. Value to initialize viewport mask with. Useful for multi-projection passes
            \param[in] enableSPS Optional. If true, use Single-Pass Stereo when executing this pass.
        */
        static UniquePtr create(const std::string& psFile, const Program::DefineList& programDefines = Program::DefineList(), bool disableDepth = true, bool disableStencil = true, uint32_t viewportMask = 0, bool enableSPS = false, Shader::CompilerFlags compilerFlags = Shader::CompilerFlags::None);
        
        /** Create a new object
            \param[in] vsFile Vertex shader filename. Can also be an absolute path or a relative path from a data directory.
            \param[in] psFile Pixel shader filename. Can also be an absolute path or a relative path from a data directory.
            \param[in] shaderDefines Optional. A list of macro definitions to be patched into the shaders.
            \param[in] disableDepth Optional. Disable depth test (and therefore depth writes).  This is the common case; however, e.g. writing depth in fullscreen passes can sometimes be useful.
            \param[in] disableStencil Optional. As DisableDepth for stencil.
            \param[in] viewportMask Optional. Value to initialize viewport mask with. Useful for multi-projection passes
            \param[in] enableSPS Optional. If true, use Single-Pass Stereo when executing this pass.
        */
        static UniquePtr create(const std::string& vsFile, const std::string& psFile, const Program::DefineList& programDefines = Program::DefineList(), bool disableDepth = true, bool disableStencil = true, uint32_t viewportMask = 0, bool enableSPS = false, Shader::CompilerFlags compilerFlags = Shader::CompilerFlags::None);

        /** Execute the pass.
            \param[in] pRenderContext The render context.
            \param[in] pDsState Optional. Use it to make the pass use a different DS state then the one created during initialization
        */
        void execute(RenderContext* pRenderContext, DepthStencilState::SharedPtr pDsState = nullptr) const;

        /** Get the program.
        */
        const Program::SharedConstPtr getProgram() const { return mpProgram; }
        Program::SharedPtr getProgram() { return mpProgram; }

    protected:
        FullScreenPass() { gFullScreenData.objectCount++; }
        void init(const std::string& vsFile, const std::string & psFile, const Program::DefineList& programDefines, bool disableDepth, bool disableStencil, uint32_t viewportMask, bool enableSPS, Shader::CompilerFlags compilerFlags);

    private:
        GraphicsProgram::SharedPtr mpProgram;
        GraphicsState::SharedPtr mpPipelineState;
        DepthStencilState::SharedPtr mpDepthStencilState;
    };
}
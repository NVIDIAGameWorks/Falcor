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
#include "BaseGraphicsPass.h"
#include "Core/Macros.h"
#include "Core/Program/Program.h"
#include <filesystem>

namespace Falcor
{
    class FALCOR_API FullScreenPass : public BaseGraphicsPass
    {
    public:
        using SharedPtr = ParameterBlockSharedPtr<FullScreenPass>;

        virtual ~FullScreenPass();

        /** Create a new fullscreen pass from file.
            \param[in] path Pixel shader file path. This method expects a pixel shader named "main()" in the file.
            \param[in] defines Optional list of macro definitions to set into the program.
            \param[in] viewportMask Optional value to initialize viewport mask with. Useful for multi-projection passes.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(const std::filesystem::path& path, const Program::DefineList& defines = Program::DefineList(), uint32_t viewportMask = 0);

        /** Create a new fullscreen pass.
            \param[in] desc The program description.
            \param[in] defines Optional list of macro definitions to set into the program.
            \param[in] viewportMask Optional value to initialize viewport mask with. Useful for multi-projection passes.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(const Program::Desc& desc, const Program::DefineList& defines = Program::DefineList(), uint32_t viewportMask = 0);

        /** Execute the pass using an FBO
            \param[in] pRenderContext The render context.
            \param[in] pFbo The target FBO
            \param[in] autoSetVpSc If true, the pass will set the viewports and scissors to match the FBO size. If you want to override the VP or SC, get the state by calling `getState()`, bind the SC and VP yourself and set this arg to false
        */
        virtual void execute(RenderContext* pRenderContext, const Fbo::SharedPtr& pFbo, bool autoSetVpSc = true) const;

    protected:
        FullScreenPass(const Program::Desc& progDesc, const Program::DefineList& programDefines);
    };
}


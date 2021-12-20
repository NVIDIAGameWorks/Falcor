/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "GBuffer.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Materials/TexLODTypes.slang"

using namespace Falcor;

/** Ray traced G-buffer pass.
    This pass renders a fixed set of G-buffer channels using ray tracing.
*/
class GBufferRT : public GBuffer
{
public:
    using SharedPtr = std::shared_ptr<GBufferRT>;

    static const Info kInfo;

    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    Dictionary getScriptingDictionary() override;
    void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;

private:
    void executeRaytrace(RenderContext* pRenderContext, const RenderData& renderData);
    void executeCompute(RenderContext* pRenderContext, const RenderData& renderData);

    Program::DefineList getShaderDefines(const RenderData& renderData) const;
    void setShaderData(const ShaderVar& var, const RenderData& renderData);
    void recreatePrograms();

    GBufferRT(const Dictionary& dict);
    void parseDictionary(const Dictionary& dict) override;

    // Internal state
    bool mComputeDOF = false;           ///< Flag indicating if depth-of-field is computed for the current frame.
    SampleGenerator::SharedPtr mpSampleGenerator;

    // UI variables
    TexLODMode mLODMode = TexLODMode::Mip0;
    bool mUseTraceRayInline = false;
    bool mUseDOF = true;                ///< Option for enabling depth-of-field when camera's aperture radius is nonzero.

    // Ray tracing resources
    struct
    {
        RtProgram::SharedPtr pProgram;
        RtProgramVars::SharedPtr pVars;
    } mRaytrace;

    ComputePass::SharedPtr mpComputePass;
};

/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Falcor.h"
#include "Core/Enum.h"
#include "RenderGraph/RenderPass.h"

using namespace Falcor;

/**
 * Simple composite pass that blends two buffers together.
 *
 * Each input A and B can be independently scaled, and the output C
 * is computed C = A <op> B, where the blend operation is configurable.
 * If the output buffer C is of integer format, floating point values
 * are converted to integers using round-to-nearest-even.
 */
class Composite : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(Composite, "Composite", "Composite pass.");

    /**
     * Composite modes.
     */
    enum class Mode
    {
        Add,
        Multiply,
    };

    FALCOR_ENUM_INFO(
        Mode,
        {
            {Mode::Add, "Add"},
            {Mode::Multiply, "Multiply"},
        }
    );

    static ref<Composite> create(ref<Device> pDevice, const Properties& props) { return make_ref<Composite>(pDevice, props); }

    Composite(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

private:
    DefineList getDefines() const;

    uint2 mFrameDim = {0, 0};
    Mode mMode = Mode::Add;
    float mScaleA = 1.f;
    float mScaleB = 1.f;
    ResourceFormat mOutputFormat = ResourceFormat::RGBA32Float;

    ref<ComputePass> mCompositePass;
};

FALCOR_ENUM_REGISTER(Composite::Mode);

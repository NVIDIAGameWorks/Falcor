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
#include "Composite.h"
#include "CompositeMode.slangh"

namespace
{
const std::string kShaderFile("RenderPasses/Utils/Composite/Composite.cs.slang");

const std::string kInputA = "A";
const std::string kInputB = "B";
const std::string kOutput = "out";

const std::string kMode = "mode";
const std::string kScaleA = "scaleA";
const std::string kScaleB = "scaleB";
const std::string kOutputFormat = "outputFormat";

const Gui::DropdownList kModeList = {
    {(uint32_t)Composite::Mode::Add, "Add"},
    {(uint32_t)Composite::Mode::Multiply, "Multiply"},
};
} // namespace

Composite::Composite(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    // Parse dictionary.
    for (const auto& [key, value] : props)
    {
        if (key == kMode)
            mMode = value;
        else if (key == kScaleA)
            mScaleA = value;
        else if (key == kScaleB)
            mScaleB = value;
        else if (key == kOutputFormat)
            mOutputFormat = value;
        else
            logWarning("Unknown property '{}' in Composite pass properties.", key);
    }

    // Create resources.
    mCompositePass = ComputePass::create(mpDevice, kShaderFile, "main", DefineList(), false);
}

Properties Composite::getProperties() const
{
    Properties props;
    props[kMode] = mMode;
    props[kScaleA] = mScaleA;
    props[kScaleB] = mScaleB;
    if (mOutputFormat != ResourceFormat::Unknown)
        props[kOutputFormat] = mOutputFormat;
    return props;
}

RenderPassReflection Composite::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputA, "Input A").bindFlags(ResourceBindFlags::ShaderResource).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kInputB, "Input B").bindFlags(ResourceBindFlags::ShaderResource).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addOutput(kOutput, "Output").bindFlags(ResourceBindFlags::UnorderedAccess).format(mOutputFormat);
    return reflector;
}

void Composite::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
}

void Composite::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Prepare program.
    const auto& pOutput = renderData.getTexture(kOutput);
    FALCOR_ASSERT(pOutput);
    mOutputFormat = pOutput->getFormat();

    if (mCompositePass->getProgram()->addDefines(getDefines()))
    {
        mCompositePass->setVars(nullptr);
    }

    // Bind resources.
    auto var = mCompositePass->getRootVar();
    var["CB"]["frameDim"] = mFrameDim;
    var["CB"]["scaleA"] = mScaleA;
    var["CB"]["scaleB"] = mScaleB;
    var["A"] = renderData.getTexture(kInputA); // Can be nullptr

    var["B"] = renderData.getTexture(kInputB); // Can be nullptr
    var["output"] = pOutput;
    mCompositePass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}

void Composite::renderUI(Gui::Widgets& widget)
{
    widget.text("This pass scales and composites inputs A and B together");
    widget.dropdown("Mode", mMode);
    widget.var("Scale A", mScaleA);
    widget.var("Scale B", mScaleB);
}

DefineList Composite::getDefines() const
{
    uint32_t compositeMode = 0;
    switch (mMode)
    {
    case Mode::Add:
        compositeMode = COMPOSITE_MODE_ADD;
        break;
    case Mode::Multiply:
        compositeMode = COMPOSITE_MODE_MULTIPLY;
        break;
    default:
        FALCOR_UNREACHABLE();
        break;
    }

    FALCOR_ASSERT(mOutputFormat != ResourceFormat::Unknown);
    uint32_t outputFormat = 0;
    switch (getFormatType(mOutputFormat))
    {
    case FormatType::Uint:
        outputFormat = OUTPUT_FORMAT_UINT;
        break;
    case FormatType::Sint:
        outputFormat = OUTPUT_FORMAT_SINT;
        break;
    default:
        outputFormat = OUTPUT_FORMAT_FLOAT;
        break;
    }

    DefineList defines;
    defines.add("COMPOSITE_MODE", std::to_string(compositeMode));
    defines.add("OUTPUT_FORMAT", std::to_string(outputFormat));

    return defines;
}

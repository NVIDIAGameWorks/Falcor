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
#include "ColorMapPass.h"

namespace
{
const std::string kShaderFile = "RenderPasses/DebugPasses/ColorMapPass/ColorMapPass.ps.slang";

const std::string kInput = "input";
const std::string kOutput = "output";

const std::string kColorMap = "colorMap";
const std::string kChannel = "channel";
const std::string kAutoRange = "autoRange";
const std::string kMinValue = "minValue";
const std::string kMaxValue = "maxValue";
} // namespace

ColorMapPass::ColorMapPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kColorMap)
            mColorMap = value;
        else if (key == kChannel)
            mChannel = value;
        else if (key == kAutoRange)
            mAutoRange = value;
        else if (key == kMinValue)
            mMinValue = value;
        else if (key == kMaxValue)
            mMaxValue = value;
        else
            logWarning("Unknown property '{}' in a ColorMapPass properties.", key);
    }

    mpFbo = Fbo::create(mpDevice);
}

Properties ColorMapPass::getProperties() const
{
    Properties props;
    props[kColorMap] = mColorMap;
    props[kChannel] = mChannel;
    props[kAutoRange] = mAutoRange;
    props[kMinValue] = mMinValue;
    props[kMaxValue] = mMaxValue;
    return props;
}

RenderPassReflection ColorMapPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addInput(kInput, "Input image").bindFlags(Falcor::ResourceBindFlags::ShaderResource).texture2D(0, 0);
    r.addOutput(kOutput, "Output image").bindFlags(Falcor::ResourceBindFlags::RenderTarget).texture2D(0, 0);
    return r;
}

void ColorMapPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& inputTexture = renderData.getTexture(kInput);
    const auto& outputTexture = renderData.getTexture(kOutput);

    FormatType inputType = inputTexture ? getFormatType(inputTexture->getFormat()) : FormatType::Float;

    if (mAutoRange && inputTexture)
    {
        if (!mpAutoRanging)
            mpAutoRanging = std::make_unique<AutoRanging>(mpDevice);

        if (auto minMax = mpAutoRanging->getMinMax(pRenderContext, inputTexture, mChannel))
        {
            auto [minValue, maxValue] = *minMax;

            // Immediately grow the auto range to include the range of the input.
            mAutoMinValue = std::min(mAutoMinValue, minValue);
            mAutoMaxValue = std::max(mAutoMaxValue, maxValue);

            // Smoothly shrink the auto range when the the range of the input shrinks.
            double alpha = 0.01;
            mAutoMinValue = math::lerp(mAutoMinValue, minValue, alpha);
            mAutoMaxValue = math::lerp(mAutoMaxValue, maxValue, alpha);

            mMinValue = (float)mAutoMinValue;
            mMaxValue = (float)mAutoMaxValue;
        }
        else
        {
            mAutoMinValue = mMinValue;
            mAutoMaxValue = mMaxValue;
        }
    }
    else
    {
        mpAutoRanging.reset();
    }

    DefineList defines;
    defines.add("_COLOR_MAP", std::to_string((uint32_t)mColorMap));
    defines.add("_CHANNEL", std::to_string(mChannel));

    switch (inputType)
    {
    case FormatType::Uint:
        defines.add("_FORMAT", "FORMAT_UINT");
        break;
    case FormatType::Sint:
        defines.add("_FORMAT", "FORMAT_SINT");
        break;
    default:
        defines.add("_FORMAT", "FORMAT_FLOAT");
        break;
    }

    if (!mpColorMapPass || mRecompile)
    {
        mpColorMapPass = FullScreenPass::create(mpDevice, kShaderFile, defines);
        mRecompile = false;
    }

    ColorMapParams params;
    params.minValue = mMinValue;
    params.maxValue = mMaxValue;

    auto var = mpColorMapPass->getRootVar();
    var["gTexture"] = inputTexture;
    var["StaticCB"]["gParams"].setBlob(params);
    mpFbo->attachColorTarget(outputTexture, 0);
    mpColorMapPass->getState()->setFbo(mpFbo);
    mpColorMapPass->execute(pRenderContext, mpFbo);
}

void ColorMapPass::renderUI(Gui::Widgets& widget)
{
    mRecompile |= widget.dropdown("Color Map", mColorMap);
    mRecompile |= widget.var("Channel", mChannel, 0u, 3u);
    widget.checkbox("Auto Range", mAutoRange);
    widget.var("Min Value", mMinValue);
    widget.var("Max Value", mMaxValue);
}

ColorMapPass::AutoRanging::AutoRanging(ref<Device> pDevice)
{
    mpParallelReduction = std::make_unique<ParallelReduction>(pDevice);
    mpReductionResult = pDevice->createBuffer(32, ResourceBindFlags::None, MemoryType::ReadBack);
    mpFence = pDevice->createFence();
}

std::optional<std::pair<double, double>> ColorMapPass::AutoRanging::getMinMax(
    RenderContext* pRenderContext,
    const ref<Texture>& texture,
    uint32_t channel
)
{
    FALCOR_ASSERT(pRenderContext);
    FALCOR_ASSERT(texture);
    FALCOR_ASSERT(channel < 4);

    std::optional<std::pair<double, double>> result;

    FormatType formatType = getFormatType(texture->getFormat());

    if (mReductionAvailable)
    {
        mpFence->wait();

        const void* values = mpReductionResult->map(Buffer::MapType::Read);

        switch (formatType)
        {
        case FormatType::Uint:
            result = {reinterpret_cast<const uint4*>(values)[0][channel], reinterpret_cast<const uint4*>(values)[1][channel]};
            break;
        case FormatType::Sint:
            result = {reinterpret_cast<const int4*>(values)[0][channel], reinterpret_cast<const int4*>(values)[1][channel]};
            break;
        default:
            result = {reinterpret_cast<const float4*>(values)[0][channel], reinterpret_cast<const float4*>(values)[1][channel]};
            break;
        }

        mpReductionResult->unmap();

        mReductionAvailable = false;
    }

    switch (formatType)
    {
    case FormatType::Uint:
        mpParallelReduction->execute<uint4>(pRenderContext, texture, ParallelReduction::Type::MinMax, nullptr, mpReductionResult);
        break;
    case FormatType::Sint:
        mpParallelReduction->execute<int4>(pRenderContext, texture, ParallelReduction::Type::MinMax, nullptr, mpReductionResult);
        break;
    default:
        mpParallelReduction->execute<float4>(pRenderContext, texture, ParallelReduction::Type::MinMax, nullptr, mpReductionResult);
        break;
    }

    pRenderContext->signal(mpFence.get());
    mReductionAvailable = true;

    return result;
}

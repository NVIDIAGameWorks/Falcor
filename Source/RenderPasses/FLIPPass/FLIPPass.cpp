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
#include "FLIPPass.h"
#include "Utils/Algorithm/ComputeParallelReduction.h"

namespace
{
    const char kDesc[] = "FLIP metric pass";

    const char kInputA[] = "inputA";
    const char kInputB[] = "inputB";
    const char kOutput[] = "output";

    const char kEnabled[] = "enabled";
    const char kUseMagma[] = "useMagma";
    const char kMonitorWidthPixels[] = "monitorWidthPixels";
    const char kMonitorWidthMeters[] = "monitorWidthMeters";
    const char kMonitorDistance[] = "monitorDistanceMeters";
    const char kCalculatePerFrameFLIP[] = "calculatePerFrameFLIP";
    const char kUseRealMonitorInfo[] = "useRealMonitorInfo";
}

// Don't remove this. it's required for hot-reload to function properly.
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("FLIPPass", kDesc, FLIPPass::create);
}

FLIPPass::SharedPtr FLIPPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new FLIPPass(dict));
}

FLIPPass::FLIPPass(const Dictionary& dict)
{
    mpFLIPPass = ComputePass::create("RenderPasses/FLIPPass/FLIPPass.cs.slang", "main", Program::DefineList());

    // Create parallel reduction helper.
    mpParallelReduction = ComputeParallelReduction::create();

    // Fill some reasonable defaults for monitor information.
    mMonitorWidthPixels = 3840;
    mMonitorWidthMeters = 0.5f;
    mMonitorDistanceMeters = 0.7f;

    // Read Settings
    for (const auto& [key, value] : dict)
    {
        if (key == kEnabled) mEnabled = value;
        else if (key == kUseMagma) mUseMagma = value;
        else if (key == kMonitorWidthPixels) mMonitorWidthPixels = value;
        else if (key == kMonitorWidthMeters) mMonitorWidthMeters = value;
        else if (key == kMonitorDistance) mMonitorDistanceMeters = value;
        else if (key == kCalculatePerFrameFLIP) mCalculatePerFrameFLIP = value;
        else if (key == kUseRealMonitorInfo) mUseRealMonitorInfo = value;
        else logWarning("Unknown field '" + key + "' in a FLIPPass dictionary");
    }

    // Evaluate monitor information.
    std::vector<MonitorInfo::MonitorDesc> monitorDescs = MonitorInfo::getMonitorDescs();

    // Override defaults by real monitor info, if available.
    if (mUseRealMonitorInfo && (monitorDescs.size() > 0))
    {
        // Assume first monitor is used.
        size_t monitorIndex = 0;
        if (monitorDescs[monitorIndex].mResolution.x > 0) mMonitorWidthPixels = uint(monitorDescs[0].mResolution.x);
        if (monitorDescs[monitorIndex].mPhysicalSize.x > 0) mMonitorWidthMeters = monitorDescs[0].mPhysicalSize.x * 0.0254f; //< Convert from inches to meters
    }

}

std::string FLIPPass::getDesc() { return kDesc; }

Dictionary FLIPPass::getScriptingDictionary()
{
    Dictionary d;
    d[kEnabled] = mEnabled;
    d[kUseMagma] = mUseMagma;
    d[kMonitorWidthPixels] = mMonitorWidthPixels;
    d[kMonitorWidthMeters] = mMonitorWidthMeters;
    d[kMonitorDistance] = mMonitorDistanceMeters;
    d[kCalculatePerFrameFLIP] = mCalculatePerFrameFLIP;
    d[kUseRealMonitorInfo] = mUseRealMonitorInfo;
    return d;
}

RenderPassReflection FLIPPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputA, "First Input").bindFlags(Falcor::Resource::BindFlags::ShaderResource).texture2D(0, 0);
    reflector.addInput(kInputB, "Second Input").bindFlags(Falcor::Resource::BindFlags::ShaderResource).texture2D(0, 0);
    reflector.addOutput(kOutput, "FLIP output").format(ResourceFormat::RGBA8Unorm).bindFlags(Falcor::Resource::BindFlags::RenderTarget).texture2D(0, 0);
    return reflector;
}

void FLIPPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mEnabled) return;

    // Pick up resources from render graph.
    const auto& pInputA = renderData[kInputA];
    const auto& pInputB = renderData[kInputB];
    const auto& pOutput = renderData[kOutput];

    // Check for mandatory resources.
    if (!pInputA || !pInputB || !pOutput)
    {
        logWarning("FLIPPass::execute() - missing mandatory resources");
        return;
    }

    // Refresh internal high precision buffer for FLIP result.
    uint2 outputDimensions = uint2(pOutput->asTexture()->getWidth(), pOutput->asTexture()->getHeight());
    if (!mpFLIPOutput || mpFLIPOutput->getWidth() != outputDimensions.x || mpFLIPOutput->getHeight() != outputDimensions.y)
    {
        mpFLIPOutput = Texture::create2D(outputDimensions.x, outputDimensions.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    }

    // Bind resources
    mpFLIPPass["gInputA"] = pInputA->asTexture();
    mpFLIPPass["gInputB"] = pInputB->asTexture();
    mpFLIPPass["gOutput"] = mpFLIPOutput->asTexture();

    // Bind CB settings
    mpFLIPPass["PerFrameCB"]["gUseMagma"] = mUseMagma;
    mpFLIPPass["PerFrameCB"]["gDimensions"] = outputDimensions;

    mpFLIPPass["PerFrameCB"]["gMonitorWidthPixels"] = mMonitorWidthPixels;
    mpFLIPPass["PerFrameCB"]["gMonitorWidthMeters"] = mMonitorWidthMeters;
    mpFLIPPass["PerFrameCB"]["gMonitorDistance"] = mMonitorDistanceMeters;

    // Do the job.
    mpFLIPPass->execute(pRenderContext, outputDimensions.x, outputDimensions.y);

    // Copy result to output buffer (possibly lower precision than our internal buffer).
    pRenderContext->blit(mpFLIPOutput->getSRV(), pOutput->asTexture()->getRTV());

    // Calculate per-frame metrics using parallel reduction.
    if (mCalculatePerFrameFLIP)
    {
        float4 flipSum, flipMinMax[2];
        mpParallelReduction->execute<float4>(pRenderContext, mpFLIPOutput, ComputeParallelReduction::Type::Sum, &flipSum);
        mpParallelReduction->execute<float4>(pRenderContext, mpFLIPOutput, ComputeParallelReduction::Type::MinMax, &flipMinMax[0]);
        pRenderContext->flush(true);

        // Extract metrics from readback values. RGB channels contain magma mapping, A channel contains FLIP value.
        mAverageFLIP = flipSum.a / (outputDimensions.x * outputDimensions.y);
        mMinFLIP = flipMinMax[0].a;
        mMaxFLIP = flipMinMax[1].a;
    }
}

void FLIPPass::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enabled", mEnabled);

    widget.text("FLIP Settings:");
    widget.checkbox("Use Magma", mUseMagma);
    widget.checkbox("Per-frame metrics", mCalculatePerFrameFLIP);

    if (mCalculatePerFrameFLIP)
    {
        widget.indent(10.0f);
        widget.text("Average: " + std::to_string(mAverageFLIP));
        widget.text("Min: " + std::to_string(mMinFLIP));
        widget.text("Max: " + std::to_string(mMaxFLIP));
        widget.indent(-10.0f);
    }

    widget.separator();
    widget.text("Monitor Information:");
    widget.slider("Monitor Distance (meters)", mMonitorDistanceMeters, 0.3f, 1.5f);
    widget.slider("Monitor Width (meters)", mMonitorWidthMeters, 0.3f, 1.5f);
    widget.slider("Monitor Resolution (horizontal)", mMonitorWidthPixels, 720u, 7680u);
}

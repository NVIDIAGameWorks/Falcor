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
#include "ModulateIllumination.h"
#include "RenderGraph/RenderPassLibrary.h"
#include "RenderGraph/RenderPassHelpers.h"

const RenderPass::Info ModulateIllumination::kInfo { "ModulateIllumination", "Modulate illumination pass." };

namespace
{
    const std::string kShaderFile("RenderPasses/ModulateIllumination/ModulateIllumination.cs.slang");

    const Falcor::ChannelList kInputChannels =
    {
        { "emission",                       "gEmission",                        "Emission",                         true /* optional */ },
        { "diffuseReflectance",             "gDiffuseReflectance",              "Diffuse Reflectance",              true /* optional */ },
        { "diffuseRadiance",                "gDiffuseRadiance",                 "Diffuse Radiance",                 true /* optional */ },
        { "specularReflectance",            "gSpecularReflectance",             "Specular Reflectance",             true /* optional */ },
        { "specularRadiance",               "gSpecularRadiance",                "Specular Radiance",                true /* optional */ },
        { "deltaReflectionEmission",        "gDeltaReflectionEmission",         "Delta Reflection Emission",        true /* optional */ },
        { "deltaReflectionReflectance",     "gDeltaReflectionReflectance",      "Delta Reflection Reflectance",     true /* optional */ },
        { "deltaReflectionRadiance",        "gDeltaReflectionRadiance",         "Delta Reflection Radiance",        true /* optional */ },
        { "deltaTransmissionEmission",      "gDeltaTransmissionEmission",       "Delta Transmission Emission",      true /* optional */ },
        { "deltaTransmissionReflectance",   "gDeltaTransmissionReflectance",    "Delta Transmission Reflectance",   true /* optional */ },
        { "deltaTransmissionRadiance",      "gDeltaTransmissionRadiance",       "Delta Transmission Radiance",      true /* optional */ },
        { "residualRadiance",               "gResidualRadiance",                "Residual Radiance",                true /* optional */ },
    };

    const std::string kOutput = "output";

    // Serialized parameters.
    const char kUseEmission[] = "useEmission";
    const char kUseDiffuseReflectance[] = "useDiffuseReflectance";
    const char kUseDiffuseRadiance[] = "useDiffuseRadiance";
    const char kUseSpecularReflectance[] = "useSpecularReflectance";
    const char kUseSpecularRadiance[] = "useSpecularRadiance";
    const char kUseDeltaReflectionEmission[] = "useDeltaReflectionEmission";
    const char kUseDeltaReflectionReflectance[] = "useDeltaReflectionReflectance";
    const char kUseDeltaReflectionRadiance[] = "useDeltaReflectionRadiance";
    const char kUseDeltaTransmissionEmission[] = "useDeltaTransmissionEmission";
    const char kUseDeltaTransmissionReflectance[] = "useDeltaTransmissionReflectance";
    const char kUseDeltaTransmissionRadiance[] = "useDeltaTransmissionRadiance";
    const char kUseResidualRadiance[] = "useResidualRadiance";
    const char kOutputSize[] = "outputSize";
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(ModulateIllumination::kInfo, ModulateIllumination::create);
}

ModulateIllumination::SharedPtr ModulateIllumination::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new ModulateIllumination(dict));
}

ModulateIllumination::ModulateIllumination(const Dictionary& dict)
    : RenderPass(kInfo)
{
    mpModulateIlluminationPass = ComputePass::create(kShaderFile, "main", Program::DefineList(), false);

    // Deserialize pass from dictionary.
    for (const auto& [key, value] : dict)
    {
        if (key == kUseEmission) mUseEmission = value;
        else if (key == kUseDiffuseReflectance) mUseDiffuseReflectance = value;
        else if (key == kUseDiffuseRadiance) mUseDiffuseRadiance = value;
        else if (key == kUseSpecularReflectance) mUseSpecularReflectance = value;
        else if (key == kUseSpecularRadiance) mUseSpecularRadiance = value;
        else if (key == kUseDeltaReflectionEmission) mUseDeltaReflectionEmission = value;
        else if (key == kUseDeltaReflectionReflectance) mUseDeltaReflectionReflectance = value;
        else if (key == kUseDeltaReflectionRadiance) mUseDeltaReflectionRadiance = value;
        else if (key == kUseDeltaTransmissionEmission) mUseDeltaTransmissionEmission = value;
        else if (key == kUseDeltaTransmissionReflectance) mUseDeltaTransmissionReflectance = value;
        else if (key == kUseDeltaTransmissionRadiance) mUseDeltaTransmissionRadiance = value;
        else if (key == kUseResidualRadiance) mUseResidualRadiance = value;
        else if (key == kOutputSize) mOutputSizeSelection = value;
        else
        {
            logWarning("Unknown field '{}' in ModulateIllumination dictionary.", key);
        }
    }
}

Falcor::Dictionary ModulateIllumination::getScriptingDictionary()
{
    Dictionary dict;
    dict[kUseEmission] = mUseEmission;
    dict[kUseDiffuseReflectance] = mUseDiffuseReflectance;
    dict[kUseDiffuseRadiance] = mUseDiffuseRadiance;
    dict[kUseSpecularReflectance] = mUseSpecularReflectance;
    dict[kUseSpecularRadiance] = mUseSpecularRadiance;
    dict[kUseDeltaReflectionEmission] = mUseDeltaReflectionEmission;
    dict[kUseDeltaReflectionReflectance] = mUseDeltaReflectionReflectance;
    dict[kUseDeltaReflectionRadiance] = mUseDeltaReflectionRadiance;
    dict[kUseDeltaTransmissionEmission] = mUseDeltaTransmissionEmission;
    dict[kUseDeltaTransmissionReflectance] = mUseDeltaTransmissionReflectance;
    dict[kUseDeltaTransmissionRadiance] = mUseDeltaTransmissionRadiance;
    dict[kUseResidualRadiance] = mUseResidualRadiance;
    dict[kOutputSize] = mOutputSizeSelection;
    return dict;
}

RenderPassReflection ModulateIllumination::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    addRenderPassInputs(reflector, kInputChannels);

    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFrameDim, compileData.defaultTexDims);
    // TODO: Allow user to specify output format
    reflector.addOutput(kOutput, "output").bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::RGBA32Float).texture2D(sz.x, sz.y);
    return reflector;
}

void ModulateIllumination::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
}

void ModulateIllumination::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutput = renderData.getTexture(kOutput);
    mFrameDim = { pOutput->getWidth(), pOutput->getHeight() };

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    Program::DefineList defineList = getValidResourceDefines(kInputChannels, renderData);

    // Override defines.
    if (!mUseEmission) defineList["is_valid_gEmission"] = "0";
    if (!mUseDiffuseReflectance) defineList["is_valid_gDiffuseReflectance"] = "0";
    if (!mUseDiffuseRadiance) defineList["is_valid_gDiffuseRadiance"] = "0";
    if (!mUseSpecularReflectance) defineList["is_valid_gSpecularReflectance"] = "0";
    if (!mUseSpecularRadiance) defineList["is_valid_gSpecularRadiance"] = "0";
    if (!mUseDeltaReflectionEmission) defineList["is_valid_gDeltaReflectionEmission"] = "0";
    if (!mUseDeltaReflectionReflectance) defineList["is_valid_gDeltaReflectionReflectance"] = "0";
    if (!mUseDeltaReflectionRadiance) defineList["is_valid_gDeltaReflectionRadiance"] = "0";
    if (!mUseDeltaTransmissionEmission) defineList["is_valid_gDeltaTransmissionEmission"] = "0";
    if (!mUseDeltaTransmissionReflectance) defineList["is_valid_gDeltaTransmissionReflectance"] = "0";
    if (!mUseDeltaTransmissionRadiance) defineList["is_valid_gDeltaTransmissionRadiance"] = "0";
    if (!mUseResidualRadiance) defineList["is_valid_gResidualRadiance"] = "0";

    if (mpModulateIlluminationPass->getProgram()->addDefines(defineList))
    {
        mpModulateIlluminationPass->setVars(nullptr);
    }

    mpModulateIlluminationPass["CB"]["frameDim"] = mFrameDim;

    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            Texture::SharedPtr pTexture = renderData.getTexture(desc.name);
            if (pTexture && (mFrameDim.x != pTexture->getWidth() || mFrameDim.y != pTexture->getHeight()))
            {
                logError("Texture {} has dim {]x{}, not compatible with the FrameDim {}x{}.",
                    pTexture->getName(), pTexture->getWidth(), pTexture->getHeight(), mFrameDim.x, mFrameDim.y);
            }
            mpModulateIlluminationPass[desc.texname] = pTexture;
        }
    };
    for (const auto& channel : kInputChannels) bind(channel);

    mpModulateIlluminationPass["gOutput"] = renderData.getTexture(kOutput);

    mpModulateIlluminationPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}

void ModulateIllumination::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Emission", mUseEmission);
    widget.checkbox("Diffuse Reflectance", mUseDiffuseReflectance);
    widget.checkbox("Diffuse Radiance", mUseDiffuseRadiance);
    widget.checkbox("Specular Reflectance", mUseSpecularReflectance);
    widget.checkbox("Specular Radiance", mUseSpecularRadiance);
    widget.checkbox("Delta Reflection Emission", mUseDeltaReflectionEmission);
    widget.checkbox("Delta Reflection Reflectance", mUseDeltaReflectionReflectance);
    widget.checkbox("Delta Reflection Radiance", mUseDeltaReflectionRadiance);
    widget.checkbox("Delta Transmission Emission", mUseDeltaTransmissionEmission);
    widget.checkbox("Delta Transmission Reflectance", mUseDeltaTransmissionReflectance);
    widget.checkbox("Delta Transmission Radiance", mUseDeltaTransmissionRadiance);
    widget.checkbox("Residual Radiance", mUseResidualRadiance);
}

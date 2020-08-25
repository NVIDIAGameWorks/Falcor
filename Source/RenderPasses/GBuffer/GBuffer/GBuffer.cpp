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
#include "GBuffer.h"

// List of primary GBuffer channels. These correspond to the render targets
// used in the GBufferRaster pixel shader. Note that channel order should
// correspond to SV_TARGET index order.
const ChannelList GBuffer::kGBufferChannels =
{
    { "posW",           "gPosW",            "world space position",         true /* optional */, ResourceFormat::RGBA32Float },
    { "normW",          "gNormW",           "world space normal",           true /* optional */, ResourceFormat::RGBA32Float },
    { "tangentW",       "gTangentW",        "world space tangent",          true /* optional */, ResourceFormat::RGBA32Float },
    { "texC",           "gTexC",            "texture coordinates",          true /* optional */, ResourceFormat::RGBA32Float },
    { "diffuseOpacity", "gDiffuseOpacity",  "diffuse color and opacity",    true /* optional */, ResourceFormat::RGBA32Float },
    { "specRough",      "gSpecRough",       "specular color and roughness", true /* optional */, ResourceFormat::RGBA32Float },
    { "emissive",       "gEmissive",        "emissive color",               true /* optional */, ResourceFormat::RGBA32Float },
    { "matlExtra",      "gMatlExtra",       "additional material data",     true /* optional */, ResourceFormat::RGBA32Float },
};

namespace
{
    // Scripting options.
    const char kForceCullMode[] = "forceCullMode";
    const char kCullMode[] = "cull";

    // UI variables.
    const Gui::DropdownList kCullModeList =
    {
        { (uint32_t)RasterizerState::CullMode::None, "None" },
        { (uint32_t)RasterizerState::CullMode::Back, "Back" },
        { (uint32_t)RasterizerState::CullMode::Front, "Front" },
    };
}

GBuffer::GBuffer() : mGBufferParams{}
{
    assert(kGBufferChannels.size() == 8); // The list of primary GBuffer channels should contain 8 entries, corresponding to the 8 render targets.
}

void GBuffer::parseDictionary(const Dictionary& dict)
{
    GBufferBase::parseDictionary(dict);

    for (const auto& [key, value] : dict)
    {
        if (key == kForceCullMode) mForceCullMode = value;
        else if (key == kCullMode) mCullMode = value;
        // TODO: Check for unparsed fields, including those parsed in base classes.
    }
}

Dictionary GBuffer::getScriptingDictionary()
{
    Dictionary dict = GBufferBase::getScriptingDictionary();
    dict[kForceCullMode] = mForceCullMode;
    dict[kCullMode] = mCullMode;
    return dict;
}

void GBuffer::renderUI(Gui::Widgets& widget)
{
    // Render the base class UI first.
    GBufferBase::renderUI(widget);

    // Cull mode controls.
    mOptionsChanged |= widget.checkbox("Force cull mode", mForceCullMode);
    widget.tooltip("Enable this option to force the same cull mode for all geometry.\n\n"
        "Otherwise the default for rasterization is to set the cull mode automatically based on triangle winding, and for ray tracing to disable culling.", true);

    if (mForceCullMode)
    {
        uint32_t cullMode = (uint32_t)mCullMode;
        if (widget.dropdown("Cull mode", kCullModeList, cullMode))
        {
            setCullMode((RasterizerState::CullMode)cullMode);
            mOptionsChanged = true;
        }
    }
}

void GBuffer::compile(RenderContext* pContext, const CompileData& compileData)
{
    GBufferBase::compile(pContext, compileData);

    mGBufferParams.frameSize = mFrameDim;
    mGBufferParams.invFrameSize = mInvFrameDim;
}

void GBuffer::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    GBufferBase::setScene(pRenderContext, pScene);

    mGBufferParams.frameCount = 0;
}

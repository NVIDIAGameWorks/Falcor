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
 **************************************************************************/
#include "Composite.h"

const char* Composite::kDesc = "Composite pass";

namespace
{
    const std::string kShaderFile("RenderPasses/Utils/Composite/Composite.cs.slang");

    const std::string kInputA = "A";
    const std::string kInputB = "B";
    const std::string kOutput = "out";

    const std::string kScaleA = "scaleA";
    const std::string kScaleB = "scaleB";
}

Composite::SharedPtr Composite::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new Composite(dict));
}

Composite::Composite(const Dictionary& dict)
{
    mCompositePass = ComputePass::create(kShaderFile);

    if (!parseDictionary(dict)) throw std::exception("Invalid dictionary");
}

bool Composite::parseDictionary(const Dictionary& dict)
{
    for (const auto& v : dict)
    {
        if (v.key() == kScaleA) mScaleA = v.val();
        else if (v.key() == kScaleB) mScaleB = v.val();
        else
        {
            logError("Unknown field `" + v.key() + "` in Composite pass dictionary");
            return false;
        }
    }
    return true;
}

Dictionary Composite::getScriptingDictionary()
{
    Dictionary dict;
    dict[kScaleA] = mScaleA;
    dict[kScaleB] = mScaleB;
    return dict;
}

RenderPassReflection Composite::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputA, "Input A").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputB, "Input B").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput(kOutput, "Output").bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::RGBA32Float); // TODO: Allow user to specify output format
    return reflector;
}

void Composite::compile(RenderContext* pContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
    mCompositePass["CB"]["frameDim"] = mFrameDim;
}

void Composite::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto cb = mCompositePass["CB"];
    cb["scaleA"] = mScaleA;
    cb["scaleB"] = mScaleB;
    mCompositePass["A"] = renderData[kInputA]->asTexture();
    mCompositePass["B"] = renderData[kInputB]->asTexture();
    mCompositePass["output"] = renderData[kOutput]->asTexture();
    mCompositePass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}

void Composite::renderUI(Gui::Widgets& widget)
{
    widget.text("This pass scales and adds inputs A and B together");
    widget.var("Scale A", mScaleA);
    widget.var("Scale B", mScaleB);
}

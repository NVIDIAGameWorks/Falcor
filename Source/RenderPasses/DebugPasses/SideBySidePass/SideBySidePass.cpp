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
#include "SideBySidePass.h"

const char* SideBySidePass::kDesc = "Allows the user to compare two inputs side-by-side.";

namespace
{
    const std::string kImageLeftBound = "imageLeftBound";

    // Where is our shader located?
    const std::string kSplitShader = "RenderPasses/DebugPasses/SideBySidePass/SideBySide.ps.slang";
}

SideBySidePass::SideBySidePass()
{
    createProgram();
}

SideBySidePass::SharedPtr SideBySidePass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new SideBySidePass());
    for (const auto& [key, value] : dict)
    {
        if (key == kImageLeftBound) pPass->mImageLeftBound = value;
        else if (!pPass->parseKeyValuePair(key, value))
        {
            logWarning("Unknown field '" + key + "' in a SideBySidePass dictionary");
        }
    }
    return pPass;
}

void SideBySidePass::createProgram()
{
    // Create our shader that splits the screen.
    mpSplitShader = FullScreenPass::create(kSplitShader);
}

void SideBySidePass::execute(RenderContext* pContext, const RenderData& renderData)
{
    mpSplitShader["GlobalCB"]["gLeftBound"] = mImageLeftBound;
    ComparisonPass::execute(pContext, renderData);
}

void SideBySidePass::renderUI(Gui::Widgets& widget)
{
    uint32_t width = pDstFbo ? pDstFbo->getWidth() : 0;
    widget.slider("View Slider", mImageLeftBound, 0u, width / 2);
    ComparisonPass::renderUI(widget);
}

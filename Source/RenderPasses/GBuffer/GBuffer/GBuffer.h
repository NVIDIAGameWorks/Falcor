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
#pragma once
#include "../GBufferBase.h"
#include "GBufferParams.slang"
#include "RenderGraph/RenderPassHelpers.h"

using namespace Falcor;

/** Base class for the different G-buffer passes.
*/
class GBuffer : public GBufferBase
{
public:
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void compile(RenderContext* pContext, const CompileData& compileData) override;
    virtual Dictionary getScriptingDictionary() override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;

protected:
    GBuffer();
    virtual void parseDictionary(const Dictionary& dict) override;
    virtual void setCullMode(RasterizerState::CullMode mode) { mCullMode = mode; }

    // Constants used in derived classes
    static const ChannelList kGBufferChannels;

    // Internal state
    GBufferParams                   mGBufferParams;

    // UI variables
    bool                            mForceCullMode = false;                         ///< Force cull mode for all geometry, otherwise set it based on the scene.
    RasterizerState::CullMode       mCullMode = RasterizerState::CullMode::Back;    ///< Cull mode to use for when mForceCullMode is true.
};

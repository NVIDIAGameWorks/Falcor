/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/
#include "Framework.h"
#include "RenderPassesLibrary.h"
#include "RenderPasses/BlitPass.h"
#include "RenderPasses/DepthPass.h"
#include "RenderPasses/SceneLightingPass.h"
#include "Effects/SkyBox/SkyBox.h"
#include "Effects/Shadows/CSM.h"
#include "Effects/ToneMapping/ToneMapping.h"
#include "Effects/FXAA/FXAA.h"
#include "Effects/AmbientOcclusion/SSAO.h"
#include "Effects/TAA/TAA.h"

namespace Falcor
{
    struct RenderPassDesc 
    {
        std::string passDesc;
        RenderPassLibrary::CreateFunc create;
    };

    static std::unordered_map<std::string, RenderPassDesc> gRenderPassList;

    static bool addBuiltinPasses()
    {
        RenderPassLibrary::addRenderPassClass("BlitPass", "Blit one texture into another", BlitPass::deserialize);
        RenderPassLibrary::addRenderPassClass("SceneLightingPass", "Forward-rendering lighting pass", SceneLightingPass::deserialize);
        RenderPassLibrary::addRenderPassClass("DepthPass", "Depth pass", DepthPass::deserialize);
        RenderPassLibrary::addRenderPassClass("CascadedShadowMaps", "Cascaded shadow maps", CascadedShadowMaps::deserialize);
        RenderPassLibrary::addRenderPassClass("ToneMappingPass", "Tone-Mapping", ToneMapping::deserialize);
        RenderPassLibrary::addRenderPassClass("FXAA", "FXAA", FXAA::deserialize);
        RenderPassLibrary::addRenderPassClass("SSAO", "Fast Approximate Anti-Aliasing", SSAO::deserialize);
        RenderPassLibrary::addRenderPassClass("TemporalAA", "Temporal Anti-Aliasing", TemporalAA::deserialize);
        RenderPassLibrary::addRenderPassClass("SkyBox", "Sky Box pass", SkyBox::deserialize);

        return true;
    };

    static const bool b = addBuiltinPasses();

    void RenderPassLibrary::addRenderPassClass(const char* className, const char* desc, CreateFunc func)
    {
        if (gRenderPassList.find(className) != gRenderPassList.end())
        {
            logWarning(std::string("Trying to add a render-pass `") + className + "` to the render-passes library,  but a render-pass with the same name already exists. Ignoring the new definition");
        }
        else
        {
            gRenderPassList[className] = { desc, func };
        }
    }

    std::shared_ptr<RenderPass> RenderPassLibrary::createRenderPass(const char* className, const RenderPassSerializer& serializer)
    {
        if (gRenderPassList.find(className) == gRenderPassList.end())
        {
            logWarning(std::string("Trying to create a render-pass named `") + className + "`, but no such class exists in the library");
            return nullptr;
        }

        auto& renderPass = gRenderPassList[className];
        return renderPass.create(serializer);
    }

    size_t RenderPassLibrary::getRenderPassCount()
    {
        return gRenderPassList.size();
    }

    const std::string& RenderPassLibrary::getRenderPassClassName(size_t pass)
    {
        assert(pass < getRenderPassCount());
        return std::next(gRenderPassList.begin(), pass)->first;
    }

    const std::string& RenderPassLibrary::getRenderPassDesc(size_t pass)
    {
        assert(pass < getRenderPassCount());
        return std::next(gRenderPassList.begin(), pass)->second.passDesc;
    }
}
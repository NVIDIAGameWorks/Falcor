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
#include "RenderPasses/ForwardLightingPass.h"
#include "Effects/SkyBox/SkyBox.h"
#include "Effects/Shadows/CSM.h"
#include "Effects/ToneMapping/ToneMapping.h"
#include "Effects/FXAA/FXAA.h"
#include "Effects/AmbientOcclusion/SSAO.h"
#include "Effects/TAA/TAA.h"
#include "RenderPasses/ResolvePass.h"

namespace Falcor
{
    struct RenderPassDesc 
    {
        std::string passDesc;
        RenderPassLibrary::CreateFunc create;
    };

    static std::unordered_map<std::string, RenderPassDesc> gRenderPassList;

    template<typename Pass>
    using PassFunc = typename Pass::SharedPtr(*)(const Dictionary&);

#define addClass(c, desc) RenderPassLibrary::addPassClass(#c, desc, (PassFunc<c>)c::create)


    static bool addBuiltinPasses()
    {
        addClass(BlitPass, "Blit one texture into another");
        addClass(ForwardLightingPass, "Forward-rendering lighting pass");
        addClass(DepthPass, "Depth pass");
        addClass(CascadedShadowMaps, "Cascaded shadow maps");
        addClass(ToneMapping, "Tone-Mapping");
        addClass(FXAA, "Fast Approximate Anti-Aliasing");
        addClass(SSAO, "Screen Space Ambient Occlusion");
        addClass(TemporalAA, "Temporal Anti-Aliasing");
        addClass(SkyBox, "Sky Box pass");
        addClass(ResolvePass, "MSAA Resolve");

        return true;
    };

    static const bool b = addBuiltinPasses();

    void RenderPassLibrary::addPassClass(const char* className, const char* desc, CreateFunc func)
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

    std::shared_ptr<RenderPass> RenderPassLibrary::createPass(const char* className, const Dictionary& dict)
    {
        if (gRenderPassList.find(className) == gRenderPassList.end())
        {
            logWarning(std::string("Trying to create a render-pass named `") + className + "`, but no such class exists in the library");
            return nullptr;
        }

        auto& renderPass = gRenderPassList[className];
        return renderPass.create(dict);
    }

    size_t RenderPassLibrary::getClassCount()
    {
        return gRenderPassList.size();
    }

    const std::string& RenderPassLibrary::getClassName(size_t pass)
    {
        assert(pass < getClassCount());
        return std::next(gRenderPassList.begin(), pass)->first;
    }

    const std::string& RenderPassLibrary::getPassDesc(size_t pass)
    {
        assert(pass < getClassCount());
        return std::next(gRenderPassList.begin(), pass)->second.passDesc;
    }

    void RenderPassLibrary::loadPassLibrary(const std::string& filename)
    {
        std::string fullpath;
        if (findFileInDataDirectories(filename, fullpath) == false)
        {
            logWarning("Can't find render-pass library file `" + filename + "`");
            return;
        }

        HMODULE h = LoadLibraryA(fullpath.c_str());
        auto f = (LibraryFunc)GetProcAddress(h, "getPasses");
        std::vector<RenderPassLibrary::RenderPassLibDesc> passesList;
        f(passesList);

        for (const auto& d : passesList) addPassClass(d.className, d.desc, d.func);
    }
}
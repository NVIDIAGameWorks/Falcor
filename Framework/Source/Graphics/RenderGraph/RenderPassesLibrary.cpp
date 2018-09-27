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
#include "API/Device.h"

namespace Falcor
{
    RenderPassLibrary* RenderPassLibrary::spInstance = nullptr;

    template<typename Pass>
    using PassFunc = typename Pass::SharedPtr(*)(const Dictionary&);

#define addClass(c, desc) registerClass(#c, desc, (PassFunc<c>)c::create)

    RenderPassLibrary& RenderPassLibrary::instance()
    {
        if (!spInstance) spInstance = new RenderPassLibrary;
        return *spInstance;
    }

    void RenderPassLibrary::shutdown()
    {
        for (auto& l : mLibs) FreeLibrary(l);
        safe_delete(spInstance);
    }

    static bool addBuiltinPasses()
    {
        auto& lib = RenderPassLibrary::instance();

        lib.addClass(BlitPass, "Blit one texture into another");
        lib.addClass(ForwardLightingPass, "Forward-rendering lighting pass");
        lib.addClass(DepthPass, "Depth pass");
        lib.addClass(CascadedShadowMaps, "Cascaded shadow maps");
        lib.addClass(ToneMapping, "Tone-Mapping");
        lib.addClass(FXAA, "Fast Approximate Anti-Aliasing");
        lib.addClass(SSAO, "Screen Space Ambient Occlusion");
        lib.addClass(TemporalAA, "Temporal Anti-Aliasing");
        lib.addClass(SkyBox, "Sky Box pass");
        lib.addClass(ResolvePass, "MSAA Resolve");

        return true;
    };

    static const bool b = addBuiltinPasses();

    RenderPassLibrary& RenderPassLibrary::registerClass(const char* className, const char* desc, CreateFunc func)
    {
        if (mPasses.find(className) != mPasses.end())
        {
            logWarning(std::string("Trying to register a render-pass `") + className + "` to the render-passes library,  but a render-pass with the same name already exists. Ignoring the new definition");
        }
        else
        {
            mPasses[className] = { desc, func };
        }

        return *this;
    }

    std::shared_ptr<RenderPass> RenderPassLibrary::createPass(const char* className, const Dictionary& dict)
    {
        if (mPasses.find(className) == mPasses.end())
        {
            logWarning(std::string("Trying to create a render-pass named `") + className + "`, but no such class exists in the library");
            return nullptr;
        }

        auto& renderPass = mPasses[className];
        return renderPass.create(dict);
    }

    size_t RenderPassLibrary::getClassCount()
    {
        return mPasses.size();
    }

    const std::string& RenderPassLibrary::getClassName(size_t pass)
    {
        assert(pass < getClassCount());
        return std::next(mPasses.begin(), pass)->first;
    }

    const std::string& RenderPassLibrary::getPassDesc(size_t pass)
    {
        assert(pass < getClassCount());
        return std::next(mPasses.begin(), pass)->second.passDesc;
    }

    void RenderPassLibrary::loadLibrary(const std::string& filename)
    {
        std::string fullpath;
        if (findFileInDataDirectories(filename, fullpath) == false)
        {
            logWarning("Can't load render-pass library `" + filename + "`. File not found");
            return;
        }
        HMODULE l = LoadLibraryA(fullpath.c_str());
        mLibs.push_back(l);
        auto func = (LibraryFunc)GetProcAddress(l, "getPasses");
        func(gpDevice, *this);
    }
}
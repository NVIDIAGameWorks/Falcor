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
#include "stdafx.h"
#include "RenderPassLibrary.h"
#include "RenderPasses/ResolvePass.h"
#include "Core/API/Device.h"
#include "RenderGraph.h"
#include <fstream>

namespace Falcor
{
    extern std::vector<RenderGraph*> gRenderGraphs;
    static const std::string kDllSuffix = ".falcor";

    RenderPassLibrary* RenderPassLibrary::spInstance = nullptr;

    template<typename Pass>
    using PassFunc = typename Pass::SharedPtr(*)(RenderContext* pRenderContext, const Dictionary&);

#define addClass(c, desc) registerClass(#c, desc, (PassFunc<c>)c::create)

    RenderPassLibrary& RenderPassLibrary::instance()
    {
        if (!spInstance) spInstance = new RenderPassLibrary;
        return *spInstance;
    }

    RenderPassLibrary::~RenderPassLibrary()
    {
        mPasses.clear();
        while (mLibs.size()) releaseLibrary(mLibs.begin()->first);
    }

    void RenderPassLibrary::shutdown()
    {
        safe_delete(spInstance);
    }

    static bool addBuiltinPasses()
    {
        auto& lib = RenderPassLibrary::instance();

        lib.addClass(ResolvePass, ResolvePass::kDesc);

        return true;
    };

    static const bool b = addBuiltinPasses();


    RenderPassLibrary& RenderPassLibrary::registerClass(const char* className, const char* desc, CreateFunc func)
    {
        registerInternal(className, desc, func, nullptr);
        return *this;
    }

    void RenderPassLibrary::registerInternal(const char* className, const char* desc, CreateFunc func, DllHandle module)
    {
        if (mPasses.find(className) != mPasses.end())
        {
            logWarning("Trying to register a render-pass '" + std::string(className) + "' to the render-passes library,  but a render-pass with the same name already exists. Ignoring the new definition");
        }
        else
        {
            mPasses[className] = ExtendedDesc(className, desc, func, module);
        }
    }

    std::shared_ptr<RenderPass> RenderPassLibrary::createPass(RenderContext* pRenderContext, const char* className, const Dictionary& dict)
    {
#ifdef _MSC_VER
        static const std::string kDllType = ".dll";
#else
        static const std::string kDllType = ".so";
#endif

        if (mPasses.find(className) == mPasses.end())
        {
            // See if we can load a DLL with the class's name and retry
            std::string libName = className + kDllType;
            logInfo("Can't find a render-pass named '" + std::string(className) + "'. Trying to load a render-pass library '" + libName + "'");
            loadLibrary(libName);

            if (mPasses.find(className) == mPasses.end())
            {
                logWarning("Trying to create a render-pass named '" + std::string(className) + "', but no such class exists in the library");
                return nullptr;
            }
        }

        auto& renderPass = mPasses[className];
        return renderPass.func(pRenderContext, dict);
    }

    RenderPassLibrary::DescVec RenderPassLibrary::enumerateClasses() const
    {
        DescVec v;
        v.reserve(mPasses.size());
        for (const auto& p : mPasses) v.push_back(p.second);
        return v;
    }

    RenderPassLibrary::StrVec RenderPassLibrary::enumerateLibraries()
    {
        StrVec libNames;
        for (const auto& lib : spInstance->mLibs)
        {
            libNames.push_back(lib.first);
        }
        return libNames;
    }

    std::string RenderPassLibrary::getClassDescription(const std::string& className)
    {
        auto classDescIt = spInstance->mPasses.find(className);
        return std::string(classDescIt->second.desc);
    }

    void copyDllFile(const std::string& fullpath)
    {
        std::ifstream src(fullpath, std::ios::binary);
        std::ofstream dst(fullpath + kDllSuffix, std::ios::binary);
        dst << src.rdbuf();
    }

    void RenderPassLibrary::loadLibrary(const std::string& filename)
    {
        std::string fullpath = getExecutableDirectory() + "/" + getFilenameFromPath(filename);

        if (doesFileExist(fullpath) == false)
        {
            logWarning("Can't load render-pass library '" + fullpath + "'. File not found");
            return;
        }

        if (mLibs.find(fullpath) != mLibs.end())
        {
            logInfo("Render-pass library '" + fullpath + "' already loaded. Ignoring 'loadLibrary()' call");
            return;
        }

        // Copy the library to a temp file
        copyDllFile(fullpath);

        DllHandle l = loadDll(fullpath + kDllSuffix);
        mLibs[fullpath] = { l, getFileModifiedTime(fullpath) };
        auto func = (LibraryFunc)getDllProcAddress(l, "getPasses");

        // Add the DLL project directory to the search paths
        if (isDevelopmentMode())
        {
            auto libProjPath = (const char*(*)(void))getDllProcAddress(l, "getProjDir");
            if (libProjPath)
            {
                const char* projDir = libProjPath();
                addDataDirectory(std::string(projDir) + "/Data/");
            }
        }

        RenderPassLibrary lib;
        func(lib);

        for (auto& p : lib.mPasses) registerInternal(p.second.className, p.second.desc, p.second.func, l);
    }

    void RenderPassLibrary::releaseLibrary(const std::string& filename)
    {
        std::string fullpath = getExecutableDirectory() + "/" + getFilenameFromPath(filename);

        auto libIt = mLibs.find(fullpath);
        if (libIt == mLibs.end())
        {
            logWarning("Can't unload render-pass library '" + fullpath + "'. The library wasn't loaded");
            return;
        }

        gpDevice->flushAndSync();

        // Delete all the classes that were owned by the module
        DllHandle module = libIt->second.module;
        for (auto it = mPasses.begin(); it != mPasses.end();)
        {
            if (it->second.module == module) it = mPasses.erase(it);
            else ++it;
        }

        // Remove the DLL project directory to the search paths
        if (isDevelopmentMode())
        {
            auto libProjPath = (const char*(*)(void))getDllProcAddress(module, "getProjDir");
            if (libProjPath)
            {
                const char* projDir = libProjPath();
                removeDataDirectory(std::string(projDir) + "/Data/");
            }
        }

        releaseDll(module);
        std::remove((fullpath + kDllSuffix).c_str());
        mLibs.erase(libIt);
    }

    void RenderPassLibrary::reloadLibrary(RenderContext* pRenderContext, std::string name)
    {
        assert(pRenderContext);

        auto lastTime = getFileModifiedTime(name);
        if ((lastTime == mLibs[name].lastModified) || (lastTime == 0)) return;

        DllHandle module = mLibs[name].module;

        struct PassesToReplace
        {
            RenderGraph* pGraph;
            std::string className;
            uint32_t nodeId;
        };

        std::vector<PassesToReplace> passesToReplace;

        for (auto& passDesc : mPasses)
        {
            if (passDesc.second.module != module) continue;

            // Go over all the graphs and remove this pass
            for (auto& pGraph : gRenderGraphs)
            {
                // Loop over the passes
                for (auto& node : pGraph->mNodeData)
                {
                    if (getClassTypeName(node.second.pPass.get()) == passDesc.first)
                    {
                        passesToReplace.push_back({ pGraph, passDesc.first, node.first });
                        node.second.pPass = nullptr;
                        pGraph->mpExe.reset();
                    }
                }
            }
        }

        // OK, we removed all the passes. Reload the library
        releaseLibrary(name);
        loadLibrary(name);

        // Recreate the passes
        for (auto& r : passesToReplace)
        {
            r.pGraph->mNodeData[r.nodeId].pPass = createPass(pRenderContext, r.className.c_str());
            r.pGraph->mpExe = nullptr;
        }
    }

    void RenderPassLibrary::reloadLibraries(RenderContext* pRenderContext)
    {
        // Copy the libs vector so we don't screw up the iterator
        auto libs = mLibs;
        for (const auto& l : libs) reloadLibrary(pRenderContext, l.first);
    }
}

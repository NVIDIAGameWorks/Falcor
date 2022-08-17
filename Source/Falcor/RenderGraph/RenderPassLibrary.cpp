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
#include "RenderPassLibrary.h"
#include "RenderGraph.h"
#include "RenderPasses/ResolvePass.h"
#include "Core/API/Device.h"
#include "Utils/Scripting/Scripting.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <fstream>

namespace Falcor
{
    extern std::vector<RenderGraph*> gRenderGraphs;
    static const std::string kDllSuffix = ".falcor";

    std::unique_ptr<RenderPassLibrary> RenderPassLibrary::spInstance;

    template<typename Pass>
    using PassFunc = typename Pass::SharedPtr(*)(RenderContext* pRenderContext, const Dictionary&);

    RenderPassLibrary& RenderPassLibrary::instance()
    {
        if (!spInstance) spInstance.reset(new RenderPassLibrary);
        return *spInstance;
    }

    RenderPassLibrary::~RenderPassLibrary()
    {
        mPasses.clear();
        while (mLibs.size()) releaseLibrary(mLibs.begin()->first);
    }

    void RenderPassLibrary::shutdown()
    {
        spInstance.reset();
    }

    static bool addBuiltinPasses()
    {
        auto& lib = RenderPassLibrary::instance();

        lib.registerPass(ResolvePass::kInfo, ResolvePass::create);

        return true;
    };

    static const bool b = addBuiltinPasses();


    RenderPassLibrary& RenderPassLibrary::registerPass(const RenderPass::Info& info, CreateFunc func)
    {
        registerInternal(info, func, nullptr);
        return *this;
    }

    void RenderPassLibrary::registerInternal(const RenderPass::Info& info, CreateFunc func, SharedLibraryHandle library)
    {
        if (mPasses.find(info.type) != mPasses.end())
        {
            logWarning("Trying to register a render-pass '{}' to the render-passes library, but a render-pass with the same name already exists. Ignoring the new definition.", info.type);
        }
        else
        {
            mPasses[info.type] = ExtendedDesc(info, func, library);
        }
    }

    std::shared_ptr<RenderPass> RenderPassLibrary::createPass(RenderContext* pRenderContext, const std::string& type, const Dictionary& dict)
    {
#if FALCOR_WINDOWS
        static const std::string kDllType = ".dll";
#elif FALCOR_LINUX
        static const std::string kDllType = ".so";
#endif

        if (mPasses.find(type) == mPasses.end())
        {
            // See if we can load a DLL with the render passes's type name and retry
            std::string libName = type + kDllType;
            logInfo("Can't find a render-pass named '{}'. Trying to load a render-pass library '{}'.", type, libName);
            loadLibrary(libName);

            if (mPasses.find(type) == mPasses.end())
            {
                logWarning("Trying to create a render-pass named '{}', but no such type exists in the library.", type);
                return nullptr;
            }
        }

        auto& renderPass = mPasses[type];
        return renderPass.func(pRenderContext, dict);
    }

    RenderPassLibrary::DescVec RenderPassLibrary::enumerateClasses() const
    {
        DescVec v;
        v.reserve(mPasses.size());
        for (const auto& p : mPasses) v.push_back(p.second);
        return v;
    }

    std::vector<std::string> RenderPassLibrary::enumerateLibraries()
    {
        std::vector<std::string> result;
        for (const auto& lib : spInstance->mLibs)
        {
            result.push_back(lib.first);
        }
        return result;
    }

    void copySharedLibrary(const std::filesystem::path& srcPath, const std::filesystem::path& dstPath)
    {
        std::ifstream src(srcPath, std::ios::binary);
        if (src.fail())
        {
            reportError(fmt::format("Failed to open '{}' for reading.", srcPath));
            return;
        }

        std::ofstream dst(dstPath, std::ios::binary);
        if (dst.fail())
        {
            logWarning("Failed to open '{}' for writing. It is likely in use by another Falcor instance.", dstPath);
            return;
        }

        dst << src.rdbuf();
        if (dst.fail())
        {
            reportError(fmt::format("An error occurred while copying '{}' to '{}'.", srcPath, dstPath));
        }
    }

    void RenderPassLibrary::loadLibrary(const std::string& filename)
    {
        auto path = getExecutableDirectory() / filename;
#if FALCOR_LINUX
        if (path.extension() == ".dll") path.replace_extension(".so");
#endif

        if (!std::filesystem::exists(path))
        {
            logWarning("Can't load render-pass library '{}'. File not found.", path);
            return;
        }

        if (mLibs.find(filename) != mLibs.end())
        {
            logInfo("Render-pass library '{}' already loaded. Ignoring 'loadLibrary()' call.", filename);
            return;
        }

#if FALCOR_ENABLE_RENDER_PASS_HOT_RELOAD
        // Copy the library to a temp file
        auto copyPath = getExecutableDirectory() / (filename + kDllSuffix);
        copySharedLibrary(path, copyPath);
        SharedLibraryHandle l = loadSharedLibrary(copyPath);
#else
        SharedLibraryHandle l = loadSharedLibrary(path);
#endif
        if (l == nullptr)
        {
            reportError(fmt::format("Failed to load render-pass library '{}'.", filename));
            return;
        }

        mLibs[filename] = { l, getFileModifiedTime(path) };
        auto func = (LibraryFunc)getProcAddress(l, "getPasses");

        // Add the DLL project directory to the search paths
        if (isDevelopmentMode())
        {
            auto libProjPath = (const char*(*)(void))getProcAddress(l, "getProjDir");
            if (libProjPath)
            {
                std::filesystem::path path(libProjPath());
                addDataDirectory(path / "Data");
            }
        }

        RenderPassLibrary lib;
        func(lib);

        for (auto& p : lib.mPasses)
        {
            const auto& desc = p.second;
            registerInternal(desc.info, desc.func, l);
        }

        // Re-import falcor package to current (executing) scripting context.
        auto ctx = Scripting::getCurrentContext();
        if (Scripting::isRunning()) Scripting::runScript("from falcor import *", ctx);
    }

    void RenderPassLibrary::releaseLibrary(const std::string& filename)
    {
        auto libIt = mLibs.find(filename);
        if (libIt == mLibs.end())
        {
            logWarning("Can't unload render-pass library '{}'. The library wasn't loaded.", filename);
            return;
        }

        gpDevice->flushAndSync();

        // Delete all the classes that were owned by the library
        SharedLibraryHandle library = libIt->second.library;
        for (auto it = mPasses.begin(); it != mPasses.end();)
        {
            if (it->second.library == library) it = mPasses.erase(it);
            else ++it;
        }

        // Remove the DLL project directory to the search paths
        if (isDevelopmentMode())
        {
            auto libProjPath = (const char*(*)(void))getProcAddress(library, "getProjDir");
            if (libProjPath)
            {
                std::filesystem::path path(libProjPath());
                removeDataDirectory(path / "Data");
            }
        }

        releaseSharedLibrary(library);
#if FALCOR_ENABLE_RENDER_PASS_HOT_RELOAD
        auto copyPath = getExecutableDirectory() / (filename + kDllSuffix);
        std::filesystem::remove(copyPath);
#endif
        mLibs.erase(libIt);
    }

    void RenderPassLibrary::reloadLibrary(RenderContext* pRenderContext, const std::string& filename)
    {
        FALCOR_ASSERT(pRenderContext);

#if !FALCOR_ENABLE_RENDER_PASS_HOT_RELOAD
        logWarning("Render pass hot reloading is disabled. Check FALCOR_ENABLE_RENDER_PASS_HOT_RELOAD.");
        return;
#endif

        auto path = getExecutableDirectory() / filename;

        auto lastTime = getFileModifiedTime(path);
        if ((lastTime == mLibs[filename].lastModified) || (lastTime == 0)) return;

        SharedLibraryHandle library = mLibs[filename].library;

        struct PassesToReplace
        {
            RenderGraph* pGraph;
            std::string className;
            uint32_t nodeId;
        };

        std::vector<PassesToReplace> passesToReplace;

        for (auto& passDesc : mPasses)
        {
            if (passDesc.second.library != library) continue;

            // Go over all the graphs and remove this pass
            for (auto& pGraph : gRenderGraphs)
            {
                // Loop over the passes
                for (auto& node : pGraph->mNodeData)
                {
                    if (node.second.pPass->getType() == passDesc.first)
                    {
                        passesToReplace.push_back({ pGraph, passDesc.first, node.first });
                        node.second.pPass = nullptr;
                        pGraph->mpExe.reset();
                    }
                }
            }
        }

        // OK, we removed all the passes. Reload the library.
        releaseLibrary(filename);
        loadLibrary(filename);

        // Recreate the passes
        for (auto& r : passesToReplace)
        {
            r.pGraph->mNodeData[r.nodeId].pPass = createPass(pRenderContext, r.className.c_str());
            r.pGraph->mpExe = nullptr;
        }
    }

    void RenderPassLibrary::reloadLibraries(RenderContext* pRenderContext)
    {
#if !FALCOR_ENABLE_RENDER_PASS_HOT_RELOAD
        logWarning("Render pass hot reloading is disabled. Check FALCOR_ENABLE_RENDER_PASS_HOT_RELOAD.");
        return;
#endif

        // Copy the libs vector so we don't screw up the iterator
        auto libs = mLibs;
        for (const auto& l : libs) reloadLibrary(pRenderContext, l.first);
    }
}

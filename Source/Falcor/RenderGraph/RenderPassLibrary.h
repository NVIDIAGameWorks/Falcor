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
#include "Utils/Scripting/Dictionary.h"
#include "RenderPass.h"

namespace Falcor
{
    class dlldecl RenderPassLibrary
    {
    public:
        RenderPassLibrary() = default;
        RenderPassLibrary(RenderPassLibrary&) = delete;
        ~RenderPassLibrary();
        using CreateFunc = std::function<RenderPass::SharedPtr(RenderContext*, const Dictionary&)>;

        struct RenderPassDesc
        {
            RenderPassDesc() = default;
            RenderPassDesc(const char* name, const char* desc_, CreateFunc func_) : className(name), desc(desc_), func(func_) {}

            const char* className = nullptr;
            const char* desc = nullptr;
            CreateFunc func = nullptr;
        };

        using DescVec = std::vector<RenderPassDesc>;

        /** Get an instance of the library. It's a singleton, you'll always get the same object
        */
        static RenderPassLibrary& instance();

        /** Call this before the app is shutting down to release all the libraries
        */
        void shutdown();

        /** Add a new pass class to the library
        */
        RenderPassLibrary& registerClass(const char* className, const char* desc, CreateFunc func);

        /** Instantiate a new render pass object.
            \param[in] pRenderContext The render context.
            \param[in] className Render pass class name.
            \param[in] dict Dictionary for serialized parameters.
            \return A new object, or an exception is thrown if creation failed. Nullptr is returned if class name cannot be found.
        */
        RenderPass::SharedPtr createPass(RenderContext* pRenderContext, const char* className, const Dictionary& dict = {});

        /** Get a list of all the registered classes
        */
        DescVec enumerateClasses() const;

        /** Load a new render-pass DLL
        */
        void loadLibrary(const std::string& filename);

        /** Release a previously loaded DLL
        */
        void releaseLibrary(const std::string& filename);

        /** Reload libraries
        */
        void reloadLibraries(RenderContext* pRenderContext);

        /** A render-pass DLL should implement a function called `getPasses` with the following signature
        */
        using LibraryFunc = void(*)(RenderPassLibrary& lib);

        using StrVec = std::vector<std::string>;

        /** Get list of registered libraries
        */
        static StrVec enumerateLibraries();

        /** Get a description from one existing render pass class
        */
        static std::string getClassDescription(const std::string& className);

    private:
        static RenderPassLibrary* spInstance;

        struct ExtendedDesc : RenderPassDesc
        {
            ExtendedDesc() = default;
            ExtendedDesc(const char* name, const char* desc_, CreateFunc func_, DllHandle module_) : RenderPassDesc(name, desc_, func_), module(module_) {}

            DllHandle module = nullptr;
        };

        void registerInternal(const char* className, const char* desc, CreateFunc func, DllHandle hmodule);

        struct LibDesc
        {
            DllHandle module;
            time_t lastModified;
        };
        std::unordered_map<std::string, LibDesc> mLibs;
        std::unordered_map<std::string, ExtendedDesc> mPasses;

        void reloadLibrary(RenderContext* pRenderContext, std::string name);
    };
}

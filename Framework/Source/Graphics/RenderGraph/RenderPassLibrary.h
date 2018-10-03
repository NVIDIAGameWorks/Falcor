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
#pragma once
#include "RenderPassReflection.h"
#include "Utils/Dictionary.h"

namespace Falcor
{
    class RenderPass;
    class Device;
    class RenderPassLibrary
    {
    public:
        RenderPassLibrary() = default;
        RenderPassLibrary(RenderPassLibrary&) = delete;
        ~RenderPassLibrary();
        using CreateFunc = std::function<std::shared_ptr<RenderPass>(const Dictionary&)>;

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

        /** Instantiate a new render-pass object
        */
        std::shared_ptr<RenderPass> createPass(const char* className, const Dictionary& dict = {});

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
        void reloadLibraries();

        /** A render-pass DLL should implement a function called `getPasses` with the following signature
        */
        using LibraryFunc = void(*)(RenderPassLibrary& lib);

    private:
        static RenderPassLibrary* spInstance;

        struct ExtendeDesc : RenderPassDesc
        {
            ExtendeDesc() = default;
            ExtendeDesc(const char* name, const char* desc_, CreateFunc func_, DllHandle module_) : RenderPassDesc(name, desc, func_), module(module_) {}

            DllHandle module = nullptr;
        };

        void registerInternal(const char* className, const char* desc, CreateFunc func, DllHandle hmodule);

        struct LibDesc
        {
            DllHandle module;
            time_t lastModified;
        };
        std::unordered_map<std::string, LibDesc> mLibs;
        std::unordered_map<std::string, ExtendeDesc> mPasses;

        void reloadLibrary(std::string name);
    };
}
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

        static RenderPassLibrary& instance();
        void shutdown();

        using CreateFunc = std::function<std::shared_ptr<RenderPass>(const Dictionary&)>;
        RenderPassLibrary& registerClass(const char* className, const char* desc, CreateFunc func);
        std::shared_ptr<RenderPass> createPass(const char* className, const Dictionary& dict = {});
        size_t getClassCount();
        const std::string& getPassDesc(size_t pass);
        const std::string& getClassName(size_t pass);

        struct RenderPassLibDesc
        {
            RenderPassLibDesc(const char* name, const char* desc_, CreateFunc func_) : className(name), desc(desc_), func(func_) {}

            const char* className;
            const char* desc;
            CreateFunc func;
        };

        using DeviceSharedPtr = std::shared_ptr<Device>;
        using DescVec = std::vector<RenderPassLibDesc>;

        using LibraryFunc = void(*)(const DeviceSharedPtr&, DescVec& passes);
        void loadLibrary(const std::string& filename);

    private:
        static RenderPassLibrary* spInstance;
        std::vector<HMODULE> mLibs;

        struct RenderPassDesc
        {
            std::string passDesc;
            RenderPassLibrary::CreateFunc create;
        };

        std::unordered_map<std::string, RenderPassDesc> mPasses;
    };
}
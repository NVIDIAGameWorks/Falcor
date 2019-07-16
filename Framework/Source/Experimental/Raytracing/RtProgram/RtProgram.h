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
#include "SingleShaderProgram.h"
#include "HitProgram.h"

namespace Falcor
{
    class ShaderLibrary;

    class RtProgram : public ProgramBase, public std::enable_shared_from_this<RtProgram>
    {
    public:
        using SharedPtr = std::shared_ptr<RtProgram>;
        using SharedConstPtr = std::shared_ptr<const RtProgram>;
        using DefineList = Program::DefineList;

        class Desc
        {
        public:
            Desc() = default;
            Desc(const std::string& filename) { addShaderLibrary(filename); }
            Desc(const std::shared_ptr<ShaderLibrary>& pLibrary) { addShaderLibrary(pLibrary); }

            Desc& addShaderLibrary(const std::shared_ptr<ShaderLibrary>& pLibrary);
            Desc& addShaderLibrary(const std::string& filename);
            Desc& setRayGen(const std::string& raygen);
            Desc& addMiss(uint32_t missIndex, const std::string& miss);
            Desc& addHitGroup(uint32_t hitIndex, const std::string& closestHit, const std::string& anyHit, const std::string& intersection = "");
            Desc& addDefine(const std::string& define, const std::string& value);

            /** Get the compiler flags
            */
            Shader::CompilerFlags getCompilerFlags() const { return shaderFlags; }

            /** Set the compiler flags. Replaces any previously set flags.
            */
            Desc& setCompilerFlags(Shader::CompilerFlags flags) { shaderFlags = flags; return *this; }
        private:
            friend class RtProgram;
            std::vector<std::shared_ptr<ShaderLibrary>> mShaderLibraries;
            DefineList mDefineList;

            struct ShaderEntry
            {
                uint32_t libraryIndex = -1;
                std::string entryPoint;
            };

            ShaderEntry mRayGen;
            std::vector<ShaderEntry> mMiss;

            struct HitProgramEntry
            {
                std::string intersection;
                std::string anyHit;
                std::string closestHit;
                uint32_t libraryIndex = -1;
            };
            std::vector<HitProgramEntry> mHit;
            uint32_t mActiveLibraryIndex = -1;
            Shader::CompilerFlags shaderFlags = Shader::CompilerFlags::None;
        };

        static RtProgram::SharedPtr create(const Desc& desc, uint32_t maxPayloadSize = FALCOR_RT_MAX_PAYLOAD_SIZE_IN_BYTES, uint32_t maxAttributesSize = FALCOR_RT_MAX_ATTRIBUTE_SIZE_IN_BYTES);

        // Ray-gen
        RayGenProgram::SharedPtr getRayGenProgram() const { return mpRayGenProgram; }

        // Hit
        uint32_t getHitProgramCount() const { return (uint32_t)mHitProgs.size(); }
        HitProgram::SharedPtr getHitProgram(uint32_t rayIndex) const { return mHitProgs[rayIndex]; }

        // Miss
        uint32_t getMissProgramCount() const { return (uint32_t)mMissProgs.size(); }
        MissProgram::SharedPtr getMissProgram(uint32_t rayIndex) const { return mMissProgs[rayIndex]; }

        virtual bool addDefine(const std::string& name, const std::string& value = "") override;
        virtual bool addDefines(const DefineList& dl) override;
        virtual bool removeDefine(const std::string& name) override;
        virtual bool removeDefines(const DefineList& dl) override { assert(false); return false; /* not implemented */ }
        virtual bool removeDefines(size_t pos, size_t len, const std::string& str) override;
        virtual bool setDefines(const DefineList& dl) override;
        virtual const DefineList& getDefines() const override { assert(false); static DefineList dummy; return dummy; /* not well defined if the ray programs have mismatching set of defines */ }

        const std::shared_ptr<RootSignature>& getGlobalRootSignature() const { updateReflection(); return mpGlobalRootSignature; }
        const std::shared_ptr<ProgramReflection>& getGlobalReflector() const { updateReflection(); return mpGlobalReflector; }

    private:

        using MissProgramList = std::vector<MissProgram::SharedPtr>;
        using HitProgramList = std::vector<HitProgram::SharedPtr>;

        void updateReflection() const;

        RtProgram(const Desc& desc, uint32_t maxPayloadSize = FALCOR_RT_MAX_PAYLOAD_SIZE_IN_BYTES, uint32_t maxAttributesSize = FALCOR_RT_MAX_ATTRIBUTE_SIZE_IN_BYTES);
        HitProgramList mHitProgs;
        MissProgramList mMissProgs;
        RayGenProgram::SharedPtr mpRayGenProgram;

        mutable bool mReflectionDirty = true;
        mutable std::shared_ptr<RootSignature> mpGlobalRootSignature;
        mutable std::shared_ptr<ProgramReflection> mpGlobalReflector;
    };
}
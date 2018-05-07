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
#include "Graphics/Program/ProgramVersion.h"
#include "../RtShader.h"
#include "API/VAO.h"
#include "Utils/Graph.h"

namespace Falcor
{
    class RootSignature;

    // The inheritence here is just so that we could work with Program. We're not actually using anything from ProgramVersion
    class RtProgramKernels : public ProgramKernels, inherit_shared_from_this<ProgramKernels, RtProgramKernels>
    {
    public:
        using SharedPtr = std::shared_ptr<RtProgramKernels>;
        using SharedConstPtr = std::shared_ptr<const RtProgramKernels>;

        ~RtProgramKernels() = default;

        enum class Type
        {
            RayGeneration,
            Hit,
            Miss,
        };

        static SharedPtr createRayGen(RtShader::SharedPtr pRayGenShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pLocalReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize);
        static SharedPtr createMiss(RtShader::SharedPtr pMissShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pLocalReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize);
        static SharedPtr createHit(RtShader::SharedPtr pAnyHit, RtShader::SharedPtr pClosestHit, RtShader::SharedPtr pIntersection, std::string& log, const std::string& name, ProgramReflection::SharedPtr pLocalReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize);

        RtShader::SharedConstPtr getShader(ShaderType type) const;

        std::shared_ptr<RootSignature> getLocalRootSignature() const { return getRootSignature(); }

        const std::wstring& getExportName() const { return mExportName; }

        Type getType() const { return mType; }

        uint32_t getMaxPayloadSize() const { return mMaxPayloadSize; }
        uint32_t getMaxAttributesSize() const { return mMaxAttributeSize; }

    private:
        template<ShaderType shaderType>
        static SharedPtr createSingleShaderProgram(RtShader::SharedPtr pShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pLocalReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize);
        bool initCommon(std::string& log);

        RtProgramKernels(std::shared_ptr<ProgramReflection> pReflector, Type progType, Shader::SharedPtr const* ppShaders, size_t shaderCount, const std::string& name, uint32_t maxPayloadSize, uint32_t maxAttributeSize, std::string const& exportName);
        Type mType;
        std::wstring mExportName;

        static uint64_t sProgId;
        uint32_t mMaxPayloadSize;
        uint32_t mMaxAttributeSize;
    };

    inline std::string to_string(RtProgramKernels::Type t)
    {
        switch (t)
        {
        case RtProgramKernels::Type::RayGeneration:
            return "RayGen";
        case RtProgramKernels::Type::Hit:
            return "Hit";
        case RtProgramKernels::Type::Miss:
            return "Miss";
        default:
            should_not_get_here();
            return "";
        }
    }

    class RtPipelineKernels : public std::enable_shared_from_this<RtPipelineKernels>
    {
    public:
        using SharedPtr = std::shared_ptr<RtPipelineKernels>;
        using SharedConstPtr = std::shared_ptr<const RtPipelineKernels>;

        using ProgramList = std::vector<RtProgramKernels::SharedConstPtr>;

        ~RtPipelineKernels() = default;

        static RtPipelineKernels::SharedPtr create(
            ProgramKernels::SharedConstPtr const& pGlobalProgram,
            RtProgramKernels::SharedConstPtr const& pRayGenProgram,
            std::vector<RtProgramKernels::SharedConstPtr> const& hitPrograms,
            std::vector<RtProgramKernels::SharedConstPtr> const& missPrograms);

        const std::shared_ptr<RootSignature>& getGlobalRootSignature() const { return mpGlobalProgram->getRootSignature(); }
        ProgramList const& getProgramList() const { return mProgramList; }

        size_t getHitProgramCount() const { return mHitPrograms.size(); }
        size_t getMissProgramCount() const { return mMissPrograms.size(); }

        ProgramKernels::SharedConstPtr getGlobalProgram() const { return mpGlobalProgram; }
        RtProgramKernels::SharedConstPtr getRayGenProgram() const { return mpRayGenProgram; }
        RtProgramKernels::SharedConstPtr getHitProgram(size_t i) const { return mHitPrograms[i]; }
        RtProgramKernels::SharedConstPtr getMissProgram(size_t i) const { return mMissPrograms[i]; }

        uint32_t getRecordSize() const { return mRecordSize; }

    private:
        RtPipelineKernels(
            ProgramKernels::SharedConstPtr const& pGlobalProgram,
            RtProgramKernels::SharedConstPtr const& pRayGenProgram,
            std::vector<RtProgramKernels::SharedConstPtr> const& hitPrograms,
            std::vector<RtProgramKernels::SharedConstPtr> const& missPrograms);

        void init();

        ProgramList mProgramList;

        ProgramKernels::SharedConstPtr mpGlobalProgram;
        RtProgramKernels::SharedConstPtr mpRayGenProgram;
        std::vector<RtProgramKernels::SharedConstPtr> mHitPrograms;
        std::vector<RtProgramKernels::SharedConstPtr> mMissPrograms;

        uint32_t mRecordSize;
    };

    class RtPipelineVersion : public std::enable_shared_from_this<RtPipelineVersion>
    {
    public:
        using SharedPtr = std::shared_ptr<RtPipelineVersion>;
        using SharedConstPtr = std::shared_ptr<const RtPipelineVersion>;

        ~RtPipelineVersion() = default;

        static RtPipelineVersion::SharedPtr create(
            std::shared_ptr<ProgramReflection> pGlobalReflector,
            ProgramVersion::SharedConstPtr pRayGenProgram,
            std::vector<ProgramVersion::SharedConstPtr> hitPrograms,
            std::vector<ProgramVersion::SharedConstPtr> missPrograms);

        const std::shared_ptr<ProgramReflection>& getGlobalReflector() const { return mpGlobalReflector; }

        ProgramVersion::SharedConstPtr getRayGenProgram() const { return mpRayGenProgram; }
        ProgramVersion::SharedConstPtr getHitProgram(size_t i) const { return mHitPrograms[i]; }
        ProgramVersion::SharedConstPtr getMissProgram(size_t i) const { return mMissPrograms[i]; }

        RtPipelineKernels::SharedPtr getKernels(RtProgramVars* pVars) const;

    private:
        RtPipelineVersion(
            std::shared_ptr<ProgramReflection> pGlobalReflector,
            ProgramVersion::SharedConstPtr pRayGenProgram,
            std::vector<ProgramVersion::SharedConstPtr> hitPrograms,
            std::vector<ProgramVersion::SharedConstPtr> missPrograms);
        bool init();

        std::shared_ptr<ProgramReflection> mpGlobalReflector;
        ProgramKernels::SharedPtr mpGlobalKernels;

        ProgramVersion::SharedConstPtr mpRayGenProgram;
        std::vector<ProgramVersion::SharedConstPtr> mHitPrograms;
        std::vector<ProgramVersion::SharedConstPtr> mMissPrograms;

        using KernelGraph = Graph<RtPipelineKernels::SharedPtr, void*>;
        KernelGraph::SharedPtr mpKernelGraph;
    };
}

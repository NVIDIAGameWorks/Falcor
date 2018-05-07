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
#include "RtProgramVersion.h"
#include "API/Device.h"
#include "API/D3D12/D3D12State.h"
#include "Utils/StringUtils.h"

namespace Falcor
{
    uint64_t RtProgramKernels::sProgId = 0;
    ProgramReflection::SharedPtr createProgramReflection(const Shader::SharedConstPtr pShaders[], std::string& log);

    RtProgramKernels::RtProgramKernels(std::shared_ptr<ProgramReflection> pReflector, Type progType, Shader::SharedPtr const* ppShaders, size_t shaderCount,  const std::string& name, uint32_t maxPayloadSize, uint32_t maxAttributeSize, std::string const& exportName)
        : ProgramKernels(pReflector, ppShaders, shaderCount, RootSignature::create(pReflector.get(), true), name)
        , mType(progType)
        , mMaxAttributeSize(maxAttributeSize)
        , mMaxPayloadSize(maxPayloadSize)
        , mExportName(string_2_wstring(exportName))
    {
        for( size_t ii = 0; ii < shaderCount; ++ii )
        {
            auto pShader = ppShaders[ii];
            mpShaders[uint32_t(pShader->getType())] = pShader;
        }

#if 0
        switch (mType)
        {
        case Falcor::RtProgramKernels::Type::RayGeneration:
            mExportName = string_2_wstring(ppShaders[0]->getEntryPoint());
            break;
        case Falcor::RtProgramKernels::Type::Hit:
            mExportName = L"RtProgramVersion" + std::to_wstring(sProgId++);
            break;
        case Falcor::RtProgramKernels::Type::Miss:
            mExportName = string_2_wstring(ppShaders[0]->getEntryPoint());
            break;
        default:
            should_not_get_here();
        }
#endif
    }

    bool RtProgramKernels::initCommon(std::string& log)
    {
        if (init(log) == false)
        {
            return false;
        }

        // Create the root signature
//        mpLocalRootSignature = RootSignature::create(mpReflector.get(), true);
        return true;
    }

    RtProgramKernels::Type getProgTypeFromShader(ShaderType shaderType)
    {
        switch (shaderType)
        {
        case ShaderType::Miss:
            return RtProgramKernels::Type::Miss;
        case ShaderType::RayGeneration:
            return RtProgramKernels::Type::RayGeneration;
        default:
            should_not_get_here();
            return RtProgramKernels::Type(-1);
        }
    }
    
    template<ShaderType shaderType>
    RtProgramKernels::SharedPtr RtProgramKernels::createSingleShaderProgram(RtShader::SharedPtr pShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pLocalReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize)
    {
        // We are using the RayGeneration structure in the union to avoid code duplication, these asserts make sure that our assumptions are correct

        if (pShader == nullptr)
        {
            log = to_string(shaderType) + " shader is null. Can't create a " + to_string(shaderType) + " RtProgramVersion";
            return nullptr;
        }

        Shader::SharedPtr ppShaders[] = { pShader };

        auto exportName = pShader->getEntryPoint();
        SharedPtr pProgram = SharedPtr(new RtProgramKernels(pLocalReflector, getProgTypeFromShader(shaderType), ppShaders, 1, name, maxPayloadSize, maxAttributeSize, exportName));
        if (pProgram->initCommon(log) == false)
        {
            return nullptr;
        }

        return pProgram;
    }

    RtShader::SharedConstPtr RtProgramKernels::getShader(ShaderType type) const
    {
        Shader::SharedConstPtr pShader = mpShaders[(uint32_t)type];
        RtShader::SharedConstPtr pRtShader = std::dynamic_pointer_cast<const RtShader>(pShader);
        assert(!pShader || pRtShader);
        return pRtShader;
    }

    RtProgramKernels::SharedPtr RtProgramKernels::createRayGen(RtShader::SharedPtr pRayGenShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pLocalReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize)
    {
        return createSingleShaderProgram<ShaderType::RayGeneration>(pRayGenShader, log, name, pLocalReflector, maxPayloadSize, maxAttributeSize);
    }

    RtProgramKernels::SharedPtr RtProgramKernels::createMiss(RtShader::SharedPtr pMissShader, std::string& log, const std::string& name, ProgramReflection::SharedPtr pLocalReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize)
    {
        return createSingleShaderProgram<ShaderType::Miss>(pMissShader, log, name, pLocalReflector, maxPayloadSize, maxAttributeSize);
    }
    
    RtProgramKernels::SharedPtr RtProgramKernels::createHit(RtShader::SharedPtr pAnyHit, RtShader::SharedPtr pClosestHit, RtShader::SharedPtr pIntersection, std::string& log, const std::string& name, ProgramReflection::SharedPtr pReflector, uint32_t maxPayloadSize, uint32_t maxAttributeSize)
    {
        size_t shaderCount = 0;
        Shader::SharedPtr pShaders[3];

        if(pAnyHit)         pShaders[shaderCount++] = pAnyHit;
        if(pClosestHit)     pShaders[shaderCount++] = pClosestHit;
        if(pIntersection)   pShaders[shaderCount++] = pIntersection;

        if (shaderCount == 0)
        {
            log = "Error when creating " + to_string(Type::Hit) + " RtProgramVersion for program" + name + ". At least one of the shaders must be valid.";
            return nullptr;
        }

        auto exportName = "RtHitGroup" + std::to_string(sProgId++);
        SharedPtr pProgram = SharedPtr(new RtProgramKernels(pReflector, Type::Hit, pShaders, shaderCount, name, maxPayloadSize, maxAttributeSize, exportName));
        if (pProgram->initCommon(log) == false)
        {
            return nullptr;
        }

        return pProgram;
    }

    RtPipelineKernels::SharedPtr RtPipelineKernels::create(
        ProgramKernels::SharedConstPtr const& pGlobalProgram,
        RtProgramKernels::SharedConstPtr const& pRayGenProgram,
        std::vector<RtProgramKernels::SharedConstPtr> const& hitPrograms,
        std::vector<RtProgramKernels::SharedConstPtr> const& missPrograms)
    {
        auto pKernels = SharedPtr(new RtPipelineKernels(pGlobalProgram, pRayGenProgram, hitPrograms, missPrograms));
        pKernels->init();
        return pKernels;
    }

    RtPipelineKernels::RtPipelineKernels(
        ProgramKernels::SharedConstPtr const& pGlobalProgram,
        RtProgramKernels::SharedConstPtr const& pRayGenProgram,
        std::vector<RtProgramKernels::SharedConstPtr> const& hitPrograms,
        std::vector<RtProgramKernels::SharedConstPtr> const& missPrograms)
        : mpGlobalProgram(pGlobalProgram)
        , mRecordSize(0)
        , mpRayGenProgram(pRayGenProgram)
        , mHitPrograms(hitPrograms)
        , mMissPrograms(missPrograms)
    {}

    void RtPipelineKernels::init()
    {
        uint32_t maxRootSigSize = 0;

        auto addProgramAndComputeMaxSize = [&](RtProgramKernels::SharedConstPtr const& pProgram)
        {
            mProgramList.push_back(pProgram);
            maxRootSigSize = std::max(maxRootSigSize, pProgram->getLocalRootSignature()->getSizeInBytes());
        };

        addProgramAndComputeMaxSize(mpRayGenProgram);

        for( auto pHitProgram : mHitPrograms )
        {
            addProgramAndComputeMaxSize(pHitProgram);
        }
        for( auto pMissProgram : mMissPrograms )
        {
            addProgramAndComputeMaxSize(pMissProgram);
        }

        // Get the program identifier size
        ID3D12DeviceRaytracingPrototypePtr pRtDevice = gpDevice->getApiHandle();
        uint32_t programIdentifierSize = pRtDevice->GetShaderIdentifierSize();

        // Calculate the record size
        uint32_t recordSize = programIdentifierSize + maxRootSigSize;
        recordSize = align_to(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, recordSize);
        assert(recordSize != 0);

        mRecordSize = recordSize;
    }

    RtPipelineVersion::SharedPtr RtPipelineVersion::create(
        std::shared_ptr<ProgramReflection> pGlobalReflector,
        ProgramVersion::SharedConstPtr pRayGenProgram,
        std::vector<ProgramVersion::SharedConstPtr> hitPrograms,
        std::vector<ProgramVersion::SharedConstPtr> missPrograms)
    {
        auto result = SharedPtr(new RtPipelineVersion(
            pGlobalReflector,
            pRayGenProgram,
            hitPrograms,
            missPrograms));

        if( !result->init() )
        {
            return nullptr;
        }
        return result;
    }

    RtPipelineVersion::RtPipelineVersion(
        std::shared_ptr<ProgramReflection> pGlobalReflector,
        ProgramVersion::SharedConstPtr pRayGenProgram,
        std::vector<ProgramVersion::SharedConstPtr> hitPrograms,
        std::vector<ProgramVersion::SharedConstPtr> missPrograms)
        : mpGlobalReflector(pGlobalReflector)
        , mpRayGenProgram(pRayGenProgram)
        , mHitPrograms(hitPrograms)
        , mMissPrograms(missPrograms)
    {}

    bool RtPipelineVersion::init()
    {
        std::string log;
        auto pGlobalReflector = mpGlobalReflector;
        auto pGlobalRootSignature = RootSignature::create(pGlobalReflector.get());
        if(!pGlobalRootSignature)
            return false;

        auto pGlobalKernels = ProgramKernels::create(
            pGlobalReflector,
            nullptr,
            0,
            pGlobalRootSignature,
            log,
            "ray tracing global parameters");
        if(!pGlobalKernels)
            return false;

        mpGlobalKernels = pGlobalKernels;
        mpKernelGraph = KernelGraph::create();

        return true;
    }
}

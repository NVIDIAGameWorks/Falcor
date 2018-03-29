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
#include "RtProgram.h"
#include "API/LowLevel/RootSignature.h"

namespace Falcor
{
    RtProgram::Desc& RtProgram::Desc::setShaderModule(const ShaderModule::SharedPtr& pModule)
    {
        if (mpModule)
        {
            logWarning("RtProgram::Desc::setShaderModule() - a module already exists. Replacing the old module");
        }
        mpModule = pModule;
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::setFilename(const std::string& filename)
    {
        if (mpModule)
        {
            logWarning("RtProgram::Desc::setFilename() - a module already exists. Replacing the old module");
        }
        mpModule = ShaderModule::create(filename);
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::setRayGen(const std::string& raygen)
    {
        if (mRayGen.size())
        {
            logWarning("RtProgram::Desc::setRayGen() - a ray-generation entry point is already set. Replacing the old entry entry-point");
        }
        mRayGen = raygen;
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addMiss(const std::string& miss)
    {
        mMiss.push_back(miss);
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addHitGroup(const std::string& closestHit, const std::string& anyHit, const std::string& intersection)
    {
        HitProgramEntry entry;
        entry.anyHit = anyHit;
        entry.closestHit = closestHit;
        entry.intersection = intersection;
        mHit.push_back(entry);
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addDefine(const std::string& name, const std::string& value)
    {
        mDefineList.add(name, value);
        return *this;
    }

    RtProgram::SharedPtr RtProgram::create(const Desc& desc, uint32_t maxPayloadSize, uint32_t maxAttributesSize)
    {
        SharedPtr pProg = SharedPtr(new RtProgram(desc, maxPayloadSize, maxAttributesSize));
        pProg->addDefine("_MS_DISABLE_ALPHA_TEST");

        return pProg;
    }

    RtProgram::RtProgram(const Desc& desc, uint32_t maxPayloadSize, uint32_t maxAttributesSize)
    {
        const std::string& filename = desc.mpModule->getFilename();
        
        // Create the programs
        mpRayGenProgram = RayGenProgram::createFromFile(filename.c_str(), desc.mRayGen.c_str(), desc.mDefineList, maxPayloadSize, maxAttributesSize);

        for (const auto& m : desc.mMiss)
        {
            mMissProgs.push_back(MissProgram::createFromFile(filename.c_str(), m.c_str(), desc.mDefineList, maxPayloadSize, maxAttributesSize));
        }

        for (const auto& h : desc.mHit)
        {
            HitProgram::SharedPtr pHit = HitProgram::createFromFile(filename.c_str(), h.closestHit, h.anyHit, h.intersection, desc.mDefineList, maxPayloadSize, maxAttributesSize);
            mHitProgs.push_back(pHit);
        }

        // Create the global reflector and root-signature
        mpGlobalReflector = ProgramReflection::create(nullptr, ProgramReflection::ResourceScope::Global, std::string());
        mpGlobalReflector->merge(mpRayGenProgram->getGlobalReflector().get());

        for (const auto m : mMissProgs)
        {
            mpGlobalReflector->merge(m->getGlobalReflector().get());
        }

        for (const auto& h : mHitProgs)
        {
            mpGlobalReflector->merge(h->getGlobalReflector().get());
        }

        mpGlobalRootSignature = RootSignature::create(mpGlobalReflector.get(), false);
    }

    void RtProgram::addDefine(const std::string& name, const std::string& value /*= ""*/)
    {
        if(mpRayGenProgram)
        {
            mpRayGenProgram->addDefine(name, value);
        }

        for (auto& pHit : mHitProgs)
        {
            pHit->addDefine(name, value);
        }

        for (auto& pMiss : mMissProgs)
        {
            pMiss->addDefine(name, value);
        }
    }

    void RtProgram::removeDefine(const std::string& name)
    {
        if (mpRayGenProgram)
        {
            mpRayGenProgram->removeDefine(name);
        }

        for (auto& pHit : mHitProgs)
        {
            pHit->addDefine(name);
        }

        for (auto& pMiss : mMissProgs)
        {
            pMiss->addDefine(name);
        }
    }
}

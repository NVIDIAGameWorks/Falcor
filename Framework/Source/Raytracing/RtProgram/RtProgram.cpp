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
#include "Graphics/Program/ShaderLibrary.h"

namespace Falcor
{
    static bool checkValidLibrary(uint32_t activeIndex, const std::string& shader)
    {
        if (activeIndex == -1)
        {
            logWarning("Can't set " + shader + " entry-point. Please add a shader-library first");
            return false;
        }
        return true;
    }

    RtProgram::Desc& RtProgram::Desc::addShaderLibrary(const ShaderLibrary::SharedPtr& pLibrary)
    {
        if (pLibrary == nullptr)
        {
            logWarning("Can't add a null library to RtProgram::Desc");
            return *this;
        }
        mActiveLibraryIndex = (uint32_t)mShaderLibraries.size();
        mShaderLibraries.emplace_back(pLibrary);
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addShaderLibrary(const std::string& filename)
    {
        return addShaderLibrary(ShaderLibrary::create(filename));
    }

    RtProgram::Desc& RtProgram::Desc::setRayGen(const std::string& raygen)
    {
        if (!checkValidLibrary(mActiveLibraryIndex, "raygen")) return *this;
        if (mRayGen.libraryIndex != -1)
        {
            logWarning("RtProgram::Desc::setRayGen() - a ray-generation entry point is already set. Replacing the old entry-point");
        }
        mRayGen.libraryIndex = mActiveLibraryIndex;
        mRayGen.entryPoint = raygen;
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addMiss(uint32_t missIndex, const std::string& miss)
    {
        if (!checkValidLibrary(mActiveLibraryIndex, "miss")) return *this;
        if (mMiss.size() <= missIndex)
        {
            mMiss.resize(missIndex + 1);
        }
        else
        {
            if (mMiss[missIndex].libraryIndex != -1)
            {
                logWarning("RtProgram::Desc::addMiss() - a miss entry point already exists for index " + std::to_string(missIndex) + ". Replacing the old entry-point");
            }
        }
        mMiss[missIndex].libraryIndex = mActiveLibraryIndex;
        mMiss[missIndex].entryPoint = miss;

        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addHitGroup(uint32_t hitIndex, const std::string& closestHit, const std::string& anyHit, const std::string& intersection /* = "" */)
    {
        if (!checkValidLibrary(mActiveLibraryIndex, "his")) return *this;
        if (mHit.size() <= hitIndex)
        {
            mHit.resize(hitIndex + 1);
        }
        else
        {
            if (mHit[hitIndex].libraryIndex != -1)
            {
                logWarning("RtProgram::Desc::addHitGroup() - a hit-group already exists for index " + std::to_string(hitIndex) + ". Replacing the old group");
            }
        }
        mHit[hitIndex].anyHit = anyHit;
        mHit[hitIndex].closestHit = closestHit;
        mHit[hitIndex].intersection = intersection;
        mHit[hitIndex].libraryIndex = mActiveLibraryIndex;
        return *this;
    }

    RtProgram::Desc& RtProgram::Desc::addDefine(const std::string& name, const std::string& value)
    {
        mDefineList.add(name, value);
        return *this;
    }

    RtProgram::SharedPtr RtProgram::create(const Desc& desc, uint32_t maxPayloadSize, uint32_t maxAttributesSize)
    {
        if (desc.mRayGen.libraryIndex == -1)
        {
            logError("Can't create an RtProgram without a ray-generation shader");
            return nullptr;
        }

        SharedPtr pProg = SharedPtr(new RtProgram(desc, maxPayloadSize, maxAttributesSize));
        pProg->addDefine("_MS_DISABLE_ALPHA_TEST");

        return pProg;
    }

    void RtProgram::updateReflection() const
    {
        if (mReflectionDirty)
        {
            // Create the global reflector and root-signature
            mpGlobalReflector = ProgramReflection::create(nullptr, ProgramReflection::ResourceScope::Global, std::string());
            mpGlobalReflector->merge(mpRayGenProgram->getGlobalReflector().get());

            for (const auto m : mMissProgs)
            {
                if (m) mpGlobalReflector->merge(m->getGlobalReflector().get());
            }

            for (const auto& h : mHitProgs)
            {
                if (h) mpGlobalReflector->merge(h->getGlobalReflector().get());
            }

            mpGlobalRootSignature = RootSignature::create(mpGlobalReflector.get(), false);

            mReflectionDirty = false;
        }
    }

    RtProgram::RtProgram(const Desc& desc, uint32_t maxPayloadSize, uint32_t maxAttributesSize)
    {
        // Create the programs
        const std::string raygenFile = desc.mShaderLibraries[desc.mRayGen.libraryIndex]->getFilename();
        mpRayGenProgram = RayGenProgram::createFromFile(raygenFile.c_str(), desc.mRayGen.entryPoint.c_str(), desc.mDefineList, maxPayloadSize, maxAttributesSize, desc.getCompilerFlags());

        mMissProgs.resize(desc.mMiss.size());
        for (size_t i = 0 ; i < desc.mMiss.size() ; i++)
        {
            const auto& m = desc.mMiss[i];

            if (m.libraryIndex != -1)
            {
                const std::string missFile = desc.mShaderLibraries[m.libraryIndex]->getFilename();
                mMissProgs[i] = MissProgram::createFromFile(missFile.c_str(), m.entryPoint.c_str(), desc.mDefineList, maxPayloadSize, maxAttributesSize, desc.getCompilerFlags());
            }
        }

        mHitProgs.resize(desc.mHit.size());
        for (size_t i = 0 ; i < desc.mHit.size() ; i++)
        {
            const auto& h = desc.mHit[i];
            if(h.libraryIndex != -1)
            {
                const std::string hitFile = desc.mShaderLibraries[h.libraryIndex]->getFilename();
                mHitProgs[i] = HitProgram::createFromFile(hitFile.c_str(), h.closestHit, h.anyHit, h.intersection, desc.mDefineList, maxPayloadSize, maxAttributesSize, desc.getCompilerFlags());
            }
        }
    }

    bool RtProgram::addDefine(const std::string& name, const std::string& value /*= ""*/)
    {
        bool changed = false;
        if(mpRayGenProgram && mpRayGenProgram->addDefine(name, value)) changed = true;

        for (auto& pHit : mHitProgs)
        {
            if (pHit && pHit->addDefine(name, value)) changed = true;
        }

        for (auto& pMiss : mMissProgs)
        {
            if (pMiss && pMiss->addDefine(name, value)) changed = true;
        }

        mReflectionDirty = changed;
        return changed;
    }

    bool RtProgram::addDefines(const DefineList& dl)
    {
        bool changed = false;
        if (mpRayGenProgram && mpRayGenProgram->addDefines(dl)) changed = true;

        for (auto& pHit : mHitProgs)
        {
            if (pHit && pHit->addDefines(dl)) changed = true;
        }

        for (auto& pMiss : mMissProgs)
        {
            if (pMiss && pMiss->addDefines(dl)) changed = true;
        }

        mReflectionDirty = changed;
        return changed;
    }

    bool RtProgram::removeDefine(const std::string& name)
    {
        bool changed = false;
        if (mpRayGenProgram && mpRayGenProgram->removeDefine(name)) changed = true;

        for (auto& pHit : mHitProgs)
        {
            if (pHit && pHit->removeDefine(name)) changed = true;
        }

        for (auto& pMiss : mMissProgs)
        {
            if (pMiss && pMiss->removeDefine(name)) changed = true;
        }

        mReflectionDirty = changed;
        return changed;
    }

    bool RtProgram::removeDefines(size_t pos, size_t len, const std::string& str)
    {
        bool changed = false;
        if (mpRayGenProgram && mpRayGenProgram->removeDefines(pos, len, str)) changed = true;

        for (auto& pHit : mHitProgs)
        {
            if (pHit && pHit->removeDefines(pos, len, str)) changed = true;
        }

        for (auto& pMiss : mMissProgs)
        {
            if (pMiss && pMiss->removeDefines(pos, len, str)) changed = true;
        }

        mReflectionDirty = changed;
        return changed;
    }

    bool RtProgram::setDefines(const DefineList& dl)
    {
        bool changed = false;
        if (mpRayGenProgram && mpRayGenProgram->setDefines(dl)) changed = true;

        for (auto& pHit : mHitProgs)
        {
            if (pHit && pHit->setDefines(dl)) changed = true;
        }

        for (auto& pMiss : mMissProgs)
        {
            if (pMiss && pMiss->setDefines(dl)) changed = true;
        }

        mReflectionDirty = changed;
        return changed;
    }
}

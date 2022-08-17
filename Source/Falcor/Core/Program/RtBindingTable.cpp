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
#include "RtBindingTable.h"
#include "Core/Errors.h"

namespace Falcor
{
    namespace
    {
        // Define API limitations.
        // See https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html
        const uint32_t kMaxMissCount = (1 << 16);
        const uint32_t kMaxRayTypeCount = (1 << 4);
    }

    RtBindingTable::SharedPtr RtBindingTable::create(uint32_t missCount, uint32_t rayTypeCount, uint32_t geometryCount)
    {
        return SharedPtr(new RtBindingTable(missCount, rayTypeCount, geometryCount));
    }

    RtBindingTable::RtBindingTable(uint32_t missCount, uint32_t rayTypeCount, uint32_t geometryCount)
        : mMissCount(missCount)
        , mRayTypeCount(rayTypeCount)
        , mGeometryCount(geometryCount)
    {
        if (missCount > kMaxMissCount)
        {
            throw ArgumentError("'missCount' exceeds the maximum supported ({})", kMaxMissCount);
        }
        if (rayTypeCount > kMaxRayTypeCount)
        {
            throw ArgumentError("'rayTypeCount' exceeds the maximum supported ({})", kMaxRayTypeCount);
        }

        size_t recordCount = 1ull + missCount + rayTypeCount * geometryCount;
        if (recordCount > std::numeric_limits<uint32_t>::max())
        {
            throw ArgumentError("Raytracing binding table is too large");
        }

        // Create the binding table. All entries will be assigned a null shader initially.
        mShaderTable.resize(recordCount);
    }

    void RtBindingTable::setRayGen(ShaderID shaderID)
    {
        mShaderTable[0] = shaderID;
    }

    void RtBindingTable::setMiss(uint32_t missIndex, ShaderID shaderID)
    {
        if (missIndex >= mMissCount)
        {
            throw ArgumentError("'missIndex' is out of range");
        }
        mShaderTable[getMissOffset(missIndex)] = shaderID;
    }

    void RtBindingTable::setHitGroup(uint32_t rayType, uint32_t geometryID, ShaderID shaderID)
    {
        if (rayType >= mRayTypeCount)
        {
            throw ArgumentError("'rayType' is out of range");
        }
        if (geometryID >= mGeometryCount)
        {
            throw ArgumentError("'geometryID' is out of range");
        }
        mShaderTable[getHitGroupOffset(rayType, geometryID)] = shaderID;
    }

    void RtBindingTable::setHitGroup(uint32_t rayType, const std::vector<uint32_t>& geometryIDs, ShaderID shaderID)
    {
        for (uint32_t geometryID : geometryIDs)
        {
            setHitGroup(rayType, geometryID, shaderID);
        }
    }

    void RtBindingTable::setHitGroup(uint32_t rayType, const std::vector<GlobalGeometryID>& geometryIDs, ShaderID shaderID)
    {
        static_assert(std::is_same_v<GlobalGeometryID::IntType, uint32_t>);
        for (GlobalGeometryID geometryID : geometryIDs)
        {
            setHitGroup(rayType, geometryID.get(), shaderID);
        }
    }
}

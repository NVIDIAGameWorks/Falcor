/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "HitInfo.h"
#include "HitInfoType.slang"
#include "Scene.h"

namespace Falcor
{
    namespace
    {
        uint32_t allocateBits(const uint32_t count)
        {
            if (count <= 1) return 0;
            uint32_t maxValue = count - 1;
            return bitScanReverse(maxValue) + 1;
        }
    }

    void HitInfo::init(const Scene& scene)
    {
        // Setup bit allocations for encoding the hit information.
        // The shader code will choose either a 64-bit or 96-bit format depending on requirements.
        // The barycentrics are encoded in 32 bits, while instance type and instance/primitive index are encoded in 32-64 bits.

        uint32_t typeCount = (uint32_t)HitType::Count;
        mTypeBits = allocateBits(typeCount);

        uint32_t instanceCount = scene.getMeshInstanceCount() + scene.getCurveInstanceCount();
        mInstanceIndexBits = allocateBits(instanceCount);

        uint32_t maxPrimitiveCount = 0;

        for (uint32_t meshID = 0; meshID < scene.getMeshCount(); meshID++)
        {
            uint32_t triangleCount = scene.getMesh(meshID).getTriangleCount();
            maxPrimitiveCount = std::max(maxPrimitiveCount, triangleCount);
        }
        for (uint32_t curveID = 0; curveID < scene.getCurveCount(); curveID++)
        {
            uint32_t curveSegmentCount = scene.getCurve(curveID).getSegmentCount();
            maxPrimitiveCount = std::max(maxPrimitiveCount, curveSegmentCount);
        }
        mPrimitiveIndexBits = allocateBits(maxPrimitiveCount);

        // Check that the final bit allocation fits.
        if (mPrimitiveIndexBits > 32 || (mTypeBits + mInstanceIndexBits) > 32)
        {
            throw std::exception("Scene requires > 96 bits for encoding hit information. This is currently not supported.");
        }

        mDataSize = 4;
        mPackedDataSize = (mTypeBits + mInstanceIndexBits + mPrimitiveIndexBits <= 32) ? 2 : 3;

        // Check if hit info needs displacement value.
        if (scene.hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            mDataSize += 1;
            mPackedDataSize += 1;
        }
    }

    Shader::DefineList HitInfo::getDefines() const
    {
        assert((mTypeBits + mInstanceIndexBits) <= 32 && mPrimitiveIndexBits <= 32);
        Shader::DefineList defines;
        defines.add("HIT_INFO_DEFINES", "1");
        defines.add("HIT_INFO_DATA_SIZE", std::to_string(mDataSize));
        defines.add("HIT_INFO_PACKED_DATA_SIZE", std::to_string(mPackedDataSize));
        defines.add("HIT_TYPE_BITS", std::to_string(mTypeBits));
        defines.add("HIT_INSTANCE_INDEX_BITS", std::to_string(mInstanceIndexBits));
        defines.add("HIT_PRIMITIVE_INDEX_BITS", std::to_string(mPrimitiveIndexBits));
        return defines;
    }

    ResourceFormat HitInfo::getFormat() const
    {
        assert(mPackedDataSize >= 2 && mPackedDataSize <= 4);
        switch (mPackedDataSize)
        {
        case 2:
            return ResourceFormat::RG32Uint;
        case 3:
            return ResourceFormat::RGBA32Uint; // RGB32Uint can't be used for UAV writes
        case 4:
            return ResourceFormat::RGBA32Uint;
        default:
            should_not_get_here();
            break;
        }
        return kDefaultFormat;
    }
}

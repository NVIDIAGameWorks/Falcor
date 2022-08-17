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
#include "HitInfo.h"
#include "HitInfoType.slang"
#include "Scene.h"
#include "Utils/Logger.h"

namespace Falcor
{
    namespace
    {
        const uint32_t kCompressedTypeBits = 2; ///< Number of bits used for type when using compression.

        // Make sure hit types used in compressed hit info fit into compressed type field.
        static_assert((uint32_t)HitType::None < (1 << kCompressedTypeBits));
        static_assert((uint32_t)HitType::Triangle < (1 << kCompressedTypeBits));
        static_assert((uint32_t)HitType::Volume < (1 << kCompressedTypeBits));

        uint32_t allocateBits(const uint32_t count)
        {
            if (count <= 1) return 0;
            uint32_t maxValue = count - 1;
            return bitScanReverse(maxValue) + 1;
        }
    }

    void HitInfo::init(const Scene& scene, bool useCompression)
    {
        // Setup bit allocations for encoding the hit information.
        // By default the shader code will use a 128-bit format.
        // If compression is requested and the hit info is small enough, a 64-bit format is used instead.

        uint32_t typeCount = (uint32_t)HitType::Count;
        mTypeBits = allocateBits(typeCount);

        mInstanceIDBits = allocateBits(scene.getGeometryInstanceCount());

        uint32_t maxPrimitiveCount = 0;

        for (MeshID meshID{ 0 }; meshID.get() < scene.getMeshCount(); ++meshID)
        {
            uint32_t triangleCount = scene.getMesh(meshID).getTriangleCount();
            maxPrimitiveCount = std::max(maxPrimitiveCount, triangleCount);
        }
        for (CurveID curveID{ 0 }; curveID.get() < scene.getCurveCount(); ++curveID)
        {
            uint32_t curveSegmentCount = scene.getCurve(curveID).getSegmentCount();
            maxPrimitiveCount = std::max(maxPrimitiveCount, curveSegmentCount);
        }

        mPrimitiveIndexBits = allocateBits(maxPrimitiveCount);
        for (SdfGridID sdfID{ 0 } ; sdfID.get() < scene.getSDFGridCount(); ++sdfID)
        {
            uint32_t sdfGridMaxPrimitiveIDBits = scene.getSDFGrid(sdfID)->getMaxPrimitiveIDBits();
            mPrimitiveIndexBits = std::max(mPrimitiveIndexBits, sdfGridMaxPrimitiveIDBits);
        }

        // Check that the final bit allocation fits.
        if (mPrimitiveIndexBits > 32 || (mTypeBits + mInstanceIDBits) > 32)
        {
            throw RuntimeError("Scene requires > 64 bits for encoding hit info header. This is currently not supported.");
        }

        // Compute size of compressed header in bits.
        const uint32_t compressedHeaderBits = kCompressedTypeBits + mInstanceIDBits + mPrimitiveIndexBits;

        // Check if compression is supported (small header and triangle meshes only).
        const bool compressionSupported = compressedHeaderBits <= 32 && scene.getGeometryTypes() == Scene::GeometryTypeFlags::TriangleMesh;

        // Use compression if supported and requested.
        mUseCompression = compressionSupported && useCompression;

        // Switch to using fewer bits for the type if compression is used.
        if (mUseCompression) mTypeBits = kCompressedTypeBits;

        logInfo(
            "HitInfo: Total size is {} bits (type: {} bits, instanceID: {} bits, primitiveIndex: {} bits)",
            mUseCompression ? 64 : 128, mTypeBits, mInstanceIDBits, mPrimitiveIndexBits
        );

    }

    Shader::DefineList HitInfo::getDefines() const
    {
        FALCOR_ASSERT((mTypeBits + mInstanceIDBits) <= 32 && mPrimitiveIndexBits <= 32);
        Shader::DefineList defines;
        defines.add("HIT_INFO_DEFINES", "1");
        defines.add("HIT_INFO_USE_COMPRESSION", mUseCompression ? "1" : "0");
        defines.add("HIT_INFO_TYPE_BITS", std::to_string(mTypeBits));
        defines.add("HIT_INFO_INSTANCE_ID_BITS", std::to_string(mInstanceIDBits));
        defines.add("HIT_INFO_PRIMITIVE_INDEX_BITS", std::to_string(mPrimitiveIndexBits));
        return defines;
    }

    ResourceFormat HitInfo::getFormat() const
    {
        return mUseCompression ? ResourceFormat::RG32Uint : ResourceFormat::RGBA32Uint;
    }
}

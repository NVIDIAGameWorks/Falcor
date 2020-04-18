/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************/
#pragma once
#include "Falcor.h"

namespace Falcor
{
    class HitInfo
    {
    public:
        static const uint32_t kInvalidIndex = 0xffffffff;

        /** Returns defines needed packing/unpacking a HitInfo struct.
        */
        static Shader::DefineList getDefines(const Scene* pScene)
        {
            // Setup bit allocations for encoding the meshInstanceID and primitive indices.

            uint32_t meshInstanceCount = pScene->getMeshInstanceCount();
            uint32_t maxInstanceID = meshInstanceCount > 0 ? meshInstanceCount - 1 : 0;
            uint32_t instanceIndexBits = maxInstanceID > 0 ? bitScanReverse(maxInstanceID) + 1 : 0;

            uint32_t maxTriangleCount = 0;
            for (uint32_t meshID = 0; meshID < pScene->getMeshCount(); meshID++)
            {
                uint32_t triangleCount = pScene->getMesh(meshID).indexCount / 3;
                maxTriangleCount = std::max(triangleCount, maxTriangleCount);
            }
            uint32_t maxTriangleID = maxTriangleCount > 0 ? maxTriangleCount - 1 : 0;
            uint32_t triangleIndexBits = maxTriangleID > 0 ? bitScanReverse(maxTriangleID) + 1 : 0;

            assert(instanceIndexBits > 0 && triangleIndexBits > 0);
            if (instanceIndexBits + triangleIndexBits > 32 ||
                (instanceIndexBits + triangleIndexBits == 32 && ((maxInstanceID << triangleIndexBits) | maxTriangleID) == kInvalidIndex))
            {
                logError("Scene requires > 32 bits for encoding meshInstanceID/triangleIndex. This is currently not supported.");
            }

            // Setup defines for the shader program.
            Shader::DefineList defines;
            defines.add("HIT_INSTANCE_INDEX_BITS", std::to_string(instanceIndexBits));
            defines.add("HIT_TRIANGLE_INDEX_BITS", std::to_string(triangleIndexBits));

            return defines;
        }
    };
}

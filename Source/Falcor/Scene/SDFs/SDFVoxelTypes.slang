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
#pragma once
#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "Utils/Math/AABB.h"
#else
import Utils.Math.AABB;
#endif

BEGIN_NAMESPACE_FALCOR

struct SDFSVSVoxel
{
    uint4 packedValuesSlices[4];    ///< 4x4 value slices in the x-direction of the voxel, packed as 8-bit snorms.
    uint validNeighborsMask;        ///< Low 27 bits used as validity mask for neighbors, wastes one bit of the mask, but fine for now.
};

struct SDFSVOHashTableVoxel
{
    uint2 locationCode;     ///< The voxel location code.
    uint svoOffset;         ///< The offset of this voxel in the SVO.
    uint validMask;         ///< The valid mask of the voxel, encoding what children are valid, padded so voxel is multiple of 4 bytes in size.
    uint2 packedValues;     ///< The eight corner values of the SDFVoxel, packed as 8-bit snorms.
};

struct SDFSVOVoxel
{
    uint relationData;      ///< Relation data for the voxel encoding the index its first child in the SVO and the validity of its children.
    uint2 locationCode;     ///< The location code of the voxel.
    uint2 packedValues;     ///< The eight corner values of the SDFVoxel, packed as 8-bit snorms.
};

END_NAMESPACE_FALCOR

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
import Scene.SDFs.SDFVoxelCommon;
import Scene.SDFs.SDFVoxelTypes;
import Scene.SDFs.SparseVoxelOctree.SDFSVOHashTable;

cbuffer CB
{
    uint gLocationCodeStartOffset;
    uint gVoxelCount;
};

ParameterBlock<SDFSVOHashTable> gVoxelHashTable;
RWByteAddressBuffer gLocationCodes;

RWStructuredBuffer<SDFSVOVoxel> gSVO;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x > gVoxelCount) return;

    // Load the location code for the current voxel.
    const uint2 locationCode = gLocationCodes.Load2(8 * (gLocationCodeStartOffset + dispatchThreadID.x));

    uint svoOffset;
    uint validMask;
    uint2 packedValues;
    // Load the current voxel values, there must be one voxel per thread so we don't need to check whether if SDFSVOHashTable::lookup() returns true or false.
    if (gVoxelHashTable.lookup(locationCode, svoOffset, validMask, packedValues))
    {
        SDFSVOVoxel currentVoxel;
        currentVoxel.locationCode = locationCode;
        currentVoxel.relationData = validMask;
        currentVoxel.packedValues = packedValues;

        // Find the ID of the first valid child.
        for (uint firstValidChildID = 0; firstValidChildID < 8; firstValidChildID++)
        {
            if ((currentVoxel.relationData & (1 << firstValidChildID)) > 0)
            {
                // Create the location code for the child.
                uint2 childLocationCode;
                SDFVoxelCommon::createChildLocationCode(currentVoxel.locationCode, firstValidChildID, childLocationCode);
                childLocationCode.y |= (1 << 31);

                uint childSVOOffset;
                uint childValidMask;
                uint2 childPackedValues;

                // The valid mask guarantees that the child voxel exist.
                gVoxelHashTable.lookup(childLocationCode, childSVOOffset, childValidMask, childPackedValues);

                // Encode the relative child offset into the top 24 bits of relation data.
                currentVoxel.relationData |= (childSVOOffset << 8);
                break;
            }
        }

        gSVO[svoOffset] = currentVoxel;
    }
}

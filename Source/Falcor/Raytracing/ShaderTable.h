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

namespace Falcor
{
    class Scene;
    class Program;
    class RtStateObject;

    /* We are using the following layout for the shader-table:

       +------------+---------+---------+-----+--------+---------+--------+-----+--------+--------+-----+--------+-----+--------+--------+-----+--------+
       |            |         |         | ... |        |         |        | ... |        |        | ... |        | ... |        |        | ... |        |
       |   RayGen   |   Ray0  |   Ray1  | ... |  RayN  |   Ray0  |  Ray1  | ... |  RayN  |  Ray0  | ... |  RayN  | ... |  Ray0  |  Ray0  | ... |  RayN  |
       |   Entry    |   Miss  |   Miss  | ... |  Miss  |   Hit   |   Hit  | ... |  Hit   |  Hit   | ... |  Hit   | ... |  Hit   |  Hit   | ... |  Hit   |
       |            |         |         | ... |        |  Mesh0  |  Mesh0 | ... |  Mesh0 |  Mesh1 | ... |  Mesh1 | ... | MeshN  |  MeshN | ... |  MeshN |
       +------------+---------+---------+-----+--------+---------+--------+-----+--------+--------+-----+--------+-----+--------+--------+-----+--------+

       The first record is the ray gen, followed by the miss records, followed by the meshes records.
       For each mesh we have N hit records, N == number of ray types in the program
       The size of each record is varying based on the type. RayGen and miss entries contain only the program identifier. Hit entries contain the program identifier and the geometry index as a shader constant

       User provided local root signatures are not supported for performance reasons. Managing and updating data for custom root-signatures results in significant overhead.
       To get the root-signature that matches this table, call the static function getRootSignatre()
    */

    class dlldecl ShaderTable
    {
    public:
        using SharedPtr = std::shared_ptr<ShaderTable>;

        /** Create a new object
        */
        static SharedPtr create();

        /** Update the shader table.
            This function doesn't do any early out. If it's called, it will always update the table.
            Call it only when the RtStateObject changed or when the program was recompiled
        */
        void update(
            RenderContext*          pCtx,
            RtStateObject*          pRtso,
            RtProgramVars const*    pVars);

        void flushBuffer(
            RenderContext*          pCtx);

        struct SubTableInfo
        {
            uint32_t recordSize = align_to(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            uint32_t recordCount = 0;
            uint32_t offset = 0;
        };

        enum class SubTableType
        {
            RayGen,
            Miss,
            Hit,
            Count,
        };

        SubTableInfo getSubTableInfo(SubTableType type) const { return mSubTables[int32_t(type)]; }
        uint32_t getRecordSize(SubTableType type) const { return getSubTableInfo(type).recordSize; }
        uint32_t getRecordCount(SubTableType type) const { return getSubTableInfo(type).recordCount; }
        uint32_t getOffset(SubTableType type) const { return getSubTableInfo(type).offset; }

        uint8_t* getRecordPtr(SubTableType type, uint32_t index);

        /** Get the buffer
        */
        const Buffer::SharedPtr& getBuffer() const { return mpBuffer; }

        /** Get the size of the RayGen record
        */
        uint32_t getRayGenRecordSize() const { return getRecordSize(SubTableType::RayGen); }

        /** Get the offset of the RayGen table
        */
        uint32_t getRayGenTableOffset() const { return getOffset(SubTableType::RayGen); }

        /** Get the size of the miss record
        */
        uint32_t getMissRecordSize() const { return getRecordSize(SubTableType::Miss); }

        /** Get the number of miss records
        */
        uint32_t getMissRecordCount() const { return getRecordCount(SubTableType::Miss); }

        /** Get the offset to the miss table
        */
        uint32_t getMissTableOffset() const { return getOffset(SubTableType::Miss); }

        /** Get the size of the hit record
        */
        uint32_t getHitRecordSize() const { return getRecordSize(SubTableType::Hit); }

        /** Get the number of hit entries
        */
        uint32_t getHitRecordCount() const { return getRecordCount(SubTableType::Hit); }

        /** Get the offset to the first hit table
        */
        uint32_t getHitTableOffset() const { return getOffset(SubTableType::Hit); }

        RtStateObject* getRtso() const { return mpRtso; }

    private:
        ShaderTable() = default;

        SubTableInfo mSubTables[int(SubTableType::Count)];

        RtStateObject*          mpRtso = nullptr;
        Buffer::SharedPtr       mpBuffer;
        std::vector<uint8_t>    mData;
    };
}

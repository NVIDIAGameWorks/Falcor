/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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

#include "Scene/SDFs/SDFGrid.h"
#include "Core/API/Buffer.h"
#include "Core/API/Texture.h"
#include "Core/Pass/ComputePass.h"

namespace Falcor
{
    /** A single SDF Sparse Voxel Set. Can only be utilized on the GPU.
    */
    class FALCOR_API SDFSVS : public SDFGrid
    {
    public:
        static ref<SDFSVS> create(ref<Device> pDevice) { return make_ref<SDFSVS>(pDevice); }

        /// Create am empty SDF sparse voxel set.
        SDFSVS(ref<Device> pDevice) : SDFGrid(pDevice) {}

        virtual size_t getSize() const override;
        virtual uint32_t getMaxPrimitiveIDBits() const override;
        virtual Type getType() const override { return Type::SparseVoxelSet; }

        virtual void createResources(RenderContext* pRenderContext, bool deleteScratchData = true) override;

        virtual const ref<Buffer>& getAABBBuffer() const override { return mpVoxelAABBBuffer; }
        virtual uint32_t getAABBCount() const override { return mVoxelCount; }

        virtual void bindShaderData(const ShaderVar& var) const override;

    protected:
        virtual void setValuesInternal(const std::vector<float>& cornerValues) override;

    private:
        // CPU data.
        std::vector<int8_t> mValues;

        // Specs.
        ref<Buffer> mpVoxelAABBBuffer;
        ref<Buffer> mpVoxelBuffer;
        uint32_t mVoxelCount = 0;

        // Compute passes used to build the SVS.
        ref<ComputePass> mpCountSurfaceVoxelsPass;
        ref<ComputePass> mpSDFSVSVoxelizerPass;

        // Scratch data used for building.
        ref<Buffer> mpSurfaceVoxelCounter;
        ref<Texture> mpSDFGridTexture;
    };
}

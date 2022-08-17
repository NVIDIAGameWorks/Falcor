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
#pragma once

#include "Scene/SDFs/SDFGrid.h"
#include "Core/API/Texture.h"

namespace Falcor
{
    /** A normalized dense SDF grid, represented as a set of textures. Can only be accessed on the GPU.
    */
    class FALCOR_API NDSDFGrid : public SDFGrid
    {
    public:
        using SharedPtr = std::shared_ptr<NDSDFGrid>;

        /** Create a new, empty normalized dense SDF grid.
            \param[in] narrowBandThickness NDSDFGrids operate on normalized distances, the distances are normalized so that a normalized distance of +- 1 represents a distance of "narrowBandThickness" voxel diameters. Should not be less than 1.
            \return NDSDFGrid object, or nullptr if errors occurred.
        */
        static SharedPtr create(float narrowBandThickness);

        virtual size_t getSize() const override;
        virtual uint32_t getMaxPrimitiveIDBits() const override;
        virtual Type getType() const override { return Type::NormalizedDenseGrid; }


        virtual void createResources(RenderContext* pRenderContext, bool deleteScratchData = true) override;
        virtual const Buffer::SharedPtr& getAABBBuffer() const override { return spNDSDFGridUnitAABBBuffer; }
        virtual uint32_t getAABBCount() const override { return 1; }
        virtual void setShaderData(const ShaderVar& var) const override;

    protected:
        virtual void setValuesInternal(const std::vector<float>& cornerValues) override;

        float calculateNormalizationFactor(uint32_t gridWidth) const;

    private:
        NDSDFGrid(float narrowBandThickness);

        // CPU data.
        std::vector<std::vector<int8_t>> mValues;

        // Specs.
        uint32_t mCoarsestLODGridWidth = 0;
        float mCoarsestLODNormalizationFactor = 0.0f;
        float mNarrowBandThickness = 0.0f;

        // Resources shared among all NDSDFGrids.
        static Sampler::SharedPtr spNDSDFGridSampler;
        static Buffer::SharedPtr spNDSDFGridUnitAABBBuffer;

        // GPU data.
        std::vector<Texture::SharedPtr> mNDSDFTextures;
    };
}

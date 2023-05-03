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
#include "fwd.h"
#include "Common.h"
#include "Handles.h"
#include "NativeHandle.h"
#include "Core/Macros.h"
#include "Utils/Math/Vector.h"
#include <memory>

namespace Falcor
{
/**
 * Abstract the API sampler state object
 */
class FALCOR_API Sampler
{
public:
    using SharedPtr = std::shared_ptr<Sampler>;

    /**
     * Filter mode
     */
    enum class Filter
    {
        Point,
        Linear,
    };

    /**
     * Addressing mode in case the texture coordinates are out of [0, 1] range
     */
    enum class AddressMode
    {
        Wrap,      ///< Wrap around
        Mirror,    ///< Wrap around and mirror on every integer junction
        Clamp,     ///< Clamp the normalized coordinates to [0, 1]
        Border,    ///< If out-of-bound, use the sampler's border color
        MirrorOnce ///< Same as Mirror, but mirrors only once around 0
    };

    /**
     * Reduction mode
     */
    enum class ReductionMode
    {
        Standard,
        Comparison,
        Min,
        Max,
    };

    /**
     * Comparison mode for the sampler.
     */
    using ComparisonMode = ComparisonFunc;

    /**
     * Descriptor used to create a new Sampler object
     */
    struct Desc
    {
        Filter magFilter = Filter::Linear;
        Filter minFilter = Filter::Linear;
        Filter mipFilter = Filter::Linear;
        uint32_t maxAnisotropy = 1;
        float maxLod = 1000;
        float minLod = -1000;
        float lodBias = 0;
        ComparisonMode comparisonMode = ComparisonMode::Disabled;
        ReductionMode reductionMode = ReductionMode::Standard;
        AddressMode addressModeU = AddressMode::Wrap;
        AddressMode addressModeV = AddressMode::Wrap;
        AddressMode addressModeW = AddressMode::Wrap;
        float4 borderColor = float4(0, 0, 0, 0);

        /**
         * Set the filter mode
         * @param[in] minFilter Filter mode in case of minification.
         * @param[in] magFilter Filter mode in case of magnification.
         * @param[in] mipFilter Mip-level sampling mode
         */
        Desc& setFilterMode(Filter minFilter, Filter magFilter, Filter mipFilter)
        {
            this->magFilter = magFilter;
            this->minFilter = minFilter;
            this->mipFilter = mipFilter;
            return *this;
        }

        /**
         * Set the maximum anisotropic filtering value. If MaxAnisotropy > 1, min/mag/mip filter modes are ignored
         */
        Desc& setMaxAnisotropy(uint32_t maxAnisotropy)
        {
            this->maxAnisotropy = maxAnisotropy;
            return *this;
        }

        /**
         * Set the lod clamp parameters
         * @param[in] minLod Minimum LOD that will be used when sampling
         * @param[in] maxLod Maximum LOD that will be used when sampling
         * @param[in] lodBias Bias to apply to the LOD
         */
        Desc& setLodParams(float minLod, float maxLod, float lodBias)
        {
            this->minLod = minLod;
            this->maxLod = maxLod;
            this->lodBias = lodBias;
            return *this;
        }

        /**
         * Set the sampler comparison mode.
         */
        Desc& setComparisonMode(ComparisonMode mode)
        {
            this->comparisonMode = mode;
            return *this;
        }

        /**
         * Set the sampler reduction mode.
         */
        Desc& setReductionMode(ReductionMode mode)
        {
            this->reductionMode = mode;
            return *this;
        }

        /**
         * Set the sampler addressing mode
         * @param[in] modeU Addressing mode for U texcoord channel
         * @param[in] modeV Addressing mode for V texcoord channel
         * @param[in] modeW Addressing mode for W texcoord channel
         */
        Desc& setAddressingMode(AddressMode modeU, AddressMode modeV, AddressMode modeW)
        {
            this->addressModeU = modeU;
            this->addressModeV = modeV;
            this->addressModeW = modeW;
            return *this;
        }

        /**
         * Set the border color. Only applies when the addressing mode is ClampToBorder
         */
        Desc& setBorderColor(const float4& borderColor)
        {
            this->borderColor = borderColor;
            return *this;
        }

        /**
         * Returns true if sampler descs are identical.
         */
        bool operator==(const Desc& other) const
        {
            return magFilter == other.magFilter && minFilter == other.minFilter && mipFilter == other.mipFilter &&
                   maxAnisotropy == other.maxAnisotropy && maxLod == other.maxLod && minLod == other.minLod && lodBias == other.lodBias &&
                   comparisonMode == other.comparisonMode && reductionMode == other.reductionMode && addressModeU == other.addressModeU &&
                   addressModeV == other.addressModeV && addressModeW == other.addressModeW && borderColor == other.borderColor;
        }

        /**
         * Returns true if sampler descs are not identical.
         */
        bool operator!=(const Desc& other) const { return !(*this == other); }
    };

    Sampler(std::shared_ptr<Device> pDevice, const Desc& desc);
    ~Sampler();

    /**
     * Create a new sampler object.
     * @param[in] desc Describes sampler settings.
     * @return A new object, or throws an exception if creation failed.
     */
    static SharedPtr create(Device* pDevice, const Desc& desc);

    /**
     * Get the sampler state.
     */
    gfx::ISamplerState* getGfxSamplerState() const { return mGfxSamplerState; }

    /**
     * Returns the native API handle:
     * - D3D12: D3D12_CPU_DESCRIPTOR_HANDLE
     * - Vulkan: VkSampler
     */
    NativeHandle getNativeHandle() const;

    /**
     * Get the magnification filter
     */
    Filter getMagFilter() const { return mDesc.magFilter; }

    /**
     * Get the minification filter
     */
    Filter getMinFilter() const { return mDesc.minFilter; }

    /**
     * Get the mip-levels filter
     */
    Filter getMipFilter() const { return mDesc.mipFilter; }

    /**
     * Get the maximum anisotropy
     */
    uint32_t getMaxAnisotropy() const { return mDesc.maxAnisotropy; }

    /**
     * Get the minimum LOD value
     */
    float getMinLod() const { return mDesc.minLod; }

    /**
     * Get the maximum LOD value
     */
    float getMaxLod() const { return mDesc.maxLod; }

    /**
     * Get the LOD bias
     */
    float getLodBias() const { return mDesc.lodBias; }

    /**
     * Get the comparison mode
     */
    ComparisonMode getComparisonMode() const { return mDesc.comparisonMode; }

    /**
     * Get the reduction mode
     */
    ReductionMode getReductionMode() const { return mDesc.reductionMode; }

    /**
     * Get the addressing mode for the U texcoord
     */
    AddressMode getAddressModeU() const { return mDesc.addressModeU; }

    /**
     * Get the addressing mode for the V texcoord
     */
    AddressMode getAddressModeV() const { return mDesc.addressModeV; }

    /**
     * Get the addressing mode for the W texcoord
     */
    AddressMode getAddressModeW() const { return mDesc.addressModeW; }

    /**
     * Get the border color
     */
    const float4& getBorderColor() const { return mDesc.borderColor; }

    /**
     * Get the descriptor that was used to create the sampler.
     */
    const Desc& getDesc() const { return mDesc; }

private:
    std::shared_ptr<Device> mpDevice;
    Desc mDesc;
    Slang::ComPtr<gfx::ISamplerState> mGfxSamplerState;
    static uint32_t getApiMaxAnisotropy();
};
} // namespace Falcor

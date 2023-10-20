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
#include "Types.h"
#include "Handles.h"
#include "NativeHandle.h"
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Core/Enum.h"
#include "Utils/Math/Vector.h"

namespace Falcor
{

/**
 * Texture filtering modes.
 */
enum class TextureFilteringMode
{
    Point,
    Linear,
};

FALCOR_ENUM_INFO(
    TextureFilteringMode,
    {
        {TextureFilteringMode::Point, "Point"},
        {TextureFilteringMode::Linear, "Linear"},
    }
);
FALCOR_ENUM_REGISTER(TextureFilteringMode);

/**
 * Addressing mode in case the texture coordinates are out of [0, 1] range.
 */
enum class TextureAddressingMode
{
    Wrap,      ///< Wrap around
    Mirror,    ///< Wrap around and mirror on every integer junction
    Clamp,     ///< Clamp the normalized coordinates to [0, 1]
    Border,    ///< If out-of-bound, use the sampler's border color
    MirrorOnce ///< Same as Mirror, but mirrors only once around 0
};

FALCOR_ENUM_INFO(
    TextureAddressingMode,
    {
        {TextureAddressingMode::Wrap, "Wrap"},
        {TextureAddressingMode::Mirror, "Mirror"},
        {TextureAddressingMode::Clamp, "Clamp"},
        {TextureAddressingMode::Border, "Border"},
        {TextureAddressingMode::MirrorOnce, "MirrorOnce"},
    }
);
FALCOR_ENUM_REGISTER(TextureAddressingMode);

/**
 * Reduction modes.
 */
enum class TextureReductionMode
{
    Standard,
    Comparison,
    Min,
    Max,
};

FALCOR_ENUM_INFO(
    TextureReductionMode,
    {
        {TextureReductionMode::Standard, "Standard"},
        {TextureReductionMode::Comparison, "Comparison"},
        {TextureReductionMode::Min, "Min"},
        {TextureReductionMode::Max, "Max"},
    }
);
FALCOR_ENUM_REGISTER(TextureReductionMode);

/**
 * Abstract the API sampler state object
 */
class FALCOR_API Sampler : public Object
{
    FALCOR_OBJECT(Sampler)
public:
    /**
     * Descriptor used to create a new Sampler object
     */
    struct Desc
    {
        TextureFilteringMode magFilter = TextureFilteringMode::Linear;
        TextureFilteringMode minFilter = TextureFilteringMode::Linear;
        TextureFilteringMode mipFilter = TextureFilteringMode::Linear;
        uint32_t maxAnisotropy = 1;
        float maxLod = 1000;
        float minLod = -1000;
        float lodBias = 0;
        ComparisonFunc comparisonFunc = ComparisonFunc::Disabled;
        TextureReductionMode reductionMode = TextureReductionMode::Standard;
        TextureAddressingMode addressModeU = TextureAddressingMode::Wrap;
        TextureAddressingMode addressModeV = TextureAddressingMode::Wrap;
        TextureAddressingMode addressModeW = TextureAddressingMode::Wrap;
        float4 borderColor = float4(0, 0, 0, 0);

        /**
         * Set the filter mode
         * @param[in] minFilter_ Filter mode in case of minification.
         * @param[in] magFilter_ Filter mode in case of magnification.
         * @param[in] mipFilter_ Mip-level sampling mode
         */
        Desc& setFilterMode(TextureFilteringMode minFilter_, TextureFilteringMode magFilter_, TextureFilteringMode mipFilter_)
        {
            magFilter = magFilter_;
            minFilter = minFilter_;
            mipFilter = mipFilter_;
            return *this;
        }

        /**
         * Set the maximum anisotropic filtering value. If MaxAnisotropy > 1, min/mag/mip filter modes are ignored
         */
        Desc& setMaxAnisotropy(uint32_t maxAnisotropy_)
        {
            maxAnisotropy = maxAnisotropy_;
            return *this;
        }

        /**
         * Set the lod clamp parameters
         * @param[in] minLod Minimum LOD that will be used when sampling
         * @param[in] maxLod Maximum LOD that will be used when sampling
         * @param[in] lodBias Bias to apply to the LOD
         */
        Desc& setLodParams(float minLod_, float maxLod_, float lodBias_)
        {
            minLod = minLod_;
            maxLod = maxLod_;
            lodBias = lodBias_;
            return *this;
        }

        /**
         * Set the sampler comparison function.
         */
        Desc& setComparisonFunc(ComparisonFunc func)
        {
            comparisonFunc = func;
            return *this;
        }

        /**
         * Set the sampler reduction mode.
         */
        Desc& setReductionMode(TextureReductionMode mode)
        {
            reductionMode = mode;
            return *this;
        }

        /**
         * Set the sampler addressing mode
         * @param[in] modeU Addressing mode for U texcoord channel
         * @param[in] modeV Addressing mode for V texcoord channel
         * @param[in] modeW Addressing mode for W texcoord channel
         */
        Desc& setAddressingMode(TextureAddressingMode modeU, TextureAddressingMode modeV, TextureAddressingMode modeW)
        {
            addressModeU = modeU;
            addressModeV = modeV;
            addressModeW = modeW;
            return *this;
        }

        /**
         * Set the border color. Only applies when the addressing mode is ClampToBorder
         */
        Desc& setBorderColor(const float4& borderColor_)
        {
            borderColor = borderColor_;
            return *this;
        }

        /**
         * Returns true if sampler descs are identical.
         */
        bool operator==(const Desc& other) const
        {
            return magFilter == other.magFilter && minFilter == other.minFilter && mipFilter == other.mipFilter &&
                   maxAnisotropy == other.maxAnisotropy && maxLod == other.maxLod && minLod == other.minLod && lodBias == other.lodBias &&
                   comparisonFunc == other.comparisonFunc && reductionMode == other.reductionMode && addressModeU == other.addressModeU &&
                   addressModeV == other.addressModeV && addressModeW == other.addressModeW && all(borderColor == other.borderColor);
        }

        /**
         * Returns true if sampler descs are not identical.
         */
        bool operator!=(const Desc& other) const { return !(*this == other); }
    };

    Sampler(ref<Device> pDevice, const Desc& desc);
    ~Sampler();

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
    TextureFilteringMode getMagFilter() const { return mDesc.magFilter; }

    /**
     * Get the minification filter
     */
    TextureFilteringMode getMinFilter() const { return mDesc.minFilter; }

    /**
     * Get the mip-levels filter
     */
    TextureFilteringMode getMipFilter() const { return mDesc.mipFilter; }

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
     * Get the comparison function
     */
    ComparisonFunc getComparisonFunc() const { return mDesc.comparisonFunc; }

    /**
     * Get the reduction mode
     */
    TextureReductionMode getReductionMode() const { return mDesc.reductionMode; }

    /**
     * Get the addressing mode for the U texcoord
     */
    TextureAddressingMode getAddressModeU() const { return mDesc.addressModeU; }

    /**
     * Get the addressing mode for the V texcoord
     */
    TextureAddressingMode getAddressModeV() const { return mDesc.addressModeV; }

    /**
     * Get the addressing mode for the W texcoord
     */
    TextureAddressingMode getAddressModeW() const { return mDesc.addressModeW; }

    /**
     * Get the border color
     */
    const float4& getBorderColor() const { return mDesc.borderColor; }

    /**
     * Get the descriptor that was used to create the sampler.
     */
    const Desc& getDesc() const { return mDesc; }

    void breakStrongReferenceToDevice();

private:
    BreakableReference<Device> mpDevice;
    Desc mDesc;
    Slang::ComPtr<gfx::ISamplerState> mGfxSamplerState;
    static uint32_t getApiMaxAnisotropy();

    friend class Device;
};

} // namespace Falcor

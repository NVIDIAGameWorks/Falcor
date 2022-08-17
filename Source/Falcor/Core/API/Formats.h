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
#include "DXGIFormats.h"
#include "Core/Assert.h"
#include "Core/Macros.h"
#include <string>
#include <cstdint>

namespace Falcor
{
    /*!
    *  \addtogroup Falcor
    *  @{
    */

    /** Flags for enumerating texture color channels.
    */
    enum class TextureChannelFlags : uint32_t
    {
        None = 0x0,
        Red = 0x1,
        Green = 0x2,
        Blue = 0x4,
        Alpha = 0x8,
        RGB = 0x7,
        RGBA = 0xf,
    };

    FALCOR_ENUM_CLASS_OPERATORS(TextureChannelFlags);

    /** These flags are hints the driver to what pipeline stages the resource will be bound to.
    */
    enum class ResourceBindFlags : uint32_t
    {
        None = 0x0,             ///< The resource will not be bound the pipeline. Use this to create a staging resource
        Vertex = 0x1,           ///< The resource will be bound as a vertex-buffer
        Index = 0x2,            ///< The resource will be bound as a index-buffer
        Constant = 0x4,         ///< The resource will be bound as a constant-buffer
        StreamOutput = 0x8,     ///< The resource will be bound to the stream-output stage as an output buffer
        ShaderResource = 0x10,  ///< The resource will be bound as a shader-resource
        UnorderedAccess = 0x20, ///< The resource will be bound as an UAV
        RenderTarget = 0x40,    ///< The resource will be bound as a render-target
        DepthStencil = 0x80,    ///< The resource will be bound as a depth-stencil buffer
        IndirectArg = 0x100,    ///< The resource will be bound as an indirect argument buffer
        Shared      = 0x200,    ///< The resource will be shared with a different adapter. Mostly useful for sharing resoures with CUDA
        AccelerationStructure = 0x80000000,  ///< The resource will be bound as an acceleration structure

        AllColorViews = ShaderResource | UnorderedAccess | RenderTarget,
        AllDepthViews = ShaderResource | DepthStencil
    };

    FALCOR_ENUM_CLASS_OPERATORS(ResourceBindFlags);

    /** Resource formats
    */
    enum class ResourceFormat : uint32_t
    {
        Unknown,
        R8Unorm,
        R8Snorm,
        R16Unorm,
        R16Snorm,
        RG8Unorm,
        RG8Snorm,
        RG16Unorm,
        RG16Snorm,
        RGB16Unorm,
        RGB16Snorm,
        R24UnormX8,
        RGB5A1Unorm,
        RGBA8Unorm,
        RGBA8Snorm,
        RGB10A2Unorm,
        RGB10A2Uint,
        RGBA16Unorm,
        RGBA8UnormSrgb,
        R16Float,
        RG16Float,
        RGB16Float,
        RGBA16Float,
        R32Float,
        R32FloatX32,
        RG32Float,
        RGB32Float,
        RGBA32Float,
        R11G11B10Float,
        RGB9E5Float,
        R8Int,
        R8Uint,
        R16Int,
        R16Uint,
        R32Int,
        R32Uint,
        RG8Int,
        RG8Uint,
        RG16Int,
        RG16Uint,
        RG32Int,
        RG32Uint,
        RGB16Int,
        RGB16Uint,
        RGB32Int,
        RGB32Uint,
        RGBA8Int,
        RGBA8Uint,
        RGBA16Int,
        RGBA16Uint,
        RGBA32Int,
        RGBA32Uint,

        BGRA8Unorm,
        BGRA8UnormSrgb,

        BGRX8Unorm,
        BGRX8UnormSrgb,
        Alpha8Unorm,
        Alpha32Float,
        R5G6B5Unorm,

        // Depth-stencil
        D32Float,
        D16Unorm,
        D32FloatS8X24,
        D24UnormS8,

        // Compressed formats
        BC1Unorm,   // DXT1
        BC1UnormSrgb,
        BC2Unorm,   // DXT3
        BC2UnormSrgb,
        BC3Unorm,   // DXT5
        BC3UnormSrgb,
        BC4Unorm,   // RGTC Unsigned Red
        BC4Snorm,   // RGTC Signed Red
        BC5Unorm,   // RGTC Unsigned RG
        BC5Snorm,   // RGTC Signed RG
        BC6HS16,
        BC6HU16,
        BC7Unorm,
        BC7UnormSrgb,

        Count
    };

    /** Falcor format Type
    */
    enum class FormatType
    {
        Unknown,        ///< Unknown format Type
        Float,          ///< Floating-point formats
        Unorm,          ///< Unsigned normalized formats
        UnormSrgb,      ///< Unsigned normalized SRGB formats
        Snorm,          ///< Signed normalized formats
        Uint,           ///< Unsigned integer formats
        Sint            ///< Signed integer formats
    };

    struct FormatDesc
    {
        ResourceFormat format;
        const std::string name;
        uint32_t bytesPerBlock;
        uint32_t channelCount;
        FormatType Type;
        struct
        {
            bool isDepth;
            bool isStencil;
            bool isCompressed;
        };
        struct
        {
            uint32_t width;
            uint32_t height;
        } compressionRatio;
        int numChannelBits[4];
    };

    extern const FALCOR_API FormatDesc kFormatDesc[];

    /** Get the number of bytes per format
    */
    inline uint32_t getFormatBytesPerBlock(ResourceFormat format)
    {
        FALCOR_ASSERT(kFormatDesc[(uint32_t)format].format == format);
        return kFormatDesc[(uint32_t)format].bytesPerBlock;
    }

    inline uint32_t getFormatPixelsPerBlock(ResourceFormat format)
    {
        FALCOR_ASSERT(kFormatDesc[(uint32_t)format].format == format);
        return kFormatDesc[(uint32_t)format].compressionRatio.width * kFormatDesc[(uint32_t)format].compressionRatio.height;
    }

    /** Check if the format has a depth component
    */
    inline bool isDepthFormat(ResourceFormat format)
    {
        FALCOR_ASSERT(kFormatDesc[(uint32_t)format].format == format);
        return kFormatDesc[(uint32_t)format].isDepth;
    }

    /** Check if the format has a stencil component
    */
    inline bool isStencilFormat(ResourceFormat format)
    {
        FALCOR_ASSERT(kFormatDesc[(uint32_t)format].format == format);
        return kFormatDesc[(uint32_t)format].isStencil;
    }

    /** Check if the format has depth or stencil components
    */
    inline bool isDepthStencilFormat(ResourceFormat format)
    {
        return isDepthFormat(format) || isStencilFormat(format);
    }

    /** Check if the format is a compressed format
    */
    inline bool isCompressedFormat(ResourceFormat format)
    {
        FALCOR_ASSERT(kFormatDesc[(uint32_t)format].format == format);
        return kFormatDesc[(uint32_t)format].isCompressed;
    }

    /** Get the format compression ration along the x-axis
    */
    inline uint32_t getFormatWidthCompressionRatio(ResourceFormat format)
    {
        FALCOR_ASSERT(kFormatDesc[(uint32_t)format].format == format);
        return kFormatDesc[(uint32_t)format].compressionRatio.width;
    }

    /** Get the format compression ration along the y-axis
    */
    inline uint32_t getFormatHeightCompressionRatio(ResourceFormat format)
    {
        FALCOR_ASSERT(kFormatDesc[(uint32_t)format].format == format);
        return kFormatDesc[(uint32_t)format].compressionRatio.height;
    }

    /** Get the number of channels
    */
    inline uint32_t getFormatChannelCount(ResourceFormat format)
    {
        FALCOR_ASSERT(kFormatDesc[(uint32_t)format].format == format);
        return kFormatDesc[(uint32_t)format].channelCount;
    }

    /** Get the format Type
    */
    inline FormatType getFormatType(ResourceFormat format)
    {
        FALCOR_ASSERT(kFormatDesc[(uint32_t)format].format == format);
        return kFormatDesc[(uint32_t)format].Type;
    }

    /** Check if a format is an integer type.
    */
    inline bool isIntegerFormat(ResourceFormat format)
    {
        FormatType type = getFormatType(format);
        return type == FormatType::Uint || type == FormatType::Sint;
    }

    /** Get number of bits used for a given color channel.
    */
    inline uint32_t getNumChannelBits(ResourceFormat format, int channel)
    {
        return kFormatDesc[(uint32_t)format].numChannelBits[channel];
    }

    /** Get number of bits used for the given color channels.
    */
    inline uint32_t getNumChannelBits(ResourceFormat format, TextureChannelFlags mask)
    {
        uint32_t bits = 0;
        if (is_set(mask, TextureChannelFlags::Red)) bits += getNumChannelBits(format, 0);
        if (is_set(mask, TextureChannelFlags::Green)) bits += getNumChannelBits(format, 1);
        if (is_set(mask, TextureChannelFlags::Blue)) bits += getNumChannelBits(format, 2);
        if (is_set(mask, TextureChannelFlags::Alpha)) bits += getNumChannelBits(format, 3);
        return bits;
    }

    /** Get mask of enabled color channels. See TextureChannelFlags.
    */
    inline TextureChannelFlags getChannelMask(ResourceFormat format)
    {
        TextureChannelFlags mask = TextureChannelFlags::None;
        if (kFormatDesc[(uint32_t)format].numChannelBits[0] > 0) mask |= TextureChannelFlags::Red;
        if (kFormatDesc[(uint32_t)format].numChannelBits[1] > 0) mask |= TextureChannelFlags::Green;
        if (kFormatDesc[(uint32_t)format].numChannelBits[2] > 0) mask |= TextureChannelFlags::Blue;
        if (kFormatDesc[(uint32_t)format].numChannelBits[3] > 0) mask |= TextureChannelFlags::Alpha;
        return mask;
    }

    /** Get the number of bytes per row. If format is compressed, width should be evenly divisible by the compression ratio.
    */
    inline uint32_t getFormatRowPitch(ResourceFormat format, uint32_t width)
    {
        FALCOR_ASSERT(width % getFormatWidthCompressionRatio(format) == 0);
        return (width / getFormatWidthCompressionRatio(format)) * getFormatBytesPerBlock(format);
    }

    /** Check if a format represents sRGB color space
    */
    inline bool isSrgbFormat(ResourceFormat format)
    {
        return (getFormatType(format) == FormatType::UnormSrgb);
    }

    /** Convert an SRGB format to linear. If the format is already linear, will return it
    */
    inline ResourceFormat srgbToLinearFormat(ResourceFormat format)
    {
        switch (format)
        {
        case ResourceFormat::BC1UnormSrgb:
            return ResourceFormat::BC1Unorm;
        case ResourceFormat::BC2UnormSrgb:
            return ResourceFormat::BC2Unorm;
        case ResourceFormat::BC3UnormSrgb:
            return ResourceFormat::BC3Unorm;
        case ResourceFormat::BGRA8UnormSrgb:
            return ResourceFormat::BGRA8Unorm;
        case ResourceFormat::BGRX8UnormSrgb:
            return ResourceFormat::BGRX8Unorm;
        case ResourceFormat::RGBA8UnormSrgb:
            return ResourceFormat::RGBA8Unorm;
        case ResourceFormat::BC7UnormSrgb:
            return ResourceFormat::BC7Unorm;
        default:
            FALCOR_ASSERT(isSrgbFormat(format) == false);
            return format;
        }
    }

    /** Convert an linear format to sRGB. If the format doesn't have a matching sRGB format, will return the original
    */
    inline ResourceFormat linearToSrgbFormat(ResourceFormat format)
    {
        switch (format)
        {
        case ResourceFormat::BC1Unorm:
            return ResourceFormat::BC1UnormSrgb;
        case ResourceFormat::BC2Unorm:
            return ResourceFormat::BC2UnormSrgb;
        case ResourceFormat::BC3Unorm:
            return ResourceFormat::BC3UnormSrgb;
        case ResourceFormat::BGRA8Unorm:
            return ResourceFormat::BGRA8UnormSrgb;
        case ResourceFormat::BGRX8Unorm:
            return ResourceFormat::BGRX8UnormSrgb;
        case ResourceFormat::RGBA8Unorm:
            return ResourceFormat::RGBA8UnormSrgb;
        case ResourceFormat::BC7Unorm:
            return ResourceFormat::BC7UnormSrgb;
        default:
            return format;
        }
    }

    inline ResourceFormat depthToColorFormat(ResourceFormat format)
    {
        switch (format)
        {
        case ResourceFormat::D16Unorm:
            return ResourceFormat::R16Unorm;
        case ResourceFormat::D24UnormS8:
            return ResourceFormat::R24UnormX8;
        case ResourceFormat::D32Float:
            return ResourceFormat::R32Float;
        case ResourceFormat::D32FloatS8X24:
            FALCOR_UNREACHABLE();
            return ResourceFormat::Unknown;
        default:
            FALCOR_ASSERT(isDepthFormat(format) == false);
            return format;
        }
    }

    inline bool doesFormatHaveAlpha(ResourceFormat format)
    {
        if (getFormatChannelCount(format) == 4)
        {
            switch (format)
            {
            case ResourceFormat::BGRX8Unorm:
            case ResourceFormat::BGRX8UnormSrgb:
                return false;
            default:
                return true;
            }
        }

        switch (format)
        {
        case ResourceFormat::Alpha32Float:
        case ResourceFormat::Alpha8Unorm:
            return true;
        default:
            return false;
        }
    }

    /** Get the supported bind-flags for a specific format
    */
    ResourceBindFlags getFormatBindFlags(ResourceFormat format);

    inline const std::string& to_string(ResourceFormat format)
    {
        FALCOR_ASSERT(kFormatDesc[(uint32_t)format].format == format);
        return kFormatDesc[(uint32_t)format].name;
    }

    inline const std::string to_string(FormatType Type)
    {
#define type_2_string(a) case FormatType::a: return #a;
        switch(Type)
        {
        type_2_string(Unknown);
        type_2_string(Float);
        type_2_string(Unorm);
        type_2_string(UnormSrgb);
        type_2_string(Snorm);
        type_2_string(Uint);
        type_2_string(Sint);
        default:
            FALCOR_UNREACHABLE();
            return "";
        }
#undef type_2_string
    }

    inline const std::string to_string(ResourceBindFlags flags)
    {
        std::string s;
        if (flags == ResourceBindFlags::None)
        {
            return "None";
        }

#define flag_to_str(f_) if (is_set(flags, ResourceBindFlags::f_)) (s += (s.size() ? " | " : "") + std::string(#f_))

        flag_to_str(Vertex);
        flag_to_str(Index);
        flag_to_str(Constant);
        flag_to_str(StreamOutput);
        flag_to_str(ShaderResource);
        flag_to_str(UnorderedAccess);
        flag_to_str(RenderTarget);
        flag_to_str(DepthStencil);
        flag_to_str(IndirectArg);
        flag_to_str(AccelerationStructure);
#undef flag_to_str

        return s;
    }

    /** Convert from Falcor to DXGI format
    */
    DXGI_FORMAT FALCOR_API getDxgiFormat(ResourceFormat format);

    /** Convert from DXGI to Falcor format
    */
    ResourceFormat FALCOR_API getResourceFormat(DXGI_FORMAT format);
    DXGI_FORMAT FALCOR_API getTypelessFormatFromDepthFormat(ResourceFormat format);

    /*! @} */
}

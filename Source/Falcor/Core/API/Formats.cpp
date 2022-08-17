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
#include "Formats.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    const FormatDesc kFormatDesc[] =
    {
        // Format                           Name,           BytesPerBlock ChannelCount  Type          {bDepth,   bStencil, bCompressed},   {CompressionRatio.Width,     CompressionRatio.Height}    {numChannelBits.x, numChannelBits.y, numChannelBits.z, numChannelBits.w}
        {ResourceFormat::Unknown,            "Unknown",         0,              0,  FormatType::Unknown,    {false,  false, false,},        {1, 1},                                                  {0, 0, 0, 0    }},
        {ResourceFormat::R8Unorm,            "R8Unorm",         1,              1,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {8, 0, 0, 0    }},
        {ResourceFormat::R8Snorm,            "R8Snorm",         1,              1,  FormatType::Snorm,      {false,  false, false,},        {1, 1},                                                  {8, 0, 0, 0    }},
        {ResourceFormat::R16Unorm,           "R16Unorm",        2,              1,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {16, 0, 0, 0   }},
        {ResourceFormat::R16Snorm,           "R16Snorm",        2,              1,  FormatType::Snorm,      {false,  false, false,},        {1, 1},                                                  {16, 0, 0, 0   }},
        {ResourceFormat::RG8Unorm,           "RG8Unorm",        2,              2,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {8, 8, 0, 0    }},
        {ResourceFormat::RG8Snorm,           "RG8Snorm",        2,              2,  FormatType::Snorm,      {false,  false, false,},        {1, 1},                                                  {8, 8, 0, 0    }},
        {ResourceFormat::RG16Unorm,          "RG16Unorm",       4,              2,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {16, 16, 0, 0  }},
        {ResourceFormat::RG16Snorm,          "RG16Snorm",       4,              2,  FormatType::Snorm,      {false,  false, false,},        {1, 1},                                                  {16, 16, 0, 0  }},
        {ResourceFormat::RGB16Unorm,         "RGB16Unorm",      6,              3,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {16, 16, 16, 0 }},
        {ResourceFormat::RGB16Snorm,         "RGB16Snorm",      6,              3,  FormatType::Snorm,      {false,  false, false,},        {1, 1},                                                  {16, 16, 16, 0 }},
        {ResourceFormat::R24UnormX8,         "R24UnormX8",      4,              2,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {24, 8, 0, 0   }},
        {ResourceFormat::RGB5A1Unorm,        "RGB5A1Unorm",     2,              4,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {5, 5, 5, 1    }},
        {ResourceFormat::RGBA8Unorm,         "RGBA8Unorm",      4,              4,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {8, 8, 8, 8    }},
        {ResourceFormat::RGBA8Snorm,         "RGBA8Snorm",      4,              4,  FormatType::Snorm,      {false,  false, false,},        {1, 1},                                                  {8, 8, 8, 8    }},
        {ResourceFormat::RGB10A2Unorm,       "RGB10A2Unorm",    4,              4,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {10, 10, 10, 2 }},
        {ResourceFormat::RGB10A2Uint,        "RGB10A2Uint",     4,              4,  FormatType::Uint,       {false,  false, false,},        {1, 1},                                                  {10, 10, 10, 2 }},
        {ResourceFormat::RGBA16Unorm,        "RGBA16Unorm",     8,              4,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {16, 16, 16, 16}},
        {ResourceFormat::RGBA8UnormSrgb,     "RGBA8UnormSrgb",  4,              4,  FormatType::UnormSrgb,  {false,  false, false,},        {1, 1},                                                  {8, 8, 8, 8    }},
        // Format                           Name,           BytesPerBlock ChannelCount  Type          {bDepth,   bStencil, bCompressed},   {CompressionRatio.Width,     CompressionRatio.Height}
        {ResourceFormat::R16Float,           "R16Float",        2,              1,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {16, 0, 0, 0   }},
        {ResourceFormat::RG16Float,          "RG16Float",       4,              2,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {16, 16, 0, 0  }},
        {ResourceFormat::RGB16Float,         "RGB16Float",      6,              3,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {16, 16, 16, 0 }},
        {ResourceFormat::RGBA16Float,        "RGBA16Float",     8,              4,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {16, 16, 16, 16}},
        {ResourceFormat::R32Float,           "R32Float",        4,              1,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {32, 0, 0, 0   }},
        {ResourceFormat::R32FloatX32,        "R32FloatX32",     8,              2,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {32, 32, 0, 0  }},
        {ResourceFormat::RG32Float,          "RG32Float",       8,              2,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {32, 32, 0, 0  }},
        {ResourceFormat::RGB32Float,         "RGB32Float",      12,             3,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {32, 32, 32, 0 }},
        {ResourceFormat::RGBA32Float,        "RGBA32Float",     16,             4,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {32, 32, 32, 32}},
        {ResourceFormat::R11G11B10Float,     "R11G11B10Float",  4,              3,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {11, 11, 10, 0 }},
        {ResourceFormat::RGB9E5Float,        "RGB9E5Float",     4,              3,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {9, 9, 9, 5    }},
        {ResourceFormat::R8Int,              "R8Int",           1,              1,  FormatType::Sint,       {false,  false, false,},        {1, 1},                                                  {8, 0, 0, 0    }},
        {ResourceFormat::R8Uint,             "R8Uint",          1,              1,  FormatType::Uint,       {false,  false, false,},        {1, 1},                                                  {8, 0, 0, 0    }},
        {ResourceFormat::R16Int,             "R16Int",          2,              1,  FormatType::Sint,       {false,  false, false,},        {1, 1},                                                  {16, 0, 0, 0   }},
        {ResourceFormat::R16Uint,            "R16Uint",         2,              1,  FormatType::Uint,       {false,  false, false,},        {1, 1},                                                  {16, 0, 0, 0   }},
        {ResourceFormat::R32Int,             "R32Int",          4,              1,  FormatType::Sint,       {false,  false, false,},        {1, 1},                                                  {32, 0, 0, 0   }},
        {ResourceFormat::R32Uint,            "R32Uint",         4,              1,  FormatType::Uint,       {false,  false, false,},        {1, 1},                                                  {32, 0, 0, 0   }},
        {ResourceFormat::RG8Int,             "RG8Int",          2,              2,  FormatType::Sint,       {false,  false, false,},        {1, 1},                                                  {8, 8, 0, 0    }},
        {ResourceFormat::RG8Uint,            "RG8Uint",         2,              2,  FormatType::Uint,       {false,  false, false,},        {1, 1},                                                  {8, 8, 0, 0    }},
        {ResourceFormat::RG16Int,            "RG16Int",         4,              2,  FormatType::Sint,       {false,  false, false,},        {1, 1},                                                  {16, 16, 0, 0  }},
        {ResourceFormat::RG16Uint,           "RG16Uint",        4,              2,  FormatType::Uint,       {false,  false, false,},        {1, 1},                                                  {16, 16, 0, 0  }},
        {ResourceFormat::RG32Int,            "RG32Int",         8,              2,  FormatType::Sint,       {false,  false, false,},        {1, 1},                                                  {32, 32, 0, 0  }},
        {ResourceFormat::RG32Uint,           "RG32Uint",        8,              2,  FormatType::Uint,       {false,  false, false,},        {1, 1},                                                  {32, 32, 0, 0  }},
        // Format                           Name,           BytesPerBlock ChannelCount  Type          {bDepth,   bStencil, bCompressed},   {CompressionRatio.Width,     CompressionRatio.Height}
        {ResourceFormat::RGB16Int,           "RGB16Int",        6,              3,  FormatType::Sint,       {false,  false, false,},        {1, 1},                                                  {16, 16, 16, 0 }},
        {ResourceFormat::RGB16Uint,          "RGB16Uint",       6,              3,  FormatType::Uint,       {false,  false, false,},        {1, 1},                                                  {16, 16, 16, 0 }},
        {ResourceFormat::RGB32Int,           "RGB32Int",       12,              3,  FormatType::Sint,       {false,  false, false,},        {1, 1},                                                  {32, 32, 32, 0 }},
        {ResourceFormat::RGB32Uint,          "RGB32Uint",      12,              3,  FormatType::Uint,       {false,  false, false,},        {1, 1},                                                  {32, 32, 32, 0 }},
        {ResourceFormat::RGBA8Int,           "RGBA8Int",        4,              4,  FormatType::Sint,       {false,  false, false,},        {1, 1},                                                  {8, 8, 8, 8    }},
        {ResourceFormat::RGBA8Uint,          "RGBA8Uint",       4,              4,  FormatType::Uint,       {false, false, false, },        {1, 1},                                                  {8, 8, 8, 8    }},
        {ResourceFormat::RGBA16Int,          "RGBA16Int",       8,              4,  FormatType::Sint,       {false,  false, false,},        {1, 1},                                                  {16, 16, 16, 16}},
        {ResourceFormat::RGBA16Uint,         "RGBA16Uint",      8,              4,  FormatType::Uint,       {false,  false, false,},        {1, 1},                                                  {16, 16, 16, 16}},
        {ResourceFormat::RGBA32Int,          "RGBA32Int",      16,              4,  FormatType::Sint,       {false,  false, false,},        {1, 1},                                                  {32, 32, 32, 32}},
        {ResourceFormat::RGBA32Uint,         "RGBA32Uint",     16,              4,  FormatType::Uint,       {false,  false, false,},        {1, 1},                                                  {32, 32, 32, 32}},
        {ResourceFormat::BGRA8Unorm,         "BGRA8Unorm",      4,              4,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {8, 8, 8, 8    }},
        {ResourceFormat::BGRA8UnormSrgb,     "BGRA8UnormSrgb",  4,              4,  FormatType::UnormSrgb,  {false,  false, false,},        {1, 1},                                                  {8, 8, 8, 8    }},
        {ResourceFormat::BGRX8Unorm,         "BGRX8Unorm",      4,              4,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {8, 8, 8, 8    }},
        {ResourceFormat::BGRX8UnormSrgb,     "BGRX8UnormSrgb",  4,              4,  FormatType::UnormSrgb,  {false,  false, false,},        {1, 1},                                                  {8, 8, 8, 8    }},
        {ResourceFormat::Alpha8Unorm,        "Alpha8Unorm",     1,              1,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {0, 0, 0, 8    }},
        {ResourceFormat::Alpha32Float,       "Alpha32Float",    4,              1,  FormatType::Float,      {false,  false, false,},        {1, 1},                                                  {0, 0, 0, 32   }},
        // Format                           Name,           BytesPerBlock ChannelCount  Type          {bDepth,   bStencil, bCompressed},   {CompressionRatio.Width,     CompressionRatio.Height}
        {ResourceFormat::R5G6B5Unorm,        "R5G6B5Unorm",     2,              3,  FormatType::Unorm,      {false,  false, false,},        {1, 1},                                                  {5, 6, 5, 0    }},
        {ResourceFormat::D32Float,           "D32Float",        4,              1,  FormatType::Float,      {true,   false, false,},        {1, 1},                                                  {32, 0, 0, 0   }},
        {ResourceFormat::D16Unorm,           "D16Unorm",        2,              1,  FormatType::Unorm,      {true,   false, false,},        {1, 1},                                                  {16, 0, 0, 0   }},
        {ResourceFormat::D32FloatS8X24,      "D32FloatS8X24",   8,              2,  FormatType::Float,      {true,   true,  false,},        {1, 1},                                                  {32, 8, 24, 0  }},
        {ResourceFormat::D24UnormS8,         "D24UnormS8",      4,              2,  FormatType::Unorm,      {true,   true,  false,},        {1, 1},                                                  {24, 8, 0, 0   }},
        {ResourceFormat::BC1Unorm,           "BC1Unorm",        8,              3,  FormatType::Unorm,      {false,  false, true, },        {4, 4},                                                  {64, 0, 0, 0   }},
        {ResourceFormat::BC1UnormSrgb,       "BC1UnormSrgb",    8,              3,  FormatType::UnormSrgb,  {false,  false, true, },        {4, 4},                                                  {64, 0, 0, 0   }},
        {ResourceFormat::BC2Unorm,           "BC2Unorm",        16,             4,  FormatType::Unorm,      {false,  false, true, },        {4, 4},                                                  {128, 0, 0, 0  }},
        {ResourceFormat::BC2UnormSrgb,       "BC2UnormSrgb",    16,             4,  FormatType::UnormSrgb,  {false,  false, true, },        {4, 4},                                                  {128, 0, 0, 0  }},
        {ResourceFormat::BC3Unorm,           "BC3Unorm",        16,             4,  FormatType::Unorm,      {false,  false, true, },        {4, 4},                                                  {128, 0, 0, 0  }},
        {ResourceFormat::BC3UnormSrgb,       "BC3UnormSrgb",    16,             4,  FormatType::UnormSrgb,  {false,  false, true, },        {4, 4},                                                  {128, 0, 0, 0  }},
        {ResourceFormat::BC4Unorm,           "BC4Unorm",        8,              1,  FormatType::Unorm,      {false,  false, true, },        {4, 4},                                                  {64, 0, 0, 0   }},
        {ResourceFormat::BC4Snorm,           "BC4Snorm",        8,              1,  FormatType::Snorm,      {false,  false, true, },        {4, 4},                                                  {64, 0, 0, 0   }},
        {ResourceFormat::BC5Unorm,           "BC5Unorm",        16,             2,  FormatType::Unorm,      {false,  false, true, },        {4, 4},                                                  {128, 0, 0, 0  }},
        {ResourceFormat::BC5Snorm,           "BC5Snorm",        16,             2,  FormatType::Snorm,      {false,  false, true, },        {4, 4},                                                  {128, 0, 0, 0  }},

        {ResourceFormat::BC6HS16,            "BC6HS16",         16,             3,  FormatType::Float,      {false,  false, true, },        {4, 4},                                                  {128, 0, 0, 0  }},
        {ResourceFormat::BC6HU16,            "BC6HU16",         16,             3,  FormatType::Float,      {false,  false, true, },        {4, 4},                                                  {128, 0, 0, 0  }},
        {ResourceFormat::BC7Unorm,           "BC7Unorm",        16,             4,  FormatType::Unorm,      {false,  false, true, },        {4, 4},                                                  {128, 0, 0, 0  }},
        {ResourceFormat::BC7UnormSrgb,       "BC7UnormSrgb",    16,             4,  FormatType::UnormSrgb,  {false,  false, true, },        {4, 4},                                                  {128, 0, 0, 0  }},
    };

    static_assert(std::size(kFormatDesc) == (size_t)ResourceFormat::BC7UnormSrgb + 1, "Format desc table has a wrong size");

    FALCOR_SCRIPT_BINDING(Formats)
    {
        // Resource formats
        pybind11::enum_<ResourceFormat> resourceFormat(m, "ResourceFormat");
        for (uint32_t i = 0; i < (uint32_t)ResourceFormat::Count; i++)
        {
            resourceFormat.value(to_string(ResourceFormat(i)).c_str(), ResourceFormat(i));
        }

        pybind11::enum_<TextureChannelFlags> textureChannels(m, "TextureChannelFlags");
        // TODO: These generate an "invalid format string" error from Python.
        //textureChannels.value("Red", TextureChannelFlags::Red);
        //textureChannels.value("Green", TextureChannelFlags::Green);
        //textureChannels.value("Blue", TextureChannelFlags::Blue);
        textureChannels.value("Alpha", TextureChannelFlags::Alpha);
        textureChannels.value("RGB", TextureChannelFlags::RGB);
        textureChannels.value("RGBA", TextureChannelFlags::RGBA);
    }

    struct DxgiFormatDesc
    {
        ResourceFormat falcorFormat;
        DXGI_FORMAT dxgiFormat;
    };

    const DxgiFormatDesc kDxgiFormatDesc[] =
    {
        {ResourceFormat::Unknown,                       DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::R8Unorm,                       DXGI_FORMAT_R8_UNORM},
        {ResourceFormat::R8Snorm,                       DXGI_FORMAT_R8_SNORM},
        {ResourceFormat::R16Unorm,                      DXGI_FORMAT_R16_UNORM},
        {ResourceFormat::R16Snorm,                      DXGI_FORMAT_R16_SNORM},
        {ResourceFormat::RG8Unorm,                      DXGI_FORMAT_R8G8_UNORM},
        {ResourceFormat::RG8Snorm,                      DXGI_FORMAT_R8G8_SNORM},
        {ResourceFormat::RG16Unorm,                     DXGI_FORMAT_R16G16_UNORM},
        {ResourceFormat::RG16Snorm,                     DXGI_FORMAT_R16G16_SNORM},
        {ResourceFormat::RGB16Unorm,                    DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::RGB16Snorm,                    DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::R24UnormX8,                    DXGI_FORMAT_R24_UNORM_X8_TYPELESS},
        {ResourceFormat::RGB5A1Unorm,                   DXGI_FORMAT_B5G5R5A1_UNORM},
        {ResourceFormat::RGBA8Unorm,                    DXGI_FORMAT_R8G8B8A8_UNORM},
        {ResourceFormat::RGBA8Snorm,                    DXGI_FORMAT_R8G8B8A8_SNORM},
        {ResourceFormat::RGB10A2Unorm,                  DXGI_FORMAT_R10G10B10A2_UNORM},
        {ResourceFormat::RGB10A2Uint,                   DXGI_FORMAT_R10G10B10A2_UINT},
        {ResourceFormat::RGBA16Unorm,                   DXGI_FORMAT_R16G16B16A16_UNORM},
        {ResourceFormat::RGBA8UnormSrgb,                DXGI_FORMAT_R8G8B8A8_UNORM_SRGB},
        {ResourceFormat::R16Float,                      DXGI_FORMAT_R16_FLOAT},
        {ResourceFormat::RG16Float,                     DXGI_FORMAT_R16G16_FLOAT},
        {ResourceFormat::RGB16Float,                    DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::RGBA16Float,                   DXGI_FORMAT_R16G16B16A16_FLOAT},
        {ResourceFormat::R32Float,                      DXGI_FORMAT_R32_FLOAT},
        {ResourceFormat::R32FloatX32,                   DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS},
        {ResourceFormat::RG32Float,                     DXGI_FORMAT_R32G32_FLOAT},
        {ResourceFormat::RGB32Float,                    DXGI_FORMAT_R32G32B32_FLOAT},
        {ResourceFormat::RGBA32Float,                   DXGI_FORMAT_R32G32B32A32_FLOAT},
        {ResourceFormat::R11G11B10Float,                DXGI_FORMAT_R11G11B10_FLOAT},
        {ResourceFormat::RGB9E5Float,                   DXGI_FORMAT_R9G9B9E5_SHAREDEXP},
        {ResourceFormat::R8Int,                         DXGI_FORMAT_R8_SINT},
        {ResourceFormat::R8Uint,                        DXGI_FORMAT_R8_UINT},
        {ResourceFormat::R16Int,                        DXGI_FORMAT_R16_SINT},
        {ResourceFormat::R16Uint,                       DXGI_FORMAT_R16_UINT},
        {ResourceFormat::R32Int,                        DXGI_FORMAT_R32_SINT},
        {ResourceFormat::R32Uint,                       DXGI_FORMAT_R32_UINT},
        {ResourceFormat::RG8Int,                        DXGI_FORMAT_R8G8_SINT},
        {ResourceFormat::RG8Uint,                       DXGI_FORMAT_R8G8_UINT},
        {ResourceFormat::RG16Int,                       DXGI_FORMAT_R16G16_SINT},
        {ResourceFormat::RG16Uint,                      DXGI_FORMAT_R16G16_UINT},
        {ResourceFormat::RG32Int,                       DXGI_FORMAT_R32G32_SINT},
        {ResourceFormat::RG32Uint,                      DXGI_FORMAT_R32G32_UINT},
        {ResourceFormat::RGB16Int,                      DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::RGB16Uint,                     DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::RGB32Int,                      DXGI_FORMAT_R32G32B32_SINT},
        {ResourceFormat::RGB32Uint,                     DXGI_FORMAT_R32G32B32_UINT},
        {ResourceFormat::RGBA8Int,                      DXGI_FORMAT_R8G8B8A8_SINT},
        {ResourceFormat::RGBA8Uint,                     DXGI_FORMAT_R8G8B8A8_UINT},
        {ResourceFormat::RGBA16Int,                     DXGI_FORMAT_R16G16B16A16_SINT},
        {ResourceFormat::RGBA16Uint,                    DXGI_FORMAT_R16G16B16A16_UINT},
        {ResourceFormat::RGBA32Int,                     DXGI_FORMAT_R32G32B32A32_SINT},
        {ResourceFormat::RGBA32Uint,                    DXGI_FORMAT_R32G32B32A32_UINT},
        {ResourceFormat::BGRA8Unorm,                    DXGI_FORMAT_B8G8R8A8_UNORM},
        {ResourceFormat::BGRA8UnormSrgb,                DXGI_FORMAT_B8G8R8A8_UNORM_SRGB},
        {ResourceFormat::BGRX8Unorm,                    DXGI_FORMAT_B8G8R8X8_UNORM},
        {ResourceFormat::BGRX8UnormSrgb,                DXGI_FORMAT_B8G8R8X8_UNORM_SRGB},
        {ResourceFormat::Alpha8Unorm,                   DXGI_FORMAT_A8_UNORM},
        {ResourceFormat::Alpha32Float,                  DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::R5G6B5Unorm,                   DXGI_FORMAT_B5G6R5_UNORM},
        {ResourceFormat::D32Float,                      DXGI_FORMAT_D32_FLOAT},
        {ResourceFormat::D16Unorm,                      DXGI_FORMAT_D16_UNORM},
        {ResourceFormat::D32FloatS8X24,                 DXGI_FORMAT_D32_FLOAT_S8X24_UINT},
        {ResourceFormat::D24UnormS8,                    DXGI_FORMAT_D24_UNORM_S8_UINT},
        {ResourceFormat::BC1Unorm,                      DXGI_FORMAT_BC1_UNORM},
        {ResourceFormat::BC1UnormSrgb,                  DXGI_FORMAT_BC1_UNORM_SRGB},
        {ResourceFormat::BC2Unorm,                      DXGI_FORMAT_BC2_UNORM},
        {ResourceFormat::BC2UnormSrgb,                  DXGI_FORMAT_BC2_UNORM_SRGB},
        {ResourceFormat::BC3Unorm,                      DXGI_FORMAT_BC3_UNORM},
        {ResourceFormat::BC3UnormSrgb,                  DXGI_FORMAT_BC3_UNORM_SRGB},
        {ResourceFormat::BC4Unorm,                      DXGI_FORMAT_BC4_UNORM},
        {ResourceFormat::BC4Snorm,                      DXGI_FORMAT_BC4_SNORM},
        {ResourceFormat::BC5Unorm,                      DXGI_FORMAT_BC5_UNORM},
        {ResourceFormat::BC5Snorm,                      DXGI_FORMAT_BC5_SNORM},
        {ResourceFormat::BC6HS16,                       DXGI_FORMAT_BC6H_SF16},
        {ResourceFormat::BC6HU16,                       DXGI_FORMAT_BC6H_UF16},
        {ResourceFormat::BC7Unorm,                      DXGI_FORMAT_BC7_UNORM},
        {ResourceFormat::BC7UnormSrgb,                  DXGI_FORMAT_BC7_UNORM_SRGB},
    };

    static_assert(std::size(kDxgiFormatDesc) == (size_t)ResourceFormat::Count, "DXGI format desc table has a wrong size");

    DXGI_FORMAT getDxgiFormat(ResourceFormat format)
    {
        FALCOR_ASSERT(kDxgiFormatDesc[(uint32_t)format].falcorFormat == format);
        return kDxgiFormatDesc[(uint32_t)format].dxgiFormat;
    }

    ResourceFormat getResourceFormat(DXGI_FORMAT format)
    {
        for (size_t i = 0; i < (size_t)ResourceFormat::Count; ++i)
        {
            const auto& desc = kDxgiFormatDesc[i];
            if (desc.dxgiFormat == format) return desc.falcorFormat;
        }
        return ResourceFormat::Unknown;
    }

    DXGI_FORMAT getTypelessFormatFromDepthFormat(ResourceFormat format)
    {
        switch (format)
        {
        case ResourceFormat::D16Unorm:
            return DXGI_FORMAT_R16_TYPELESS;
        case ResourceFormat::D32FloatS8X24:
            return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        case ResourceFormat::D24UnormS8:
            return DXGI_FORMAT_R24G8_TYPELESS;
        case ResourceFormat::D32Float:
            return DXGI_FORMAT_R32_TYPELESS;
        default:
            FALCOR_ASSERT(isDepthFormat(format) == false);
            return kDxgiFormatDesc[(uint32_t)format].dxgiFormat;
        }
    }
}

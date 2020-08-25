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
#include "stdafx.h"
#include "Formats.h"

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

    static_assert(arraysize(kFormatDesc) == (uint32_t)ResourceFormat::BC7UnormSrgb + 1, "Format desc table has a wrong size");

    SCRIPT_BINDING(ResourceFormat)
    {
        // Resource formats
        pybind11::enum_<ResourceFormat> resourceFormat(m, "ResourceFormat");
        for (uint32_t i = 0; i < (uint32_t)ResourceFormat::Count; i++)
        {
            resourceFormat.value(to_string(ResourceFormat(i)).c_str(), ResourceFormat(i));
        }
    }
}

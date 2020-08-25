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

/** GPU side implementation of pixel debugging utils.

    Call the function printSetPixel() in the shader program to set the
    current pixel before the first print() or assert() calls.

    The host sets the following defines:

    _PIXEL_DEBUG_ENABLED     Nonzero when pixel debugging is enabled.

*/

import PixelDebugTypes;

cbuffer PixelDebugCB
{
    uint2 gPixelLogSelected;    // Currently selected pixel to log.
    uint  gPixelLogSize;        // Number of elements in the output buffer.
    uint  gAssertLogSize;
};

RWStructuredBuffer<PixelLogValue> gPixelLog;
RWStructuredBuffer<AssertLogValue> gAssertLog;

#ifdef _PIXEL_DEBUG_ENABLED
static uint2 gPixelDebugPixel;
#endif

void printSetPixel(uint2 pixel)
{
#ifdef _PIXEL_DEBUG_ENABLED
    gPixelDebugPixel = pixel;
#endif
}

#ifdef _PIXEL_DEBUG_ENABLED

    /** Define overloaded 'void print(Type x)' functions. Each takes a different
        basic type as parameter and appends it in encoded form to the log.
    */
    #define PRINT_FUNC(Type, Count, ValueType)                  \
    void print(vector<Type, Count> v)                           \
    {                                                           \
        if (all(gPixelDebugPixel == gPixelLogSelected))         \
        {                                                       \
            uint i = gPixelLog.IncrementCounter();              \
            if (i < gPixelLogSize)                              \
            {                                                   \
                PixelLogValue val;                              \
                val.type = (uint)ValueType;                     \
                val.count = Count;                              \
                for (int j = 0; j < 4; j++)                     \
                {                                               \
                    val.data[j] = j < Count ? asuint(v[j]) : 0; \
                }                                               \
                gPixelLog[i] = val;                             \
            }                                                   \
        }                                                       \
    }

    /** Shader assert.
        \param[in] condition If false, the current pixel is recorded in the log.
    */
    void assert(bool condition)
    {
        if (!condition)
        {
            uint i = gAssertLog.IncrementCounter();
            if (i < gAssertLogSize)
            {
                AssertLogValue val;
                val.launchIndex = uint3(gPixelDebugPixel, 0);
                gAssertLog[i] = val;
            }
        }
    }

#else

    /** Define null functions if debugging is disabled.
    */
    #define PRINT_FUNC(Type, Count, DataType) \
    void print(vector<Type, Count> v) {}

    void assert(bool condition) {}

#endif // !_PIXEL_DEBUG_ENABLED

PRINT_FUNC(bool, 1, PixelLogValueType::Bool)
PRINT_FUNC(bool, 2, PixelLogValueType::Bool)
PRINT_FUNC(bool, 3, PixelLogValueType::Bool)
PRINT_FUNC(bool, 4, PixelLogValueType::Bool)
PRINT_FUNC(int, 1, PixelLogValueType::Int)
PRINT_FUNC(int, 2, PixelLogValueType::Int)
PRINT_FUNC(int, 3, PixelLogValueType::Int)
PRINT_FUNC(int, 4, PixelLogValueType::Int)
PRINT_FUNC(uint, 1, PixelLogValueType::Uint)
PRINT_FUNC(uint, 2, PixelLogValueType::Uint)
PRINT_FUNC(uint, 3, PixelLogValueType::Uint)
PRINT_FUNC(uint, 4, PixelLogValueType::Uint)
PRINT_FUNC(float, 1, PixelLogValueType::Float)
PRINT_FUNC(float, 2, PixelLogValueType::Float)
PRINT_FUNC(float, 3, PixelLogValueType::Float)
PRINT_FUNC(float, 4, PixelLogValueType::Float)

#undef PRINT_FUNC

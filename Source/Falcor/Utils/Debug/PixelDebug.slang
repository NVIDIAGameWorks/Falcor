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

/**
 * GPU side implementation of pixel debugging utils.
 *
 * Call the function printSetPixel() in the shader program to set the
 * current pixel before calling any of the following functions:
 *
 * print(value) for printing basic types
 * print(msg, value) for printing basic types with a prepended string
 * assert(condition, msg) for asserting on a condition (msg is optional)
 *
 * The host sets the following defines:
 *
 * _PIXEL_DEBUG_ENABLED     Defined when pixel debugging is enabled.
 *
 */

import PixelDebugTypes;

struct PixelDebug
{
    RWByteAddressBuffer counterBuffer;
    RWStructuredBuffer<PrintRecord> printBuffer;
    RWStructuredBuffer<AssertRecord> assertBuffer;

    uint printBufferCapacity;  ///< Capacity of the print buffer.
    uint assertBufferCapacity; ///< Capacity of the assert buffer.

    uint2 selectedPixel; ///< Currently selected pixel to log.

    /**
     * Add print record.
     * @param[in] pixel The pixel the print originated from.
     * @param[in] msg Message string.
     * @param[in] valueType The type of values encoded into `data`
     * @param[in] count The number of components of `valueType` encoded into `data`
     * @param[in] data The bits of the value to be printed.
     */
    [ForceInline]
    void print(uint2 pixel, String msg, PrintValueType valueType, int count, uint4 data)
    {
        // TODO: Previously this was an early out:
        // if (all(pixel == selectedPixel)) return;
        // Due to a bug in slang this leads to expontentially growing compile times.
        // This will be fixed in the compiler. Once it is, we should revert this workaround.
        if (all(pixel == selectedPixel))
        {
            uint index = 0;
            counterBuffer.InterlockedAdd(0, 1, index);
            if (index < printBufferCapacity)
            {
                PrintRecord rec = {};
                rec.msgHash = getStringHash(msg);
                rec.type = (uint)valueType;
                rec.count = count;
                rec.data = data;
                printBuffer[index] = rec;
            }
        }
    }

    /**
     * Add an assert record.
     * @param[in] pixel The pixel the print originated from.
     * @param[in] msg Message string.
     */
    [ForceInline]
    void assert(uint2 pixel, String msg)
    {
        uint index = 0;
        counterBuffer.InterlockedAdd(4, 1, index);
        if (index < assertBufferCapacity)
        {
            AssertRecord rec;
            rec.launchIndex = uint3(pixel, 0);
            rec.msgHash = getStringHash(msg);
            assertBuffer[index] = rec;
        }
    }
};

ParameterBlock<PixelDebug> gPixelDebug;
static uint2 gPixelDebugPixel;

void printSetPixel(uint2 pixel)
{
#ifdef _PIXEL_DEBUG_ENABLED
    gPixelDebugPixel = pixel;
#endif
}

/// Interface for values that can be printed.
interface IPrintable
{
    /// Convert this value to a single `uint` holding its bits.
    uint getPrintableDataVal();

    /// Get the `PrintValueType` that represents this type.
    static PrintValueType getPrintableValueType();
}

/// Values of type `bool` are printable.
extension bool : IPrintable
{
    uint getPrintableDataVal()
    {
        return uint(this);
    }
    static PrintValueType getPrintableValueType()
    {
        return PrintValueType::Bool;
    }
}

/// Values of type `int` are printable.
extension int : IPrintable
{
    uint getPrintableDataVal()
    {
        return asuint(this);
    }
    static PrintValueType getPrintableValueType()
    {
        return PrintValueType::Int;
    }
}

/// Values of type `uint` are printable.
extension uint : IPrintable
{
    uint getPrintableDataVal()
    {
        return this;
    }
    static PrintValueType getPrintableValueType()
    {
        return PrintValueType::Uint;
    }
}

/// Values of type `float` are printable.
extension float : IPrintable
{
    uint getPrintableDataVal()
    {
        return asuint(this);
    }
    static PrintValueType getPrintableValueType()
    {
        return PrintValueType::Float;
    }
}

#ifdef _PIXEL_DEBUG_ENABLED

/**
 * Print a value if the current pixel is the selected one.
 * @param[in] msg A string message to accompany the value.
 * @param[in] v The value to print
 */
[ForceInline]
void print<T : IPrintable>(String msg, T v)
{
    vector<uint, 4> u = 0;
    u[0] = v.getPrintableDataVal();
    gPixelDebug.print(gPixelDebugPixel, msg, T.getPrintableValueType(), 1, u);
}

/**
 * Print a vector or values if the current pixel is the selected one.
 * @param[in] msg A string message to accompany the value.
 * @param[in] v The value to print
 */
[ForceInline]
void print<T : IPrintable, let N : int>(String msg, vector<T, N> v)
{
    vector<uint, 4> u = 0;
    for (int i = 0; i < N; ++i)
        u[i] = v[i].getPrintableDataVal();
    gPixelDebug.print(gPixelDebugPixel, msg, T.getPrintableValueType(), N, u);
}

/**
 * Shader assert.
 * @param[in] condition If false, the current pixel is recorded in the log.
 */
[ForceInline]
void assert(bool condition, String msg = "")
{
    if (!condition)
        gPixelDebug.assert(gPixelDebugPixel, msg);
}

#else // _PIXEL_DEBUG_ENABLED

/**
 * Define null functions if debugging is disabled.
 */

[ForceInline]
void print<T : IPrintable, let N : int>(String msg, vector<T, N> v) {}

[ForceInline]
void print<T : IPrintable>(String msg, T v) {}

[ForceInline]
void assert(bool condition, String msg = "") {}

#endif // _PIXEL_DEBUG_ENABLED

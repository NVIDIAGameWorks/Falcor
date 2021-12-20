/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
    current pixel before calling any of the following functions:

    print(value) for printing basic types
    print(msg, value) for printing basic types with a prepended string
    assert(condition, msg) for asserting on a condition (msg is optional)

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

/** Interface for values that can be printed.
*/
interface IPrintable
{
    /** Convert this value to a single `uint` holding its bits.
    */
    uint getPrintableDataVal();

    /** Get the `PixelLogValueType` that represents this type.
    */
    static PixelLogValueType getPrintableValueType();
}

/** Values of type `bool` are printable.
*/
extension bool : IPrintable
{
    uint getPrintableDataVal() { return uint(this); }
    static PixelLogValueType getPrintableValueType() { return PixelLogValueType::Bool; }
}

/** Values of type `int` are printable.
*/
extension int : IPrintable
{
    uint getPrintableDataVal() { return asuint(this); }
    static PixelLogValueType getPrintableValueType() { return PixelLogValueType::Int; }
}

/** Values of type `uint` are printable.
*/
extension uint : IPrintable
{
    uint getPrintableDataVal() { return this; }
    static PixelLogValueType getPrintableValueType() { return PixelLogValueType::Uint; }
}

/** Values of type `float` are printable.
*/
extension float : IPrintable
{
    uint getPrintableDataVal() { return asuint(this); }
    static PixelLogValueType getPrintableValueType() { return PixelLogValueType::Float; }
}

#ifdef _PIXEL_DEBUG_ENABLED

    /** Add a value to the pixel log if the current pixel is the selected one.
        \param[in] msg A string message to accompany the value.
        \param[in] valueType The type of values encoded into `data`
        \param[in] count The number of components of `valueType` encoded into `data`
        \param[in] data The bits of the value to be printed.

        This is a lower-level worker routine used by `print()`.
    */
    void _print(String msg, PixelLogValueType valueType, int count, uint4 data)
    {
        if (!all(gPixelDebugPixel == gPixelLogSelected)) return;

        uint i = gPixelLog.IncrementCounter();
        if (i < gPixelLogSize)
        {
            PixelLogValue val = {};
            val.msgHash = getStringHash(msg);
            val.type = (uint)valueType;
            val.count = count;
            val.data = data;
            gPixelLog[i] = val;
        }
    }

    /** Print a value if the current pixel is the selected one.
        \param[in] msg A string message to accompany the value.
        \param[in] v The value to print
    */
    void print<T : IPrintable>(String msg, T v)
    {
        vector<uint,4> u = 0;
        u[0] = v.getPrintableDataVal();
        _print(msg, T.getPrintableValueType(), 1, u);
    }

    /** Print a vector or values if the current pixel is the selected one.
        \param[in] msg A string message to accompany the value.
        \param[in] v The value to print
    */
    void print<T : IPrintable, let N : int>(String msg, vector<T,N> v)
    {
        vector<uint,4> u = 0;
        for(int i = 0; i < N; ++i)
        {
            u[i] = v[i].getPrintableDataVal();
        }
        _print(msg, T.getPrintableValueType(), N, u);
    }

    /** Shader assert.
        \param[in] condition If false, the current pixel is recorded in the log.
    */
    void assert(bool condition, String msg = "")
    {
        if (!condition)
        {
            uint i = gAssertLog.IncrementCounter();
            if (i < gAssertLogSize)
            {
                AssertLogValue val;
                val.launchIndex = uint3(gPixelDebugPixel, 0);
                val.msgHash = getStringHash(msg);
                gAssertLog[i] = val;
            }
        }
    }

#else

    /** Define null functions if debugging is disabled.
    */

    void print<T : IPrintable, let N : int>(String msg, vector<T,N> v)
    {}

    void print<T : IPrintable>(String msg, T v)
    {}

    void assert(bool condition, String msg = "") {}

#endif // !_PIXEL_DEBUG_ENABLED

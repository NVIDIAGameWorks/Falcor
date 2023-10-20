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
#include "PixelDebugTypes.slang"
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/API/Fence.h"
#include "Core/Program/Program.h"
#include "Utils/UI/Gui.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace Falcor
{
class RenderContext;

/**
 * Helper class for shader debugging using print() and assert().
 *
 * Host-side integration:
 * - Create PixelDebug object
 * - Call beginFrame()/endFrame() before and after executing programs with debugging.
 * - Call prepareProgram() before launching a program to use debugging.
 * - Call onMouseEvent() and renderUI() from the respective callbacks in the render pass.
 *
 * Runtime usage:
 * - Import PixelDebug.slang in your shader.
 * - Use printSetPixel() in the shader to set the current pixel.
 * - Use print() in the shader to output values for the selected pixel.
 * All basic types (e.g. bool, int3, float2, uint4) are supported.
 * - Click the left mouse button (or edit the coords) to select a pixel.
 * - Use assert() in the shader to test a condition for being true.
 * All pixels are tested, and failed asserts logged. The coordinates
 * of asserts that trigger can be used with print() to debug further.
 *
 * The shader code is disabled (using macros) when debugging is off.
 * When enabled, async readback is used but expect a minor perf loss.
 */
class FALCOR_API PixelDebug
{
public:
    /**
     * Constructor. Throws an exception on error.
     * @param[in] pDevice GPU device.
     * @param[in] printCapacity Maximum number of shader print() statements per frame.
     * @param[in] assertCapacity Maximum number of shader assert() statements per frame.
     */
    PixelDebug(ref<Device> pDevice, uint32_t printCapacity = 100, uint32_t assertCapacity = 100);

    void beginFrame(RenderContext* pRenderContext, const uint2& frameDim);
    void endFrame(RenderContext* pRenderContext);

    /**
     * Perform program specialization and bind resources.
     * This call doesn't change any resource declarations in the program.
     */
    void prepareProgram(const ref<Program>& pProgram, const ShaderVar& var);

    void renderUI(Gui::Widgets& widget) { renderUI(&widget); }
    void renderUI(Gui::Widgets* widget = nullptr);
    bool onMouseEvent(const MouseEvent& mouseEvent);

    void enable() { mEnabled = true; }

protected:
    bool copyDataToCPU();

    // Internal state
    ref<Device> mpDevice;
    ref<Program> mpReflectProgram; ///< Program for reflection of types.
    ref<Buffer> mpCounterBuffer;   ///< Counter buffer (print, assert) on the GPU.
    ref<Buffer> mpPrintBuffer;     ///< Print buffer on the GPU.
    ref<Buffer> mpAssertBuffer;    ///< Assert buffer on the GPU.
    ref<Buffer> mpReadbackBuffer;  ///< Staging buffer for async readback of all data.
    ref<Fence> mpFence;            ///< GPU fence for sychronizing readback.

    // Configuration
    bool mEnabled = false;         ///< Enable debugging features.
    uint2 mSelectedPixel = {0, 0}; ///< Currently selected pixel.

    // Runtime data
    uint2 mFrameDim = {0, 0};

    bool mRunning = false;        ///< True when data collection is running (inbetween begin()/end() calls).
    bool mWaitingForData = false; ///< True if we are waiting for data to become available on the GPU.
    bool mDataValid = false;      ///< True if data has been read back and is valid.

    std::unordered_map<uint32_t, std::string> mHashToString; ///< Map of string hashes to string values.

    std::vector<PrintRecord> mPrintData;   ///< Print data read back from the GPU.
    std::vector<AssertRecord> mAssertData; ///< Assert log data read back from the GPU.

    const uint32_t mPrintCapacity = 0;  ///< Capacity of the print buffer in elements.
    const uint32_t mAssertCapacity = 0; ///< Capacity of the assert buffer in elements.
};
} // namespace Falcor

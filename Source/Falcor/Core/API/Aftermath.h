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
 * This file defines a wrapper of NSight Aftermath SDK and is used within
 * the Device class to add Aftermath support.
 *
 * Aftermath generates a GPU crash dump when the application crashes. This can
 * be very useful for debugging GPU crashes.
 */

#pragma once

#if FALCOR_HAS_AFTERMATH

#include "Core/Macros.h"

#include <string_view>

namespace Falcor
{

class Device;
class LowLevelContextData;

/// Aftermath feature flags.
/// See section on GFSDK_Aftermath_FeatureFlags in GFSDK_Aftermath.h for details.
/// Note: For using EnableMarkers, the Aftermath Monitor must be running on the
/// host machine.
enum AftermathFlags
{
    Minimum = 0x00000000,
    EnableMarkers = 0x00000001,
    EnableResourceTracking = 0x00000002,
    CallStackCapturing = 0x40000000,
    GenerateShaderDebugInfo = 0x00000008,
    EnableShaderErrorReporting = 0x00000010,

    Defaults = EnableMarkers | EnableResourceTracking | CallStackCapturing | GenerateShaderDebugInfo | EnableShaderErrorReporting,
};
FALCOR_ENUM_CLASS_OPERATORS(AftermathFlags);

/// Aftermath per-device context.
class AftermathContext
{
public:
    AftermathContext(Device* pDevice);
    ~AftermathContext();

    /// Initialize Aftermath on the device.
    bool initialize(AftermathFlags flags = AftermathFlags::Defaults);

    /// Add a marker to the command list.
    void addMarker(const LowLevelContextData* pLowLevelContextData, std::string_view name);

private:
    Device* mpDevice;
    bool mInitialized = false;
    void* mpLastCommandList = nullptr;
    int32_t mContextHandle = 0;
};

/// Enable GPU crash dump tracking.
void enableAftermath();

/// Disable GPU crash dump tracking.
void disableAftermath();

/// Wait for GPU crash dumps to be generated. Timeout is in seconds.
bool waitForAftermathDumps(int timeout = 5);

} // namespace Falcor

#endif // FALCOR_HAS_AFTERMATH

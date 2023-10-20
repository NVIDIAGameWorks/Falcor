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
#include "GFXAPI.h"
#include "Aftermath.h"
#include "Core/Error.h"
#include "Utils/Logger.h"

#if FALCOR_HAS_D3D12
#include "dxgi.h"
#endif

namespace Falcor
{
void gfxReportError(const char* api, const char* call, gfx::Result result)
{
    const char* resultStr = nullptr;
#if FALCOR_HAS_D3D12
    switch (result)
    {
    case DXGI_ERROR_DEVICE_REMOVED:
        resultStr = "DXGI_ERROR_DEVICE_REMOVED";
        break;
    case DXGI_ERROR_DEVICE_HUNG:
        resultStr = "DXGI_ERROR_DEVICE_HUNG";
        break;
    case DXGI_ERROR_DEVICE_RESET:
        resultStr = "DXGI_ERROR_DEVICE_RESET";
        break;
    }
#endif

#if FALCOR_HAS_AFTERMATH
    if (!waitForAftermathDumps())
        logError("Aftermath GPU crash dump generation failed.");
#endif

    std::string fullMsg = resultStr ? fmt::format("{} call '{}' failed with error {} ({}).", api, call, result, resultStr)
                                    : fmt::format("{} call '{}' failed with error {}", api, call, result);

    reportFatalErrorAndTerminate(fullMsg);
}
} // namespace Falcor

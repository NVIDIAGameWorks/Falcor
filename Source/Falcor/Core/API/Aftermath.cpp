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
#if FALCOR_HAS_AFTERMATH

#include "Aftermath.h"
#include "Device.h"
#include "LowLevelContextData.h"
#include "NativeHandle.h"
#include "NativeHandleTraits.h"
#include "Core/Macros.h"
#include "Core/Version.h"
#include "Utils/Logger.h"

#include <GFSDK_Aftermath.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>
#include <GFSDK_Aftermath_GpuCrashDumpDecoding.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Note: This is based on the Aftermath sample code.

// Helper for comparing GFSDK_Aftermath_ShaderDebugInfoIdentifier.
inline bool operator<(const GFSDK_Aftermath_ShaderDebugInfoIdentifier& lhs, const GFSDK_Aftermath_ShaderDebugInfoIdentifier& rhs)
{
    if (lhs.id[0] == rhs.id[0])
    {
        return lhs.id[1] < rhs.id[1];
    }
    return lhs.id[0] < rhs.id[0];
}

namespace Falcor
{

static std::string getResultString(GFSDK_Aftermath_Result result)
{
    switch (result)
    {
    case GFSDK_Aftermath_Result_FAIL_DriverVersionNotSupported:
        return "Unsupported driver version - requires an NVIDIA R495 display driver or newer.";
    case GFSDK_Aftermath_Result_FAIL_D3dDllInterceptionNotSupported:
        return "Aftermath is incompatible with D3D API interception, such as PIX or Nsight Graphics.";
    case GFSDK_Aftermath_Result_FAIL_D3DDebugLayerNotCompatible:
        return "Aftermath is incompatible with the D3D debug layer.";
    default:
        return fmt::format("Aftermath error {:#x}", int32_t(result));
    }
}

// Helper macro for checking Nsight Aftermath results and throwing exception
// in case of a failure.
#define AFTERMATH_CHECK_ERROR(FC)                   \
    do                                              \
    {                                               \
        GFSDK_Aftermath_Result _result = FC;        \
        if (!GFSDK_Aftermath_SUCCEED(_result))      \
            FALCOR_THROW(getResultString(_result)); \
    } while (0)

static std::mutex sMutex;
static bool sInitialized = false;
static std::map<GFSDK_Aftermath_ShaderDebugInfoIdentifier, std::vector<uint8_t>> sShaderDebugInfo;

static void shaderDebugInfoLookupCallback(
    const GFSDK_Aftermath_ShaderDebugInfoIdentifier* pIdentifier,
    PFN_GFSDK_Aftermath_SetData setShaderDebugInfo,
    void* pUserData
);
static void shaderLookupCallback(
    const GFSDK_Aftermath_ShaderBinaryHash* pShaderHash,
    PFN_GFSDK_Aftermath_SetData setShaderBinary,
    void* pUserData
);
static void shaderSourceDebugInfoLookupCallback(
    const GFSDK_Aftermath_ShaderDebugName* pShaderDebugName,
    PFN_GFSDK_Aftermath_SetData setShaderBinary,
    void* pUserData
);

/// Helper for writing a GPU crash dump to a file.
static void writeGpuCrashDumpToFile(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize)
{
    // Create a GPU crash dump decoder object for the GPU crash dump.
    GFSDK_Aftermath_GpuCrashDump_Decoder decoder = {};
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_CreateDecoder(GFSDK_Aftermath_Version_API, pGpuCrashDump, gpuCrashDumpSize, &decoder)
    );

    // Use the decoder object to read basic information, like application
    // name, PID, etc. from the GPU crash dump.
    GFSDK_Aftermath_GpuCrashDump_BaseInfo baseInfo = {};
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetBaseInfo(decoder, &baseInfo));

    // Use the decoder object to query the application name that was set
    // in the GPU crash dump description.
    uint32_t applicationNameLength = 0;
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetDescriptionSize(
        decoder, GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, &applicationNameLength
    ));

    std::vector<char> applicationName(applicationNameLength, '\0');

    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetDescription(
        decoder, GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, uint32_t(applicationName.size()), applicationName.data()
    ));

    // Create a unique file name for writing the crash dump data to a file.
    // Note: due to an Nsight Aftermath bug (will be fixed in an upcoming
    // driver release) we may see redundant crash dumps. As a workaround,
    // attach a unique count to each generated file name.
    static int count = 0;
    const std::string baseFileName =
        (getRuntimeDirectory() / fmt::format("{}-{}-{}", applicationName.data(), baseInfo.pid, ++count)).string();

    // Write the crash dump data to a file using the .nv-gpudmp extension
    // registered with Nsight Graphics.
    const std::string crashDumpFileName = baseFileName + ".nv-gpudmp";
    std::ofstream dumpFile(crashDumpFileName, std::ios::out | std::ios::binary);
    if (dumpFile)
    {
        dumpFile.write(reinterpret_cast<const char*>(pGpuCrashDump), gpuCrashDumpSize);
        dumpFile.close();
    }

    // Decode the crash dump to a JSON string.
    // Step 1: Generate the JSON and get the size.
    uint32_t jsonSize = 0;
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GenerateJSON(
        decoder,
        GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO,
        GFSDK_Aftermath_GpuCrashDumpFormatterFlags_NONE,
        shaderDebugInfoLookupCallback,
        shaderLookupCallback,
        shaderSourceDebugInfoLookupCallback,
        nullptr,
        &jsonSize
    ));
    // Step 2: Allocate a buffer and fetch the generated JSON.
    std::vector<char> json(jsonSize);
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetJSON(decoder, uint32_t(json.size()), json.data()));

    // Write the crash dump data as JSON to a file.
    const std::string jsonFileName = crashDumpFileName + ".json";
    std::ofstream jsonFile(jsonFileName, std::ios::out | std::ios::binary);
    if (jsonFile)
    {
        // Write the JSON to the file (excluding string termination)
        jsonFile.write(json.data(), json.size() - 1);
        jsonFile.close();
    }

    // Destroy the GPU crash dump decoder object.
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_DestroyDecoder(decoder));
}

// Helper for writing shader debug information to a file
static void writeShaderDebugInformationToFile(
    GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier,
    const void* pShaderDebugInfo,
    const uint32_t shaderDebugInfoSize
)
{
    // Create a unique file name.
    const std::string path = fmt::format("shader-{:x}{:x}.nvdbg", identifier.id[0], identifier.id[1]);

    std::ofstream f(path, std::ios::out | std::ios::binary);
    if (f)
    {
        f.write((const char*)pShaderDebugInfo, shaderDebugInfoSize);
    }
}

// Handler for GPU crash dump callbacks from Nsight Aftermath
static void gpuCrashDumpCallback(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize, void* pUserData)
{
    std::lock_guard<std::mutex> lock(sMutex);

    // Write to file for later in-depth analysis with Nsight Graphics.
    writeGpuCrashDumpToFile(pGpuCrashDump, gpuCrashDumpSize);
}

// Handler for shader debug information callbacks
static void shaderDebugInfoCallback(const void* pShaderDebugInfo, const uint32_t shaderDebugInfoSize, void* pUserData)
{
    std::lock_guard<std::mutex> lock(sMutex);

    // Get shader debug information identifier
    GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier = {};
    AFTERMATH_CHECK_ERROR(
        GFSDK_Aftermath_GetShaderDebugInfoIdentifier(GFSDK_Aftermath_Version_API, pShaderDebugInfo, shaderDebugInfoSize, &identifier)
    );

    // Store information for decoding of GPU crash dumps with shader address mapping
    // from within the application.
    std::vector<uint8_t> data((uint8_t*)pShaderDebugInfo, (uint8_t*)pShaderDebugInfo + shaderDebugInfoSize);
    sShaderDebugInfo[identifier].swap(data);

    // Write to file for later in-depth analysis of crash dumps with Nsight Graphics
    writeShaderDebugInformationToFile(identifier, pShaderDebugInfo, shaderDebugInfoSize);
}

// Handler for GPU crash dump description callbacks
static void crashDumpDescriptionCallback(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription, void* pUserData)
{
    addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, "Falcor");
    addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion, getLongVersionString().c_str());
    // addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_UserDefined, "user defined string 1");
    // addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_UserDefined + 1, "used defined string 2");
}

// Handler for app-managed marker resolve callbacks
static void resolveMarkerCallback(const void* pMarker, void* pUserData, void** resolvedMarkerData, uint32_t* markerSize) {}

// Handler for shader source debug info lookup callbacks.
// This is used by the JSON decoder for mapping shader instruction addresses to
// HLSL source lines, if the shaders used by the application were compiled with
// separate debug info data files.
static void shaderDebugInfoLookupCallback(
    const GFSDK_Aftermath_ShaderDebugInfoIdentifier* pIdentifier,
    PFN_GFSDK_Aftermath_SetData setShaderDebugInfo,
    void* pUserData
)
{
    // Search the list of shader debug information blobs received earlier.
    auto it = sShaderDebugInfo.find(*pIdentifier);
    if (it == sShaderDebugInfo.end())
    {
        // Early exit, nothing found. No need to call setShaderDebugInfo.
        return;
    }

    // Let the GPU crash dump decoder know about the shader debug information
    // that was found.
    setShaderDebugInfo(it->second.data(), uint32_t(it->second.size()));
}

// Handler for shader lookup callbacks.
// This is used by the JSON decoder for mapping shader instruction
// addresses to DXIL lines or HLSL source lines.
// NOTE: If the application loads stripped shader binaries (-Qstrip_debug),
// Aftermath will require access to both the stripped and the not stripped
// shader binaries.
static void shaderLookupCallback(
    const GFSDK_Aftermath_ShaderBinaryHash* pShaderHash,
    PFN_GFSDK_Aftermath_SetData setShaderBinary,
    void* pUserData
)
{
    // Find shader binary data for the shader hash in the shader database.
    std::vector<uint8_t> shaderBinary;
    // if (!m_shaderDatabase.FindShaderBinary(shaderHash, shaderBinary))
    {
        // Early exit, nothing found. No need to call setShaderBinary.
        return;
    }

    // Let the GPU crash dump decoder know about the shader data
    // that was found.
    setShaderBinary(shaderBinary.data(), uint32_t(shaderBinary.size()));
}

// Handler for shader source debug info lookup callbacks.
// This is used by the JSON decoder for mapping shader instruction addresses to
// HLSL source lines, if the shaders used by the application were compiled with
// separate debug info data files.
static void shaderSourceDebugInfoLookupCallback(
    const GFSDK_Aftermath_ShaderDebugName* pShaderDebugName,
    PFN_GFSDK_Aftermath_SetData setShaderBinary,
    void* pUserData
)
{
    // Find source debug info for the shader DebugName in the shader database.
    std::vector<uint8_t> sourceDebugInfo;
    // if (!m_shaderDatabase.FindSourceShaderDebugData(shaderDebugName, sourceDebugInfo))
    {
        // Early exit, nothing found. No need to call setShaderBinary.
        return;
    }

    // Let the GPU crash dump decoder know about the shader debug data that was
    // found.
    setShaderBinary(sourceDebugInfo.data(), uint32_t(sourceDebugInfo.size()));
}

void enableAftermath()
{
    std::lock_guard<std::mutex> lock(sMutex);

    if (sInitialized)
        return;

    // Enable GPU crash dumps and set up the callbacks for crash dump notifications,
    // shader debug information notifications, and providing additional crash
    // dump description data.Only the crash dump callback is mandatory. The other two
    // callbacks are optional and can be omitted, by passing nullptr, if the corresponding
    // functionality is not used.
    // The DeferDebugInfoCallbacks flag enables caching of shader debug information data
    // in memory. If the flag is set, ShaderDebugInfoCallback will be called only
    // in the event of a crash, right before GpuCrashDumpCallback. If the flag is not set,
    // ShaderDebugInfoCallback will be called for every shader that is compiled.
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_EnableGpuCrashDumps(
        // API version
        GFSDK_Aftermath_Version_API,
        // Device flags
        GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_DX | GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
        // Let the Nsight Aftermath library cache shader debug information
        GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks,
        // Callbacks
        gpuCrashDumpCallback,
        shaderDebugInfoCallback,
        crashDumpDescriptionCallback,
        // Do not resolve markers for now (they are embedded with string data)
        nullptr /* resolveMarkerCallback */,
        // User data
        nullptr
    ));

    sInitialized = true;
}

void disableAftermath()
{
    std::lock_guard<std::mutex> lock(sMutex);

    if (sInitialized)
    {
        GFSDK_Aftermath_DisableGpuCrashDumps();
        sInitialized = false;
    }
}

bool waitForAftermathDumps(int timeout)
{
    if (!sInitialized)
        return true;

    // DXGI_ERROR error notification is asynchronous to the NVIDIA display
    // driver's GPU crash handling. Give the Nsight Aftermath GPU crash dump
    // thread some time to do its work before terminating the process.
    auto tdrTerminationTimeout = std::chrono::seconds(timeout);
    auto tStart = std::chrono::steady_clock::now();
    auto tElapsed = std::chrono::milliseconds::zero();

    GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetCrashDumpStatus(&status));

    while (status != GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed && status != GFSDK_Aftermath_CrashDump_Status_Finished &&
           tElapsed < tdrTerminationTimeout)
    {
        // Sleep 50ms and poll the status again until timeout or Aftermath finished processing the crash dump.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetCrashDumpStatus(&status));

        auto tEnd = std::chrono::steady_clock::now();
        tElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    }

    return status == GFSDK_Aftermath_CrashDump_Status_Finished;
}

AftermathContext::AftermathContext(Device* pDevice) : mpDevice(pDevice) {}

AftermathContext::~AftermathContext() {}

bool AftermathContext::initialize(AftermathFlags flags)
{
    if (mInitialized)
        return true;

    flags |= AftermathFlags::EnableMarkers;
    flags |= AftermathFlags::EnableResourceTracking;
    flags |= AftermathFlags::CallStackCapturing;
    // flags |= AftermathFlags::GenerateShaderDebugInfo;
    // flags |= AftermathFlags::EnableShaderErrorReporting;

    switch (mpDevice->getType())
    {
#if FALCOR_HAS_D3D12
    case Device::Type::D3D12:
    {
        ID3D12Device* pD3D12Device = mpDevice->getNativeHandle(0).as<ID3D12Device*>();
        GFSDK_Aftermath_Result result = GFSDK_Aftermath_DX12_Initialize(GFSDK_Aftermath_Version_API, flags, pD3D12Device);
        if (!GFSDK_Aftermath_SUCCEED(result))
        {
            logWarning("Aftermath failed to initialize on D3D12 device: {}", getResultString(result));
            return false;
        }
    }
    break;
#endif
#if FALCOR_HAS_VULKAN
    case Device::Type::Vulkan:
        logWarning("Aftermath on Vulkan only supports basic GPU crash dumps.");
        return false;
#endif
    default:
        logWarning("Aftermath is not supported on this device.");
        return false;
    }

    mInitialized = true;
    return true;
}

void AftermathContext::addMarker(const LowLevelContextData* pLowLevelContextData, std::string_view name)
{
    if (!mInitialized)
        return;

#if FALCOR_HAS_D3D12
    if (mpDevice->getType() == Device::Type::D3D12)
    {
        auto& context = reinterpret_cast<GFSDK_Aftermath_ContextHandle&>(mContextHandle);
        ID3D12GraphicsCommandList* pCommandList = pLowLevelContextData->getCommandBufferNativeHandle().as<ID3D12GraphicsCommandList*>();
        GFSDK_Aftermath_Result result;

        if (pCommandList != mpLastCommandList)
        {
            // TODO should we call
            // GFSDK_Aftermath_ReleaseContextHandle(context);
            mpLastCommandList = pCommandList;
            result = GFSDK_Aftermath_DX12_CreateContextHandle(pCommandList, &context);
            if (!GFSDK_Aftermath_SUCCEED(result))
            {
                logWarning("Aftermath failed to create context handle: {}", getResultString(result));
                mInitialized = false;
                return;
            }
        }

        result = GFSDK_Aftermath_SetEventMarker(context, name.data(), name.size());
        if (!GFSDK_Aftermath_SUCCEED(result))
        {
            logWarning("Aftermath failed to set event marker: {}", getResultString(result));
            mInitialized = false;
            return;
        }
    }
#endif
}

} // namespace Falcor

#endif // FALCOR_HAS_AFTERMATH

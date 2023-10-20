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
#include "MonitorInfo.h"
#include "Utils/StringUtils.h"

#if FALCOR_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <setupapi.h>
#include <cfgmgr32.h>
#elif FALCOR_LINUX
#include "Core/GLFW.h"
#endif

namespace Falcor
{
#if FALCOR_WINDOWS
// With some inspiration from:
// http://ofekshilon.com/2011/11/13/reading-monitor-physical-dimensions-or-getting-the-edid-the-right-way/
// http://ofekshilon.com/2014/06/19/reading-specific-monitor-dimensions/
// Assumes hDevRegKey is valid
bool GetMonitorSizeFromEDID(const HKEY hDevRegKey, short& WidthMm, short& HeightMm);
bool GetSizeForDevID(const std::wstring& TargetDevID, short& WidthMm, short& HeightMm);

const GUID GUID_CLASS_MONITOR = {0x4d36e96e, 0xe325, 0x11ce, 0xbf, 0xc1, 0x08, 0x00, 0x2b, 0xe1, 0x03, 0x18};

#define NAME_SIZE 128

// Assumes hDevRegKey is valid
bool GetMonitorSizeFromEDID(const HKEY hDevRegKey, short& WidthMm, short& HeightMm)
{
    DWORD dwType, AcutalValueNameLength = NAME_SIZE;
    TCHAR valueName[NAME_SIZE];

    BYTE EDIDdata[1024];
    DWORD edidsize = sizeof(EDIDdata);

    for (LONG i = 0, retValue = ERROR_SUCCESS; retValue != ERROR_NO_MORE_ITEMS; ++i)
    {
        retValue = RegEnumValue(
            hDevRegKey,
            i,
            &valueName[0],
            &AcutalValueNameLength,
            NULL,
            &dwType,
            EDIDdata, // buffer
            &edidsize // buffer size
        );

        if (retValue != ERROR_SUCCESS || 0 != wcscmp(valueName, L"EDID"))
            continue;

        WidthMm = ((EDIDdata[68] & 0xF0) << 4) + EDIDdata[66];
        HeightMm = ((EDIDdata[68] & 0x0F) << 8) + EDIDdata[67];

        return true; // valid EDID found
    }

    return false; // EDID not found
}

bool GetSizeForDevID(const std::wstring& TargetDevID, short& WidthMm, short& HeightMm)
{
    HDEVINFO devInfo = SetupDiGetClassDevsEx(
        &GUID_CLASS_MONITOR, // class GUID
        NULL,                // enumerator
        NULL,                // HWND
        DIGCF_PRESENT,       // Flags //DIGCF_ALLCLASSES|
        NULL,                // device info, create a new one.
        NULL,                // machine name, local machine
        NULL
    ); // reserved

    if (NULL == devInfo)
        return false;

    bool bRes = false;

    for (ULONG i = 0; ERROR_NO_MORE_ITEMS != GetLastError(); ++i)
    {
        SP_DEVINFO_DATA devInfoData;
        memset(&devInfoData, 0, sizeof(devInfoData));
        devInfoData.cbSize = sizeof(devInfoData);

        if (SetupDiEnumDeviceInfo(devInfo, i, &devInfoData))
        {
            TCHAR Instance[MAX_DEVICE_ID_LEN];
            SetupDiGetDeviceInstanceId(devInfo, &devInfoData, Instance, MAX_DEVICE_ID_LEN, NULL);

            std::wstring sInstance(Instance);

            if (sInstance.find(TargetDevID) == std::wstring::npos)
                continue;

            HKEY hDevRegKey = SetupDiOpenDevRegKey(devInfo, &devInfoData, DICS_FLAG_GLOBAL, 0, DIREG_DEV, KEY_READ);

            if (!hDevRegKey || (hDevRegKey == INVALID_HANDLE_VALUE))
                continue;

            bRes = GetMonitorSizeFromEDID(hDevRegKey, WidthMm, HeightMm);

            RegCloseKey(hDevRegKey);
        }
    }
    SetupDiDestroyDeviceInfoList(devInfo);
    return bRes;
}

BOOL DisplayDeviceFromHMonitor(HMONITOR hMonitor, DISPLAY_DEVICE& ddMonOut)
{
    MONITORINFOEX mi;
    mi.cbSize = sizeof(MONITORINFOEX);
    GetMonitorInfo(hMonitor, &mi);

    DISPLAY_DEVICE dd;
    dd.cb = sizeof(dd);
    DWORD devIdx = 0; // device index

    bool bFoundDevice = false;
    while (EnumDisplayDevices(0, devIdx, &dd, 0))
    {
        devIdx++;
        if (0 != wcscmp(dd.DeviceName, mi.szDevice))
            continue;

        DISPLAY_DEVICE ddMon;
        ZeroMemory(&ddMon, sizeof(ddMon));
        ddMon.cb = sizeof(ddMon);
        DWORD MonIdx = 0;

        while (EnumDisplayDevices(dd.DeviceName, MonIdx, &ddMon, 0))
        {
            MonIdx++;

            ddMonOut = ddMon;
            return TRUE;

            ZeroMemory(&ddMon, sizeof(ddMon));
            ddMon.cb = sizeof(ddMon);
        }

        ZeroMemory(&dd, sizeof(dd));
        dd.cb = sizeof(dd);
    }

    return FALSE;
}

std::vector<MonitorInfo::MonitorDesc> internalDescs;

BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData)
{
    MONITORINFO info;
    info.cbSize = sizeof(info);
    if (GetMonitorInfo(hMonitor, &info))
    {
        DISPLAY_DEVICE dev;
        if (!DisplayDeviceFromHMonitor(hMonitor, dev))
            return TRUE;

        std::wstring DeviceID(dev.DeviceID);
        DeviceID = DeviceID.substr(8, DeviceID.find(L'\\', 9) - 8);

        short WidthMm, HeightMm;

        GetSizeForDevID(DeviceID, WidthMm, HeightMm);

        float wInch = float(WidthMm) / 25.4f;
        float hInch = float(HeightMm) / 25.4f;

        MonitorInfo::MonitorDesc desc;
        desc.identifier = wstring_2_string(DeviceID);
        desc.resolution = uint2(std::abs(info.rcMonitor.left - info.rcMonitor.right), std::abs(info.rcMonitor.top - info.rcMonitor.bottom));

        desc.physicalSize = float2(wInch, hInch);

        auto vPpi = float2(desc.resolution) / desc.physicalSize;
        desc.ppi = (vPpi.x + vPpi.y) * 0.5f;

        desc.isPrimary = (info.dwFlags & MONITORINFOF_PRIMARY);

        internalDescs.push_back(desc);
    }
    return TRUE; // continue enumerating
}

std::vector<MonitorInfo::MonitorDesc> MonitorInfo::getMonitorDescs()
{
    internalDescs.clear();
    EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, 0);
    return internalDescs;
}
#endif // FALCOR_WINDOWS

#if FALCOR_LINUX
std::vector<MonitorInfo::MonitorDesc> MonitorInfo::getMonitorDescs()
{
    std::vector<MonitorDesc> descs;

    int monitorCount;
    GLFWmonitor** monitors = glfwGetMonitors(&monitorCount);

    for (int i = 0; i < monitorCount; ++i)
    {
        auto monitor = monitors[i];

        MonitorDesc desc;
        desc.identifier = glfwGetMonitorName(monitor);

        // Determine monitor resolution.
        desc.resolution = uint2(0);
        int modeCount;
        const GLFWvidmode* modes = glfwGetVideoModes(monitor, &modeCount);
        for (int j = 0; j < modeCount; ++j)
        {
            const auto& mode = modes[j];
            if (mode.width * mode.height > desc.resolution.x * desc.resolution.y)
            {
                desc.resolution = uint2(mode.width, mode.height);
            }
        }

        // Get physical size in mm and convert to inches.
        int width, height;
        glfwGetMonitorPhysicalSize(monitor, &width, &height);
        desc.physicalSize = float2(width, height) / 25.4f;

        // Compute pixel density.
        auto vPpi = float2(desc.resolution) / desc.physicalSize;
        desc.ppi = (vPpi.x + vPpi.y) * 0.5f;

        desc.isPrimary = monitor == glfwGetPrimaryMonitor();

        descs.push_back(desc);
    }

    return descs;
}
#endif // FALCOR_LINUX

void MonitorInfo::displayMonitorInfo()
{
    for (const auto& desc : getMonitorDescs())
    {
        fmt::print(
            "{}{}: {} x {} pix, {:0.1f} x {:0.1f} in, {:0.2f} ppi\n",
            desc.identifier,
            desc.isPrimary ? " (Primary) " : " ",
            desc.resolution.x,
            desc.resolution.y,
            desc.physicalSize.x,
            desc.physicalSize.y,
            desc.ppi
        );
    }
}

} // namespace Falcor

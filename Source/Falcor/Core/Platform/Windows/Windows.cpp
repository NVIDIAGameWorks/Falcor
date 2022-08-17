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
#include "Core/Platform/OS.h"
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Utils/Logger.h"
#include "Utils/StringUtils.h"

#include <commctrl.h>
#include <commdlg.h>
#include <comutil.h>
#include <psapi.h>
#include <shellscalingapi.h>
#include <ShlObj_core.h>
#include <winioctl.h>

#include <string>
#include <vector>
#include <filesystem>

#define os_call(a) {auto hr_ = a; if(FAILED(hr_)) { reportError(#a); }}

// Always run in Optimus mode on laptops
extern "C"
{
    _declspec(dllexport) DWORD NvOptimusEnablement = 1;
}

namespace Falcor
{
    extern std::string gMsgBoxTitle;

    static HWND gMainWindowHandle;

    void setMainWindowHandle(HWND windowHandle)
    {
        gMainWindowHandle = windowHandle;
    }

    MsgBoxButton msgBox(const std::string& msg, MsgBoxType type, MsgBoxIcon icon)
    {
        const MsgBoxCustomButton buttonOk{uint32_t(MsgBoxButton::Ok), "Ok"};
        const MsgBoxCustomButton buttonRetry{uint32_t(MsgBoxButton::Retry), "Retry"};
        const MsgBoxCustomButton buttonCancel{uint32_t(MsgBoxButton::Cancel), "Cancel"};
        const MsgBoxCustomButton buttonAbort{uint32_t(MsgBoxButton::Abort), "Abort"};
        const MsgBoxCustomButton buttonIgnore{uint32_t(MsgBoxButton::Ignore), "Ignore"};
        const MsgBoxCustomButton buttonYes{uint32_t(MsgBoxButton::Yes), "Yes"};
        const MsgBoxCustomButton buttonNo{uint32_t(MsgBoxButton::No), "No"};

        std::vector<MsgBoxCustomButton> buttons;
        switch (type)
        {
        case MsgBoxType::Ok: buttons = { buttonOk }; break;
        case MsgBoxType::OkCancel: buttons = { buttonOk, buttonCancel }; break;
        case MsgBoxType::RetryCancel: buttons = { buttonRetry, buttonCancel }; break;
        case MsgBoxType::AbortRetryIgnore: buttons = { buttonAbort, buttonRetry, buttonIgnore }; break;
        case MsgBoxType::YesNo: buttons = { buttonYes, buttonNo }; break;
        default: FALCOR_UNREACHABLE();
        }

        return (MsgBoxButton)msgBox(msg, buttons, icon);
    }

    uint32_t msgBox(const std::string& msg, std::vector<MsgBoxCustomButton> buttons, MsgBoxIcon icon, uint32_t defaultButtonId)
    {
        FALCOR_ASSERT(buttons.size() > 0);

        // Helper to convert a string to a wide string
        auto toWideString = [](const std::string& str) { std::wstring wstr(str.begin(), str.end()); return wstr; };

        // TASKDIALOGCONFIG has a flag TDF_SIZE_TO_CONTENT to automatically size the dialog to fit the content.
        // Unfortunately this flag leads to the content being modified with ellipsis when lines are too long.
        // For that reason we go through the excercise of determining the width of the longest text line manually.

        // Compute the width of the given text in pixels using the default message font.
        auto computeTextWidth = [&toWideString](const std::string& text)
        {
            LONG textWidth = 0;

            // Query windows for common metrics.
            HWND hwnd = gMainWindowHandle;
            UINT dpi = GetDpiForSystem();
            NONCLIENTMETRICS metrics;
            metrics.cbSize = sizeof(metrics);
            if (!SystemParametersInfoForDpi(SPI_GETNONCLIENTMETRICS, sizeof(metrics), &metrics, 0, dpi)) return textWidth;

            // Setup DC with message font.
            HDC hdc = GetDC(hwnd);
            HFONT font = CreateFontIndirect(&metrics.lfMessageFont);
            HGDIOBJ oldFont = SelectObject(hdc, font);

            // Compute text width of longest line.
            for (const auto& line : splitString(text, "\n"))
            {
                auto wideLine = toWideString(line);
                SIZE textSize;
                if (GetTextExtentPoint32(hdc, wideLine.c_str(), (int)line.size(), &textSize)) textWidth = std::max(textWidth, textSize.cx);
            }

            // Restore DC.
            SelectObject(hdc, oldFont);
            ReleaseDC(hwnd, hdc);
            DeleteObject(font);

            return textWidth;
        };

        auto wideTitle = toWideString(gMsgBoxTitle);
        auto wideMsg = toWideString(msg);

        // We need to map button ids to a range beyond IDCANCEL, which is used to indicate when the dialog was canceled/closed.
        auto mapButtonId = [](uint32_t id) -> int { return id + IDCANCEL + 1; };

        // Highlight the first button by default.
        int defaultId = mapButtonId(buttons.front().id);

        // Set up button configs
        std::vector<std::wstring> wideButtonTitles(buttons.size());
        std::vector<TASKDIALOG_BUTTON> buttonConfigs(buttons.size());
        for (size_t i = 0; i < buttons.size(); ++i)
        {
            wideButtonTitles[i] = toWideString(buttons[i].title);
            buttonConfigs[i].pszButtonText = wideButtonTitles[i].c_str();
            buttonConfigs[i].nButtonID = mapButtonId(buttons[i].id);
            if (buttons[i].id == defaultButtonId) defaultId = buttonConfigs[i].nButtonID;
        }

        // The width of the dialog is expressed in "dialog units", not pixels.
        LONG horizontalBaseUnit = GetDialogBaseUnits() & 0xffff;
        UINT dialogWidth = (computeTextWidth(msg) * 4) / horizontalBaseUnit;

        // We add a margin of 16 units (and another 32 if using an icon) to make sure the text fits the dialog size.
        dialogWidth += 16 + (icon != MsgBoxIcon::None ? 32 : 0);

        // Set up dialog config
        TASKDIALOGCONFIG config = {};
        config.cbSize = sizeof(TASKDIALOGCONFIG);
        config.hwndParent = gMainWindowHandle;
        config.dwFlags = TDF_ALLOW_DIALOG_CANCELLATION;
        config.pszWindowTitle = wideTitle.c_str();
        switch (icon)
        {
        case MsgBoxIcon:: None: break;
        case MsgBoxIcon::Info: config.pszMainIcon = TD_INFORMATION_ICON; break;
        case MsgBoxIcon::Warning: config.pszMainIcon = TD_WARNING_ICON; break;
        case MsgBoxIcon::Error: config.pszMainIcon = TD_ERROR_ICON; break;
        }
        config.pszContent = wideMsg.c_str();
        config.cButtons = (UINT)buttonConfigs.size();
        config.pButtons = buttonConfigs.data();
        config.nDefaultButton = defaultId;
        config.cxWidth = dialogWidth;

        // By default return the id of the last button.
        uint32_t result = buttons.back().id;

        // Execute dialog.
        int selectedId;
        if (TaskDialogIndirect(&config, &selectedId, nullptr, nullptr) == S_OK)
        {
            // Map selected id back to the user provided button id.
            for (const auto& button : buttons)
            {
                if (selectedId == mapButtonId(button.id)) result = button.id;
            }
        }

        return result;
    }

    // As defined in ntifs.h
    typedef struct _REPARSE_DATA_BUFFER {
        ULONG  ReparseTag;
        USHORT ReparseDataLength;
        USHORT Reserved;
        union {
            struct {
                USHORT SubstituteNameOffset;
                USHORT SubstituteNameLength;
                USHORT PrintNameOffset;
                USHORT PrintNameLength;
                ULONG Flags;
                WCHAR PathBuffer[1];
            } SymbolicLinkReparseBuffer;
            struct {
                USHORT SubstituteNameOffset;
                USHORT SubstituteNameLength;
                USHORT PrintNameOffset;
                USHORT PrintNameLength;
                WCHAR PathBuffer[1];
            } MountPointReparseBuffer;
            struct {
                UCHAR  DataBuffer[1];
            } GenericReparseBuffer;
        } DUMMYUNIONNAME;
    } REPARSE_DATA_BUFFER;

    // As defined in wdm.h
    #define FILE_DIRECTORY_FILE                     0x00000001
    #define FILE_WRITE_THROUGH                      0x00000002
    #define FILE_SEQUENTIAL_ONLY                    0x00000004
    #define FILE_NO_INTERMEDIATE_BUFFERING          0x00000008

    #define FILE_SYNCHRONOUS_IO_ALERT               0x00000010
    #define FILE_SYNCHRONOUS_IO_NONALERT            0x00000020
    #define FILE_NON_DIRECTORY_FILE                 0x00000040
    #define FILE_CREATE_TREE_CONNECTION             0x00000080

    #define FILE_COMPLETE_IF_OPLOCKED               0x00000100
    #define FILE_NO_EA_KNOWLEDGE                    0x00000200
    #define FILE_OPEN_REMOTE_INSTANCE               0x00000400
    #define FILE_RANDOM_ACCESS                      0x00000800

    #define FILE_DELETE_ON_CLOSE                    0x00001000
    #define FILE_OPEN_BY_FILE_ID                    0x00002000
    #define FILE_OPEN_FOR_BACKUP_INTENT             0x00004000
    #define FILE_NO_COMPRESSION                     0x00008000

    bool createJunction(const std::filesystem::path& link, const std::filesystem::path& target)
    {
        // Get absolute target.
        WCHAR absoluteTarget[MAX_PATH];
        if (GetFullPathNameW(target.native().c_str(), MAX_PATH, absoluteTarget, NULL) == 0) return false;

        // Create junction as normal directory.
        if (!CreateDirectoryW(link.native().c_str(), NULL)) return false;

        // Open file handle to junction directory.
        HANDLE handle = CreateFileW(link.native().c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, NULL);
        if (handle == INVALID_HANDLE_VALUE) return false;

        // Allocate reparse data.
        char tmp[MAXIMUM_REPARSE_DATA_BUFFER_SIZE] = {0};
        REPARSE_DATA_BUFFER& reparseData = *reinterpret_cast<REPARSE_DATA_BUFFER*>(tmp);

        // Setup reparse data.
        std::wstring substituteName = std::wstring(L"\\??\\") + absoluteTarget;
        std::wstring printName = absoluteTarget;

        reparseData.ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;
        reparseData.Reserved = 0;
        reparseData.MountPointReparseBuffer.SubstituteNameOffset = (USHORT)0;
        reparseData.MountPointReparseBuffer.SubstituteNameLength = (USHORT)(substituteName.length() * sizeof(wchar_t));
        reparseData.MountPointReparseBuffer.PrintNameOffset = (USHORT)((substituteName.length() + 1) * sizeof(wchar_t));
        reparseData.MountPointReparseBuffer.PrintNameLength = (USHORT)(printName.length() * sizeof(wchar_t));
        reparseData.ReparseDataLength = (USHORT)(sizeof(reparseData.MountPointReparseBuffer) + reparseData.MountPointReparseBuffer.PrintNameOffset + reparseData.MountPointReparseBuffer.PrintNameLength);
        if (reparseData.ReparseDataLength > MAXIMUM_REPARSE_DATA_BUFFER_SIZE) return false;

        std::memcpy(&reparseData.MountPointReparseBuffer.PathBuffer[0], substituteName.data(), substituteName.length() * sizeof(wchar_t));
        std::memcpy(&reparseData.MountPointReparseBuffer.PathBuffer[substituteName.length() + 1], printName.data(), printName.length() * sizeof(wchar_t));

        // Set the total size of the data buffer.

        // Use DeviceIoControl to set the reparse point data on the file handle.
        size_t headerSize = FIELD_OFFSET(REPARSE_DATA_BUFFER, MountPointReparseBuffer);
        bool success = DeviceIoControl(handle, FSCTL_SET_REPARSE_POINT, &reparseData, (DWORD)headerSize + reparseData.ReparseDataLength, NULL, 0, NULL, 0);

        // Close file handle.
        CloseHandle(handle);

        return success;
    }

    bool deleteJunction(const std::filesystem::path& link)
    {
        // Open file handle to junction directory.
        HANDLE handle = CreateFileW(link.native().c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, NULL);
        if (handle == INVALID_HANDLE_VALUE) return false;

        REPARSE_GUID_DATA_BUFFER rgdb = {0};
        rgdb.ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;

        // Delete reparse point data.
        bool success = DeviceIoControl(handle, FSCTL_DELETE_REPARSE_POINT, &rgdb, REPARSE_GUID_DATA_BUFFER_HEADER_SIZE, NULL, 0, NULL, 0);

        // Close file handle.
        CloseHandle(handle);

        return success;
    }

    const std::filesystem::path& getExecutablePath()
    {
        static std::filesystem::path path;
        if (path.empty())
        {
            CHAR pathStr[1024];
            if (GetModuleFileNameA(nullptr, pathStr, ARRAYSIZE(pathStr)) == 0)
            {
                throw RuntimeError("Failed to get the executable path.");
            }
            path = pathStr;
        }
        return path;
    }

    const std::filesystem::path& getAppDataDirectory()
    {
        static std::filesystem::path path;
        if (path.empty())
        {
            PWSTR pathStr;
            if (!SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, NULL, &pathStr)))
            {
                throw RuntimeError("Failed to get the application data directory.");
            }
            path = pathStr;
        }
        return path;
    }

    bool getEnvironmentVariable(const std::string& varName, std::string& value)
    {
        static char buff[4096];
        DWORD numChar = GetEnvironmentVariableA(varName.c_str(), buff, (DWORD)std::size(buff)); //what is the best way to deal with wchar ?
        FALCOR_ASSERT(numChar < (DWORD)std::size(buff));
        if (numChar == 0)
        {
            return false;
        }
        value = std::string(buff);
        return true;
    }

    template<bool open>
    static std::string getExtensionsFilterString(const FileDialogFilterVec& filters)
    {
        std::string s;
        std::string d;
        bool appendForOpen = open && filters.size() > 1;
        if (appendForOpen) s.append(1, 0);

        for (size_t i = 0 ; i < filters.size() ; i++)
        {
            const auto& f = filters[i];
            if (appendForOpen)
            {
                bool last = i == (filters.size() - 1);
                std::string e = "*." + f.ext;
                if (last == false) e += ';';
                d += e;
                s += e;
            }
            else
            {
                s += f.desc.empty() ? f.ext + " files" : f.desc + " (*." + f.ext + ')';
                s.append(1, 0);
                s += "*." + f.ext + ';';
                s.append(1, 0);
            }
        }
        if (appendForOpen) s = "Supported Formats (" + d + ')' + s;
        s.append(appendForOpen ? 2 : 1, 0);
        return s;
    };

    struct FilterSpec
    {
        FilterSpec(const FileDialogFilterVec& filters, bool forOpen)
        {
            size_t size = forOpen ? filters.size() + 1 : filters.size();
            comDlg.reserve(size);
            descs.reserve(size);
            ext.reserve(size);

            if (forOpen) comDlg.push_back({});
            std::wstring all;
            for(const auto& f : filters)
            {
                descs.push_back(string_2_wstring(f.desc));
                ext.push_back(L"*." + string_2_wstring(f.ext));
                comDlg.push_back({ descs.back().c_str(), ext.back().c_str() });
                all += ext.back() + L";";
            }

            if (forOpen)
            {
                descs.push_back(L"Supported Formats");
                ext.push_back(all);
                comDlg[0] = { descs.back().c_str(), ext.back().c_str() };
            }
        }

        size_t size() const { return comDlg.size(); }
        const COMDLG_FILTERSPEC* data() const { return comDlg.data(); }
    private:
        std::vector<COMDLG_FILTERSPEC> comDlg;
        std::vector<std::wstring> descs;
        std::vector<std::wstring> ext;
    };

    template<typename DialogType>
    static bool fileDialogCommon(const FileDialogFilterVec& filters, std::filesystem::path& path, DWORD options, const CLSID clsid)
    {
        FilterSpec fs(filters, typeid(DialogType) == typeid(IFileOpenDialog));

        DialogType* pDialog;
        os_call(CoCreateInstance(clsid, NULL, CLSCTX_ALL, IID_PPV_ARGS(&pDialog)));
        pDialog->SetOptions(options | FOS_FORCEFILESYSTEM);
        pDialog->SetFileTypes((uint32_t)fs.size(), fs.data());
        pDialog->SetDefaultExtension(fs.data()->pszSpec);

        if (pDialog->Show(nullptr) == S_OK)
        {
            IShellItem* pItem;
            if (pDialog->GetResult(&pItem) == S_OK)
            {
                PWSTR pathStr;
                if (pItem->GetDisplayName(SIGDN_FILESYSPATH, &pathStr) == S_OK)
                {
                    path = pathStr;
                    // filename = wstring_2_string(std::wstring(path)); // TODO
                    CoTaskMemFree(pathStr);
                    return true;
                }
            }
        }

        return false;
    }

    bool saveFileDialog(const FileDialogFilterVec& filters, std::filesystem::path& path)
    {
        return fileDialogCommon<IFileSaveDialog>(filters, path, FOS_OVERWRITEPROMPT, CLSID_FileSaveDialog);
    }

    bool openFileDialog(const FileDialogFilterVec& filters, std::filesystem::path& path)
    {
        return fileDialogCommon<IFileOpenDialog>(filters, path, FOS_FILEMUSTEXIST, CLSID_FileOpenDialog);
    };

    bool chooseFolderDialog(std::filesystem::path& path)
    {
        return fileDialogCommon<IFileOpenDialog>({}, path, FOS_PICKFOLDERS | FOS_PATHMUSTEXIST, CLSID_FileOpenDialog);
    }

    void setWindowIcon(const std::filesystem::path& iconPath, WindowHandle windowHandle)
    {
        std::filesystem::path fullPath;
        if (findFileInDataDirectories(iconPath, fullPath))
        {
            HANDLE hIcon = LoadImageW(GetModuleHandleW(NULL), fullPath.c_str(), IMAGE_ICON, 0, 0, LR_DEFAULTSIZE | LR_LOADFROMFILE);
            HWND hWnd = windowHandle ? windowHandle : GetActiveWindow();
            SendMessage(hWnd, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
        }
        else
        {
            logWarning("Error when loading icon. Can't find the file '{}'.", iconPath);
        }
    }

    int getDisplayDpi()
    {
        ::SetProcessDPIAware();
        HDC screen = GetDC(NULL);
        double hPixelsPerInch = GetDeviceCaps(screen, LOGPIXELSX);
        double vPixelsPerInch = GetDeviceCaps(screen, LOGPIXELSY);
        ::ReleaseDC(NULL, screen);
        return int((hPixelsPerInch + vPixelsPerInch) * 0.5);
    }

    float getDisplayScaleFactor()
    {
        float dpi = (float)getDisplayDpi();
        float scale = dpi / 96.0f;
        return scale;

        ::SetProcessDPIAware();
        DEVICE_SCALE_FACTOR factor;
        if (GetScaleFactorForMonitor(nullptr, &factor) == S_OK)
        {
            switch (factor)
            {
            case SCALE_100_PERCENT: return 1.0f;
            case SCALE_120_PERCENT: return 1.2f;
            case SCALE_125_PERCENT: return 1.25f;
            case SCALE_140_PERCENT: return 1.40f;
            case SCALE_150_PERCENT: return 1.50f;
            case SCALE_160_PERCENT: return 1.60f;
            case SCALE_175_PERCENT: return 1.70f;
            case SCALE_180_PERCENT: return 1.80f;
            case SCALE_200_PERCENT: return 2.00f;
            case SCALE_225_PERCENT: return 2.25f;
            case SCALE_250_PERCENT: return 2.50f;
            case SCALE_300_PERCENT: return 3.00f;
            case SCALE_350_PERCENT: return 3.50f;
            case SCALE_400_PERCENT: return 4.00f;
            case SCALE_450_PERCENT: return 4.50f;
            case SCALE_500_PERCENT: return 4.60f;
            default:
                FALCOR_UNREACHABLE();
                return 1.0f;
            }
        }
        return 1.0f;
    }

    bool isDebuggerPresent()
    {
        return ::IsDebuggerPresent() == TRUE;
    }

    void printToDebugWindow(const std::string& s)
    {
        OutputDebugStringA(s.c_str());
    }

    void debugBreak()
    {
        __debugbreak();
    }

    size_t executeProcess(const std::string& appName, const std::string& commandLineArgs)
    {
        std::string commandLine = appName + ".exe " + commandLineArgs;
        STARTUPINFOA startupInfo{}; PROCESS_INFORMATION processInformation{};
        if (!CreateProcessA(nullptr, (LPSTR)commandLine.c_str(), nullptr, nullptr, TRUE, NORMAL_PRIORITY_CLASS, nullptr, nullptr, &startupInfo, &processInformation))
        {
            throw RuntimeError("Unable to execute process: " + commandLine);
        }

        return reinterpret_cast<size_t>(processInformation.hProcess);
    }

    bool isProcessRunning(size_t processID)
    {
        uint32_t exitCode = 0;
        if (GetExitCodeProcess((HANDLE)processID, (LPDWORD)&exitCode))
        {
            if (exitCode != STILL_ACTIVE)
            {
                return false;
            }
        }

        return true;
    }

    void terminateProcess(size_t processID)
    {
        TerminateProcess((HANDLE)processID, 0);
        CloseHandle((HANDLE)processID);
    }

    static std::unordered_map<std::wstring, std::pair<std::thread, bool> > fileThreads;

    static void checkFileModifiedStatus(const std::filesystem::path& path, const std::function<void()>& callback)
    {
        auto filename = path.filename();
        auto dir = path.parent_path();

        HANDLE hFile = CreateFileW(dir.native().c_str(), GENERIC_READ | FILE_LIST_DIRECTORY,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, NULL);
        FALCOR_ASSERT(hFile != INVALID_HANDLE_VALUE);

        // overlapped struct requires unique event handle to be valid
        OVERLAPPED overlapped{};

        while (true)
        {
            size_t offset = 0;
            uint32_t bytesReturned = 0;
            std::vector<uint32_t> buffer;
            buffer.resize(1024);

            if (!ReadDirectoryChangesW(hFile, buffer.data(), static_cast<uint32_t>(sizeof(uint32_t) * buffer.size()), FALSE,
                FILE_NOTIFY_CHANGE_LAST_WRITE, 0, &overlapped, nullptr))
            {
                logError("Failed to read directory changes for shared file '{}'. Aborting monitoring.", path);
                CloseHandle(hFile);
            }

            if (!GetOverlappedResult(hFile, &overlapped, (LPDWORD)&bytesReturned, true))
            {
                logError("Failed to read directory changes for shared file '{}'. Aborting monitoring.", path);
                CloseHandle(hFile);
            }

            // don't check for another overlapped result if main thread is closed
            if (!fileThreads.at(path.native()).second)
            {
                break;
            }

            if (!bytesReturned) continue;

            while (offset < buffer.size())
            {
                _FILE_NOTIFY_INFORMATION* pNotifyInformation = reinterpret_cast<_FILE_NOTIFY_INFORMATION*>(buffer.data());
                std::wstring currentFilename(pNotifyInformation->FileName, pNotifyInformation->FileName + pNotifyInformation->FileNameLength);

                if (currentFilename == filename && pNotifyInformation->Action == FILE_ACTION_MODIFIED)
                {
                    callback();
                    break;
                }

                if (!pNotifyInformation->NextEntryOffset) break;
                offset += pNotifyInformation->NextEntryOffset;
            }
        }

        CloseHandle(hFile);
    }

    void monitorFileUpdates(const std::filesystem::path& path, const std::function<void()>& callback)
    {
        const auto& fileThreadsIt = fileThreads.find(path.native());

        // only have one thread waiting on file write
        if(fileThreadsIt != fileThreads.end())
        {
            if (fileThreadsIt->second.first.joinable())
            {
                fileThreadsIt->second.first.join();
            }
        }

        auto pathStr = path.native();
        fileThreads[pathStr].first = std::thread(checkFileModifiedStatus, path, callback);
        fileThreads[pathStr].second = true;
    }

    void closeSharedFile(const std::filesystem::path& path)
    {
        const auto& fileThreadsIt = fileThreads.find(path.native());

        // only have one thread waiting on file write
        if (fileThreadsIt != fileThreads.end())
        {
            fileThreadsIt->second.second = false;

            fileThreadsIt->second.first.detach();
        }
    }

    std::thread::native_handle_type getCurrentThread()
    {
        return ::GetCurrentThread();
    }

    void setThreadAffinity(std::thread::native_handle_type thread, uint32_t affinityMask)
    {
        ::SetThreadAffinityMask(thread, affinityMask);
        if (DWORD dwError = GetLastError() != 0)
        {
            LPVOID lpMsgBuf;
            FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, dwError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, NULL);
            std::wstring err((LPTSTR)lpMsgBuf);
            logWarning("setThreadAffinity failed with error: {}", wstring_2_string(err));
            LocalFree(lpMsgBuf);
        }
    }

    void setThreadPriority(std::thread::native_handle_type thread, ThreadPriorityType priority)
    {
        if (priority >= ThreadPriorityType::Lowest)
            ::SetThreadPriority(thread, THREAD_BASE_PRIORITY_MIN + (int32_t)priority);
        else if (priority == ThreadPriorityType::BackgroundBegin)
            ::SetThreadPriority(thread, THREAD_MODE_BACKGROUND_BEGIN);
        else if (priority == ThreadPriorityType::BackgroundEnd)
            ::SetThreadPriority(thread, THREAD_MODE_BACKGROUND_END);
        else
            FALCOR_UNREACHABLE();

        if (DWORD dwError = GetLastError() != 0)
        {
            LPVOID lpMsgBuf;
            FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, dwError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, NULL);
            std::wstring err((LPTSTR)lpMsgBuf);
            logWarning("setThreadPriority failed with error: {}", wstring_2_string(err));
            LocalFree(lpMsgBuf);
        }
    }

    time_t getFileModifiedTime(const std::filesystem::path& path)
    {
        struct stat s;
        if (stat(path.string().c_str(), &s) != 0)
        {
            throw RuntimeError("Can't get file time for '{}'.", path);
        }

        return s.st_mtime;
    }

    uint64_t getTotalVirtualMemory()
    {
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        DWORDLONG totalVirtualMem = memInfo.ullTotalPageFile;

        return totalVirtualMem;
    }

    uint64_t getUsedVirtualMemory()
    {
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        DWORDLONG totalVirtualMem = memInfo.ullTotalPageFile;
        DWORDLONG virtualMemUsed = memInfo.ullTotalPageFile - memInfo.ullAvailPageFile;

        return virtualMemUsed;
    }

    uint64_t getProcessUsedVirtualMemory()
    {
        PROCESS_MEMORY_COUNTERS_EX pmc;
        GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
        SIZE_T virtualMemUsedByMe = pmc.PrivateUsage;

        return virtualMemUsedByMe;
    }

    uint32_t bitScanReverse(uint32_t a)
    {
        unsigned long index;
        auto ret = _BitScanReverse(&index, a);
        return ret ? (uint32_t)index : 0;
    }

    uint32_t bitScanForward(uint32_t a)
    {
        unsigned long index;
        auto ret = _BitScanForward(&index, a);
        return ret ? (uint32_t)index : 0;
    }

    uint32_t popcount(uint32_t a)
    {
        return __popcnt(a);
    }


    SharedLibraryHandle loadSharedLibrary(const std::filesystem::path& path)
    {
        return LoadLibraryW(path.c_str());
    }

    /** Release a shared-library
    */
    void releaseSharedLibrary(SharedLibraryHandle library)
    {
        FreeLibrary(library);
    }

    /** Get a function pointer from a library
    */
    void* getProcAddress(SharedLibraryHandle library, const std::string& funcName)
    {
        return reinterpret_cast<void*>(GetProcAddress(library, funcName.c_str()));
    }

    void postQuitMessage(int32_t exitCode)
    {
        PostQuitMessage(exitCode);
    }

    void OSServices::start()
    {
        os_call(CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE));
    }

    void OSServices::stop()
    {
        CoUninitialize();
    }

    size_t getCurrentRSS()
    {
        PROCESS_MEMORY_COUNTERS memoryCounter;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &memoryCounter, sizeof(PROCESS_MEMORY_COUNTERS)))
            return memoryCounter.WorkingSetSize;
        return 0;
    }

    size_t getPeakRSS()
    {
        PROCESS_MEMORY_COUNTERS memoryCounter;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &memoryCounter, sizeof(PROCESS_MEMORY_COUNTERS)))
            return memoryCounter.PeakWorkingSetSize;
        return 0;
    }
}

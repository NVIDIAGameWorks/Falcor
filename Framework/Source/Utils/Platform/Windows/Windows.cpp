/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/

#include "Framework.h"
#include "Utils/Platform/OS.h"
#include <vector>
#include <stdint.h>
#include "Utils/StringUtils.h"
#include <Shlwapi.h>
#include <shlobj.h>
#include <sys/types.h>
#include "API/Window.h"
#include "psapi.h"
#include "Utils/ThreadPool.h"
#include <future>

// Always run in Optimus mode on laptops
extern "C"
{
    _declspec(dllexport) DWORD NvOptimusEnablement = 1;
}

namespace Falcor
{
    MsgBoxButton msgBox(const std::string& msg, MsgBoxType mbType)
    {
        UINT Type = MB_OK;
        switch (mbType)
        {
        case MsgBoxType::Ok:
            Type = MB_OK;
            break;
        case MsgBoxType::OkCancel:
            Type = MB_OKCANCEL;
            break;
        case  MsgBoxType::RetryCancel:
            Type = MB_RETRYCANCEL;
            break;
        case  MsgBoxType::AbortRetryIgnore:
            Type = MB_ABORTRETRYIGNORE;
            break;
        default:
            should_not_get_here();
            break;
        }

        int value = MessageBoxA(nullptr, msg.c_str(), "Falcor", Type | MB_TOPMOST);
        switch (value)
        {
        case IDOK:
            return MsgBoxButton::Ok;
        case IDCANCEL:
            return MsgBoxButton::Cancel;
        case IDRETRY:
            return MsgBoxButton::Retry;
        case IDABORT:
            return MsgBoxButton::Abort;
        case IDIGNORE:
            return MsgBoxButton::Ignore;
        default:
            should_not_get_here();
            return MsgBoxButton::Cancel;
        }
    }

    bool doesFileExist(const std::string& filename)
    {
        DWORD attr = GetFileAttributesA(filename.c_str());
        return (attr != INVALID_FILE_ATTRIBUTES);
    }

    bool isDirectoryExists(const std::string& filename)
    {
        DWORD attr = GetFileAttributesA(filename.c_str());
        return ((attr != INVALID_FILE_ATTRIBUTES) && (attr & FILE_ATTRIBUTE_DIRECTORY));
    }

    bool createDirectory(const std::string& path)
    {
        DWORD res = CreateDirectoryA(path.c_str(), NULL);

        return res == TRUE;
    }

    const std::string& getExecutableDirectory()
    {
        static std::string folder;
        if (folder.size() == 0)
        {
            CHAR exeName[MAX_PATH];
            GetModuleFileNameA(nullptr, exeName, ARRAYSIZE(exeName));
            const std::string tmp(exeName);

            auto last = tmp.find_last_of("/\\");
            folder = tmp.substr(0, last);
        }
        return folder;
    }

    const std::string getWorkingDirectory()
    {
        CHAR curDir[MAX_PATH];
        GetCurrentDirectoryA(MAX_PATH, curDir);
        return std::string(curDir);
    }

    const std::string& getExecutableName()
    {
        static std::string filename;
        if (filename.size() == 0)
        {
            CHAR exeName[MAX_PATH];
            GetModuleFileNameA(nullptr, exeName, ARRAYSIZE(exeName));
            const std::string tmp(exeName);

            auto last = tmp.find_last_of("/\\");
            filename = tmp.substr(last + 1, std::string::npos);
        }
        return filename;
    }

    bool getEnvironmentVariable(const std::string& varName, std::string& value)
    {
        static char buff[4096];
        int numChar = GetEnvironmentVariableA(varName.c_str(), buff, arraysize(buff)); //what is the best way to deal with wchar ?
        assert(numChar < arraysize(buff));
        if (numChar == 0)
        {
            return false;
        }
        value = std::string(buff);
        return true;
    }

    template<bool bOpen>
    bool fileDialogCommon(const char* pFilters, std::string& filename)
    {
        OPENFILENAMEA ofn;
        CHAR chars[512] = "";
        ZeroMemory(&ofn, sizeof(ofn));

        ofn.lStructSize = sizeof(OPENFILENAME);
        ofn.hwndOwner = GetForegroundWindow();
        ofn.lpstrFilter = pFilters;
        ofn.lpstrFile = chars;
        ofn.nMaxFile = arraysize(chars);
        ofn.Flags = OFN_HIDEREADONLY | OFN_NOCHANGEDIR;
        if (bOpen == true)
        {
            ofn.Flags |= OFN_FILEMUSTEXIST;
        }
        ofn.lpstrDefExt = "";

        BOOL b = bOpen ? GetOpenFileNameA(&ofn) : GetSaveFileNameA(&ofn);
        if (b)
        {
            filename = std::string(chars);
            return true;
        }

        return false;
    }

    template bool fileDialogCommon<true>(const char* pFilters, std::string& filename);
    template bool fileDialogCommon<false>(const char* pFilters, std::string& filename);

    void setWindowIcon(const std::string& iconFile, Window::ApiHandle windowHandle)
    {
        std::string fullpath;
        if (findFileInDataDirectories(iconFile, fullpath))
        {
            HANDLE hIcon = LoadImageA(GetModuleHandle(NULL), fullpath.c_str(), IMAGE_ICON, 0, 0, LR_DEFAULTSIZE | LR_LOADFROMFILE);
            HWND hWnd = windowHandle ? windowHandle : GetActiveWindow();
            SendMessage(hWnd, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
        }
        else
        {
            logError("Error when loading icon. Can't find the file " + iconFile + ".");
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

    bool isDebuggerPresent()
    {
#ifdef _DEBUG
        return ::IsDebuggerPresent() == TRUE;
#else
        return false;
#endif
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
            logError("Unable to execute the render graph editor");
            return 0;
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

    static std::unordered_map<std::string, std::pair<std::thread, bool> > fileThreads;

    static void checkFileModifiedStatus(const std::string& filePath, const std::function<void(const std::string&)>& callback)
    {
        std::string fileName = getFilenameFromPath(filePath);
        std::string dir = filePath.substr(0, filePath.find_last_of('\\'));
        
        HANDLE hFile = CreateFileA(dir.c_str(), GENERIC_READ | FILE_LIST_DIRECTORY,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, NULL);
        assert(hFile != INVALID_HANDLE_VALUE);

        // overlapped struct requires unique event handle to be valid
        OVERLAPPED overlapped;

        while (true)
        {
            size_t offset = 0;
            uint32_t bytesReturned = 0;
            std::vector<uint32_t> buffer;
            buffer.resize(1024);

            if (!ReadDirectoryChangesW(hFile, buffer.data(), static_cast<uint32_t>(sizeof(uint32_t) * buffer.size()), FALSE,
                FILE_NOTIFY_CHANGE_LAST_WRITE, 0, &overlapped, nullptr))
            {
                logError("Failed to read directory changes for shared file.");
                CloseHandle(hFile);
                return;
            }

            if (!GetOverlappedResult(hFile, &overlapped, (LPDWORD)&bytesReturned, true))
            {
                logError("Failed to read directory changes for shared file.");
                CloseHandle(hFile);
                return;
                
            }

            // don't check for another overlapped result if main thread is closed
            if (!fileThreads.at(filePath).second)
            {
                break;
            }

            if (!bytesReturned) continue;

            while (offset < buffer.size())
            {
                _FILE_NOTIFY_INFORMATION* pNotifyInformation = reinterpret_cast<_FILE_NOTIFY_INFORMATION*>(buffer.data());
                std::string currentFileName;
                currentFileName.resize(pNotifyInformation->FileNameLength / 2);
                wcstombs(&currentFileName.front(), pNotifyInformation->FileName, pNotifyInformation->FileNameLength);

                if (currentFileName == fileName && pNotifyInformation->Action == FILE_ACTION_MODIFIED)
                {
                    callback(filePath);
                    break;
                }

                if (!pNotifyInformation->NextEntryOffset) break;
                offset += pNotifyInformation->NextEntryOffset;
            }
        }

        CloseHandle(hFile);
    }

    void openSharedFile(const std::string& filePath, const std::function<void(const std::string&)>& callback)
    {
        const auto& fileThreadsIt = fileThreads.find(filePath);

        // only have one thread waiting on file write
        if(fileThreadsIt != fileThreads.end())
        {
            if (fileThreadsIt->second.first.joinable())
            {
                fileThreadsIt->second.first.join();
            }
        }
        
        fileThreads[filePath].first = std::thread(checkFileModifiedStatus, filePath, callback);
        fileThreads[filePath].second = true;
    }

    void closeSharedFile(const std::string& filePath)
    {
        const auto& fileThreadsIt = fileThreads.find(filePath);

        // only have one thread waiting on file write
        if (fileThreadsIt != fileThreads.end())
        {
            fileThreadsIt->second.second = false;

            fileThreadsIt->second.first.detach();
        }
    }

    void enumerateFiles(std::string searchString, std::vector<std::string>& filenames)
    {
        WIN32_FIND_DATAA ffd;
        HANDLE hFind = INVALID_HANDLE_VALUE;

        char szFile[512];
        strcpy_s(szFile, searchString.length() + 1, searchString.c_str());

        hFind = FindFirstFileA(szFile, &ffd);

        if (INVALID_HANDLE_VALUE == hFind)
        {
            return;
        }
        else
        {
            do
            {
                if ((ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
                {
                    filenames.push_back(std::string(ffd.cFileName));
                }
            } while (FindNextFileA(hFind, &ffd) != 0);
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
            logWarning("setThreadAffinity failed with error: " + std::string(err.begin(), err.end()));
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
            should_not_get_here();

        if (DWORD dwError = GetLastError() != 0)
        {
            LPVOID lpMsgBuf;
            FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, dwError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, NULL);
            std::wstring err((LPTSTR)lpMsgBuf);
            logWarning("setThreadPriority failed with error: " + std::string(err.begin(), err.end()));
            LocalFree(lpMsgBuf);
        }
    }

    time_t getFileModifiedTime(const std::string& filename)
    {
        struct stat s;
        if (stat(filename.c_str(), &s) != 0)
        {
            logError("Can't get file time for '" + filename + "'");
            return 0;
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
        _BitScanReverse(&index, a);
        return (uint32_t)index;
    }

    uint32_t bitScanForward(uint32_t a)
    {
        unsigned long index;
        _BitScanForward(&index, a);
        return (uint32_t)index;
    }

    uint32_t popcount(uint32_t a)
    {
        return __popcnt(a);
    }
}

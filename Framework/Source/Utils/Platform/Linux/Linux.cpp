/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/StringUtils.h"
#include "Utils/Platform/OS.h"
#include "Utils/Logger.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ptrace.h>
#include <gtk/gtk.h>
#include <fstream>
#include <fcntl.h>
#include <libgen.h>
#include <errno.h>
#include <algorithm>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

namespace Falcor
{
    MsgBoxButton msgBox(const std::string& msg, MsgBoxType mbType)
    {
        if (!gtk_init_check(0, nullptr))
        {
            should_not_get_here();
        }

        GtkButtonsType buttonType = GTK_BUTTONS_OK;
        switch (mbType)
        {
        case MsgBoxType::Ok:
            buttonType = GTK_BUTTONS_OK;
            break;
        case MsgBoxType::OkCancel:
            buttonType = GTK_BUTTONS_OK_CANCEL;
            break;
        case MsgBoxType::RetryCancel:
            buttonType = GTK_BUTTONS_YES_NO;
            break;
        default:
            should_not_get_here();
            break;
        }

        GtkWidget* pParent = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        GtkWidget* pDialog = gtk_message_dialog_new(
            GTK_WINDOW(pParent),
            GTK_DIALOG_MODAL,
            GTK_MESSAGE_INFO,
            buttonType,
            "%s",
            msg.c_str()
        );

        gtk_window_set_title(GTK_WINDOW(pDialog), "Falcor");
        gint result = gtk_dialog_run(GTK_DIALOG(pDialog));
        gtk_widget_destroy(pDialog);
        gtk_widget_destroy(pParent);
        while(gtk_events_pending())
        {
            gtk_main_iteration();
        }

        switch (result)
        {
        case GTK_RESPONSE_OK: 
            return MsgBoxButton::Ok;
        case GTK_RESPONSE_CANCEL:
            return MsgBoxButton::Cancel;
        case GTK_RESPONSE_YES:
            return MsgBoxButton::Retry;
        default:
            should_not_get_here();
            return MsgBoxButton::Cancel;
        }
    }

    bool doesFileExist(const std::string& filename)
    {
        int32_t handle = open(filename.c_str(), O_RDONLY);
        struct stat fileStat;
        bool exists = fstat(handle, &fileStat) == 0;
        close(handle);
        return exists;
    }

    bool isDirectoryExists(const std::string& filename)
    {
        const char* pathname = filename.c_str();
        struct stat sb;
        return (stat(pathname, &sb) == 0) && S_ISDIR(sb.st_mode);
    }

    const std::string& getExecutableDirectory()
    {
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        const char* path;
        if (count != -1)
        {
            path = dirname(result);
        }
        static std::string strpath(path);
        return strpath;
    }

    const std::string getWorkingDirectory()
    {
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != nullptr)
        {
            return std::string(cwd);
        }

        return std::string();
    }

    const std::string& getExecutableName()
    {
        static std::string filename = fs::path(program_invocation_name).filename();
        return filename;
    }

    bool getEnvironmentVariable(const std::string& varName, std::string& value)
    {
        const char* val = ::getenv(varName.c_str());
        if (val == 0)
        {
            return false;
        }
        static std::string strvar(val);
        value = strvar;
        return true;
    }

    template<bool bOpen>
    bool fileDialogCommon(const char* pFilters, std::string& filename)
    {
        if (!gtk_init_check(0, nullptr))
        {
            should_not_get_here();
        }

        GtkWidget* pParent = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        GtkWidget* pDialog = nullptr;
        gint result = 0;

        bool success = false;
        if(bOpen)
        {
            pDialog = gtk_file_chooser_dialog_new(
                "Open File",
                GTK_WINDOW(pParent),
                GTK_FILE_CHOOSER_ACTION_OPEN,
                "_Cancel",
                GTK_RESPONSE_CANCEL,
                "_Open",
                GTK_RESPONSE_ACCEPT,
                NULL);

            result = gtk_dialog_run(GTK_DIALOG(pDialog));
            if (result == GTK_RESPONSE_ACCEPT)
            {
                char* gtkFilename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(pDialog));
                filename = std::string(gtkFilename);
                g_free(gtkFilename);
                success = true;
            }
        }
        else
        {
            pDialog = gtk_file_chooser_dialog_new(
                "Save File",
                GTK_WINDOW(pParent),
                GTK_FILE_CHOOSER_ACTION_SAVE,
                "_Cancel",
                GTK_RESPONSE_CANCEL,
                "_Save",
                GTK_RESPONSE_ACCEPT,
                NULL);

            GtkFileChooser* pAsChooser = GTK_FILE_CHOOSER(pDialog);

            gtk_file_chooser_set_do_overwrite_confirmation(pAsChooser, TRUE);
            gtk_file_chooser_set_current_name(pAsChooser, "");

            result = gtk_dialog_run(GTK_DIALOG(pDialog));
            if (result == GTK_RESPONSE_ACCEPT)
            {
                char* gtkFilename = gtk_file_chooser_get_filename(pAsChooser);
                filename = std::string(gtkFilename);
                g_free(gtkFilename);
                success = true;
            }
        }

        gtk_widget_destroy(pDialog);
        gtk_widget_destroy(pParent);
        while(gtk_events_pending())
        {
            gtk_main_iteration();
        }
        return success;
    }

    template bool fileDialogCommon<true>(const char* pFilters, std::string& filename);
    template bool fileDialogCommon<false>(const char* pFilters, std::string& filename);

    void setActiveWindowIcon(const std::string& iconFile)
    {
        // #TODO Not yet implemented
    }

    int getDisplayDpi()
    {
        // #TODO Not yet implemented
        return int(200);
    }

    bool isDebuggerPresent()
    {
#ifdef _DEBUG
        static bool debuggerAttached = false;
        static bool isChecked = false;
        if(isChecked == false)
        {
            if(ptrace(PTRACE_TRACEME, 0, 1, 0) < 0)
            {
                debuggerAttached = true;
            }
            else
            {
                ptrace(PTRACE_DETACH, 0, 1, 0);
            }
            isChecked = true;
        }

        return debuggerAttached;
#else
        return false;
#endif
    }

    void debugBreak()
    {
        raise(SIGTRAP);
    }

    void printToDebugWindow(const std::string& s)
    {
        std::cerr << s;
    }

    void enumerateFiles(std::string searchString, std::vector<std::string>& filenames)
    {
        DIR* pDir = opendir(searchString.c_str());
        if(pDir != nullptr)
        {
            struct dirent* pDirEntry = readdir(pDir);
            while(pDirEntry != nullptr)
            {
                // Only add files, no subdirectories, symlinks, or other objects
                if(pDirEntry->d_type == DT_REG)
                {
                    filenames.push_back(pDirEntry->d_name);
                }

                pDirEntry = readdir(pDir);
            }
        }
    }

    std::thread::native_handle_type getCurrentThread()
    {
        return pthread_self();
    }

    std::string threadErrorToString(int32_t error)
    {
        // Error details can vary depending on what function returned it,
        // just convert error id to string for easy lookup.
        switch(error)
        {
            case EFAULT: return "EFAULT";
            case ENOTSUP: return "ENOTSUP";
            case EINVAL: return "EINVAL";
            case EPERM: return "EPERM";
            case ESRCH: return "ESRCH";
            default: return std::to_string(error);
        }
    }

    void setThreadAffinity(std::thread::native_handle_type thread, uint32_t affinityMask)
    {
        cpu_set_t cpuMask;
        CPU_ZERO(&cpuMask);

        uint32_t bitCount = min(sizeof(cpu_set_t), sizeof(uint32_t)) * 8;
        for(uint32_t i = 0; i < bitCount; i++)
        {
            if((affinityMask & (1 << i)) > 0)
            {
                CPU_SET(i, &cpuMask);
            }
        }

        int32_t result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuMask);
        if(result != 0)
        {
            logError("setThreadAffinity() - pthread_setaffinity_np() failed with error code " + threadErrorToString(result));
        }
    }

    void setThreadPriority(std::thread::native_handle_type thread, ThreadPriorityType priority)
    {
        pthread_attr_t thAttr;
        int32_t policy = 0;
        pthread_getattr_np(thread, &thAttr);
        pthread_attr_getschedpolicy(&thAttr, &policy);

        int32_t result = 0;
        if (priority >= ThreadPriorityType::Lowest)
        {
            // Remap enum value range to what was queried from system
            float minPriority = (float)sched_get_priority_min(policy);
            float maxPriority = (float)sched_get_priority_max(policy);
            float value = (float)priority * (maxPriority - minPriority) / (float)(ThreadPriorityType::Highest) + minPriority;
            result = pthread_setschedprio(thread, (int32_t)value);
            pthread_attr_destroy(&thAttr);
        }
        // #TODO: Is there a "Background" priority in Linux? Is there a way to emulate it?
        else
        {
            should_not_get_here();
        }

        if(result != 0)
        {
            logError("setThreadPriority() - pthread_setschedprio() failed with error code " + threadErrorToString(result));
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

    uint32_t bitScanReverse(uint32_t a)
    {
        // __builtin_clz counts 0's from the MSB, convert to index from the LSB
        return (sizeof(uint32_t) * 8) - (uint32_t)__builtin_clz(a) - 1;
    }

    /** Returns index of least significant set bit, or 0 if no bits were set
    */
    uint32_t bitScanForward(uint32_t a)
    {
        // __builtin_ctz() counts 0's from LSB, which is the same as the index of the first set bit
        // Manually return 0 if a is 0 to match Microsoft behavior. __builtin_ctz(0) produces undefined results.
        return (a > 0) ? ((uint32_t)__builtin_ctz(a)) : 0;
    }

    uint32_t popcount(uint32_t a)
    {
        return (uint32_t)__builtin_popcount(a);
    }

}

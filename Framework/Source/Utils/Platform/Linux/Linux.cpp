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

//#include <iostream>
//#include <stdint.h>
//#include <sstream>
//#include <string>
//#include <vector>
//#include <unistd.h>
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

namespace Falcor
{
    MsgBoxButton msgBox(const std::string& msg, MsgBoxType mbType)
    {
        const char *message = msg.c_str();
        if (!gtk_init_check(0, nullptr))
        {
            should_not_get_here();
        }
        std::string title = "Falcor";
        GtkButtonsType buttontype = GTK_BUTTONS_OK;
        switch (mbType)
        {
        case MsgBoxType::Ok:
            buttontype = GTK_BUTTONS_OK;
            break;
        case MsgBoxType::OkCancel:
            buttontype = GTK_BUTTONS_OK_CANCEL;
            break;
        case MsgBoxType::RetryCancel:
            buttontype = GTK_BUTTONS_YES_NO;
            break;
        default:
            should_not_get_here();
            break;
        }
        GtkWidget *parent = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        GtkWidget *dialog = gtk_message_dialog_new(
            GTK_WINDOW(parent),
            GTK_DIALOG_MODAL,
            GTK_MESSAGE_INFO,
            buttontype,
            "%s",
            message
        );
        gtk_window_set_title(GTK_WINDOW(dialog), "Falcor");
        gint value = gtk_dialog_run(GTK_DIALOG(dialog));
        gtk_widget_destroy(GTK_WIDGET(dialog));
        gtk_widget_destroy(GTK_WIDGET(parent));
        switch (value)
        {
        case GTK_RESPONSE_OK:
            while (gtk_events_pending())
                gtk_main_iteration();
            return MsgBoxButton::Ok;
        case GTK_RESPONSE_CANCEL:
            while (gtk_events_pending())
                gtk_main_iteration();
            return MsgBoxButton::Cancel;
        case GTK_RESPONSE_YES:
            while (gtk_events_pending())
                gtk_main_iteration();
            return MsgBoxButton::Retry;
        default:
            while (gtk_events_pending())
                gtk_main_iteration();
            should_not_get_here();
            return MsgBoxButton::Cancel;
        }
    }

    bool doesFileExist(const std::string& filename)
    {
        const char *file = filename.c_str();
        int fn = open(file, O_RDONLY);
        struct stat fileStat;
        if (fstat(fn, &fileStat) < 0)
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }

    bool isDirectoryExists(const std::string& filename)
    {
        const char *pathname = filename.c_str();
        struct stat sb;
        if (stat(pathname, &sb) == 0 && S_ISDIR(sb.st_mode))
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    const std::string& getExecutableDirectory()
    {
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        const char *path;
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
        static std::string filename;
        filename = program_invocation_name;
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

    std::vector<std::string> gDataDirectories =
    {
        // Ordering matters here, we want that while developing, resources will be loaded from the development media directory
        std::string(getWorkingDirectory()),
        std::string(getWorkingDirectory() + "/data"),
        std::string(getExecutableDirectory()),
        std::string(getExecutableDirectory() + "/data"),

        // The local solution media folder
        std::string(getExecutableDirectory() + "/../../../Media"),
    };

    const std::vector<std::string>& getDataDirectoriesList()
    {
        return gDataDirectories;
    }

    void addDataDirectory(const std::string& dataDir)
    {
        //Insert unique elements
        if (std::find(gDataDirectories.begin(), gDataDirectories.end(), dataDir) == gDataDirectories.end())
        {
            gDataDirectories.push_back(dataDir);
        }
    }

    std::string canonicalizeFilename(const std::string& filename)
    {
        return std::experimental::filesystem::canonical(filename).string();
    }

    bool findFileInDataDirectories(const std::string& filename, std::string& fullpath)
    {
        static bool bInit = false;
        if (bInit == false)
        {
            std::string dataDirs;
            if (getEnvironmentVariable("FALCOR_MEDIA_FOLDERS", dataDirs))
            {
                auto folders = splitString(dataDirs, ";");
                gDataDirectories.insert(gDataDirectories.end(), folders.begin(), folders.end());
            }
            bInit = true;
        }

        // Check if this is an absolute path
        if (doesFileExist(filename))
        {
            fullpath = canonicalizeFilename(filename);
            return true;
        }

        for (const auto& Dir : gDataDirectories)
        {
            fullpath = canonicalizeFilename(Dir + '/' + filename);
            if (doesFileExist(fullpath))
            {
                return true;
            }
        }

        return false;
    }

    template<bool bOpen>
    static bool fileDialogCommon(const char* pFilters, std::string& filename)
    {
        if (!gtk_init_check(0, nullptr))
        {
            should_not_get_here();
        }

        GtkWidget *dialog;
        GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
        gint res;

        dialog = gtk_file_chooser_dialog_new("File Dialog",
            NULL,
            action,
            "_Cancel",
            GTK_RESPONSE_CANCEL,
            "_Open",
            GTK_RESPONSE_ACCEPT,
            NULL);

        res = gtk_dialog_run(GTK_DIALOG(dialog));
        if (res == GTK_RESPONSE_ACCEPT)
        {
            char *fn;
            GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
            fn = gtk_file_chooser_get_filename(chooser);
            std::stringstream ss;
            ss << fn;
            ss >> filename;
            //open_file(fn);
            //g_free(fn);
            gtk_widget_destroy(dialog);
            while (gtk_events_pending())
                gtk_main_iteration();
            return true;
        }

        while (gtk_events_pending())
            gtk_main_iteration();
        gtk_widget_destroy(dialog);
        return false;
    }

    bool openFileDialog(const char* pFilters, std::string& filename)
    {
        return fileDialogCommon<true>(pFilters, filename);
    }

    bool saveFileDialog(const char* pFilters, std::string& filename)
    {
        return fileDialogCommon<false>(pFilters, filename);
    }

    bool readFileToString(const std::string& fullpath, std::string& str)
    {
        std::ifstream t(fullpath.c_str());
        if ((t.rdstate() & std::ifstream::failbit) == 0)
        {
            str = std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
            return true;
        }
        return false;
    }

    bool findAvailableFilename(const std::string& prefix, const std::string& directory, const std::string& extension, std::string& filename)
    {
        for (uint32_t i = 0; i < (uint32_t)-1; i++)
        {
            std::string newPrefix = prefix + '.' + std::to_string(i);
            filename = directory + '/' + newPrefix + "." + extension;

            if (doesFileExist(filename) == false)
            {
                return true;
            }
        }
        should_not_get_here();
        filename = "";
        return false;
    }

    void setActiveWindowIcon(const std::string& iconFile)
    {
        //////////////////////////////////////////
        // THIS IS NOT IMPLEMENTED IN LINUX.
        //////////////////////////////////////////
    }

    int getDisplayDpi()
    {
        //////////////////////////////////////////
        // THIS IS NOT IMPLEMENTED IN LINUX.
        //////////////////////////////////////////
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

    std::string stripDataDirectories(const std::string& filename)
    {
        std::string stripped = filename;
        std::string canonFile = canonicalizeFilename(filename);
        for (const auto& dir : gDataDirectories)
        {
            std::string canonDir = canonicalizeFilename(dir);
            if (hasPrefix(canonFile, canonDir, false))
            {
                std::string tmp = canonFile.erase(0, canonDir.length() + 1);
                if (tmp.length() < stripped.length())
                {
                    stripped = tmp;
                }
            }
        }

        return stripped;
    }

    std::string getDirectoryFromFile(const std::string& filename)
    {
        char* path = const_cast<char*>(filename.c_str());
        path = dirname(path);
        return std::string(path);
    }

    std::string getFilenameFromPath(const std::string& filename)
    {
        char* path = const_cast<char*>(filename.c_str());
        path = basename(path);
        return std::string(path);
    }

    std::string swapFileExtension(const std::string& str, const std::string& currentExtension, const std::string& newExtension)
    {
        if (hasSuffix(str, currentExtension))
        {
            std::string ret = str;
            return (ret.erase(ret.rfind(currentExtension)) + newExtension);
        }
        else
        {
            return str;
        }
    }

    void enumerateFiles(std::string searchString, std::vector<std::string>& filenames)
    {
        //////////////////////////////////////////
        // THIS IS NOT IMPLEMENTED IN LINUX.
        //////////////////////////////////////////
        should_not_get_here();
    }

    std::thread::native_handle_type getCurrentThread()
    {
        //////////////////////////////////////////
        // THIS IS NOT IMPLEMENTED IN LINUX.
        //////////////////////////////////////////
        should_not_get_here();
        return std::thread::native_handle_type();
    }

    void setThreadAffinity(std::thread::native_handle_type thread, uint32_t affinityMask)
    {
        //////////////////////////////////////////
        // THIS IS NOT IMPLEMENTED IN LINUX.
        //////////////////////////////////////////
        should_not_get_here();
    }

    void setThreadPriority(std::thread::native_handle_type thread, ThreadPriorityType priority)
    {
        //////////////////////////////////////////
        // THIS IS NOT IMPLEMENTED IN LINUX.
        //////////////////////////////////////////
        should_not_get_here();
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
}

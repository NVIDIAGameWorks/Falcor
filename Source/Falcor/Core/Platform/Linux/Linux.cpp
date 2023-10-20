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
#include "Core/Platform/OS.h"
#include "Core/Error.h"
#include "Core/GLFW.h"
#include "Utils/Logger.h"
#include "Utils/StringUtils.h"

#include <gtk/gtk.h>

#include <iostream>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pwd.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE // needed for dladdr()
#endif
#include <dlfcn.h>

#include <mutex>

namespace Falcor
{

MsgBoxButton msgBox(const std::string& title, const std::string& msg, MsgBoxType type, MsgBoxIcon icon)
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
    case MsgBoxType::Ok:
        buttons = {buttonOk};
        break;
    case MsgBoxType::OkCancel:
        buttons = {buttonOk, buttonCancel};
        break;
    case MsgBoxType::RetryCancel:
        buttons = {buttonRetry, buttonCancel};
        break;
    case MsgBoxType::AbortRetryIgnore:
        buttons = {buttonAbort, buttonRetry, buttonIgnore};
        break;
    case MsgBoxType::YesNo:
        buttons = {buttonYes, buttonNo};
        break;
    default:
        FALCOR_UNREACHABLE();
    }

    return (MsgBoxButton)msgBox(title, msg, buttons, icon);
}

uint32_t msgBox(
    const std::string& title,
    const std::string& msg,
    std::vector<MsgBoxCustomButton> buttons,
    MsgBoxIcon icon,
    uint32_t defaultButtonId
)
{
    if (!gtk_init_check(0, nullptr))
        FALCOR_THROW("Failed to initialize GTK.");

    GtkWidget* pParent = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    GtkWidget* pDialog = gtk_message_dialog_new(
        GTK_WINDOW(pParent), // parent
        GTK_DIALOG_MODAL,    // flags
        GTK_MESSAGE_INFO,    // type
        GTK_BUTTONS_NONE,    // buttons
        "%s",                // message_format
        msg.c_str()          // args
    );

    for (const auto& button : buttons)
    {
        gtk_dialog_add_button(GTK_DIALOG(pDialog), button.title.c_str(), (gint)button.id);
    }

    gtk_window_set_title(GTK_WINDOW(pDialog), title.c_str());
    gint result = gtk_dialog_run(GTK_DIALOG(pDialog));
    gtk_widget_destroy(pDialog);
    gtk_widget_destroy(pParent);
    while (gtk_events_pending())
    {
        gtk_main_iteration();
    }

    return result;
}

size_t executeProcess(const std::string& appName, const std::string& commandLineArgs)
{
    std::string linuxAppName = getRuntimeDirectory() / appName;
    std::vector<const char*> argv;
    std::vector<std::string> argvStrings;

    auto argStrings = splitString(commandLineArgs, " ");
    argvStrings.insert(argvStrings.end(), argStrings.begin(), argStrings.end());

    for (const std::string& argString : argvStrings)
    {
        argv.push_back(argString.c_str());
    }
    argv.push_back(nullptr);

    int32_t forkVal = fork();

    FALCOR_ASSERT(forkVal != -1);
    if (forkVal == 0)
    {
        if (execv(linuxAppName.c_str(), (char* const*)argv.data()))
        {
            FALCOR_THROW("Unable to execute process: {} {}", appName, commandLineArgs);
        }
    }

    return forkVal;
}

bool isProcessRunning(size_t processID)
{
    FALCOR_UNIMPLEMENTED();
    return static_cast<bool>(processID);
}

void terminateProcess(size_t processID)
{
    (void)processID;
    FALCOR_UNIMPLEMENTED();
}

void monitorFileUpdates(const std::filesystem::path& path, const std::function<void()>& callback)
{
    (void)path;
    (void)callback;
    FALCOR_UNIMPLEMENTED();
}

void closeSharedFile(const std::filesystem::path& path)
{
    (void)path;
    FALCOR_UNIMPLEMENTED();
}

bool createJunction(const std::filesystem::path& link, const std::filesystem::path& target)
{
    std::error_code ec;
    std::filesystem::create_directory_symlink(target, link, ec);
    if (ec)
    {
        logWarning("Failed to create symlink {} to {}: {}", link, target, ec.value());
    }
    return !ec;
}

bool deleteJunction(const std::filesystem::path& link)
{
    std::error_code ec;
    std::filesystem::remove(link, ec);
    if (ec)
    {
        logWarning("Failed to remove symlink {}: {}", link, ec.value());
    }
    return !ec;
}

const std::filesystem::path& getExecutablePath()
{
    static std::filesystem::path path(
        []()
        {
            char pathStr[PATH_MAX] = {0};
            if (readlink("/proc/self/exe", pathStr, PATH_MAX) == -1)
            {
                FALCOR_THROW("Failed to get the executable path.");
            }
            return std::filesystem::path(pathStr);
        }()
    );
    return path;
}

const std::filesystem::path& getRuntimeDirectory()
{
    static std::filesystem::path path(
        []()
        {
            Dl_info info;
            if (dladdr((void*)&getRuntimeDirectory, &info) == 0)
            {
                FALCOR_THROW("Failed to get the falcor directory. dladdr() failed.");
            }
            return std::filesystem::path(info.dli_fname).parent_path();
        }()
    );
    return path;
}

const std::filesystem::path& getAppDataDirectory()
{
    static std::filesystem::path path(
        []()
        {
            const char* homeDir;
            if ((homeDir = getenv("HOME")) == nullptr)
            {
                homeDir = getpwuid(getuid())->pw_dir;
            }
            return std::filesystem::path(homeDir) / ".falcor";
        }()
    );
    return path;
}

const std::filesystem::path& getHomeDirectory()
{
    static std::filesystem::path path(
        []()
        {
            if (auto value = getEnvironmentVariable("HOME"))
                return std::filesystem::path(*value);
            return std::filesystem::path();
        }()
    );
    return path;
}

std::optional<std::string> getEnvironmentVariable(const std::string& varName)
{
    const char* val = ::getenv(varName.c_str());
    return val != nullptr ? std::string(val) : std::optional<std::string>{};
}

template<bool bOpen>
bool fileDialogCommon(const FileDialogFilterVec& filters, std::filesystem::path& path)
{
    if (!gtk_init_check(0, nullptr))
        FALCOR_THROW("Failed to initialize GTK.");

    GtkWidget* pParent = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    GtkWidget* pDialog = nullptr;
    gint result = 0;

    bool success = false;
    if (bOpen)
    {
        pDialog = gtk_file_chooser_dialog_new(
            "Open File",                  // title
            GTK_WINDOW(pParent),          // parent
            GTK_FILE_CHOOSER_ACTION_OPEN, // action
            "_Cancel",                    // button text
            GTK_RESPONSE_CANCEL,          // button response id
            "_Open",                      // button text
            GTK_RESPONSE_ACCEPT,          // button response id
            NULL                          // end of buttons
        );

        result = gtk_dialog_run(GTK_DIALOG(pDialog));
        if (result == GTK_RESPONSE_ACCEPT)
        {
            gchar* gtkFilename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(pDialog));
            path = static_cast<const char*>(gtkFilename);
            g_free(gtkFilename);
            success = true;
        }
    }
    else
    {
        pDialog = gtk_file_chooser_dialog_new(
            "Save File",                  // title
            GTK_WINDOW(pParent),          // parent
            GTK_FILE_CHOOSER_ACTION_SAVE, // action
            "_Cancel",                    // button text
            GTK_RESPONSE_CANCEL,          // button response id
            "_Save",                      // button text
            GTK_RESPONSE_ACCEPT,          // button response id
            NULL                          // end of buttons
        );

        GtkFileChooser* pAsChooser = GTK_FILE_CHOOSER(pDialog);

        gtk_file_chooser_set_do_overwrite_confirmation(pAsChooser, TRUE);
        gtk_file_chooser_set_current_name(pAsChooser, "");

        result = gtk_dialog_run(GTK_DIALOG(pDialog));
        if (result == GTK_RESPONSE_ACCEPT)
        {
            char* gtkFilename = gtk_file_chooser_get_filename(pAsChooser);
            path = std::filesystem::path(gtkFilename);
            g_free(gtkFilename);
            success = true;
        }
    }

    gtk_widget_destroy(pDialog);
    gtk_widget_destroy(pParent);
    while (gtk_events_pending())
    {
        gtk_main_iteration();
    }
    return success;
}

bool openFileDialog(const FileDialogFilterVec& filters, std::filesystem::path& path)
{
    return fileDialogCommon<true>(filters, path);
}

bool saveFileDialog(const FileDialogFilterVec& filters, std::filesystem::path& path)
{
    return fileDialogCommon<false>(filters, path);
}

bool chooseFolderDialog(std::filesystem::path& path)
{
    FALCOR_UNIMPLEMENTED();
}

float getDisplayScaleFactor()
{
    float xscale = 1.f;
    float yscale = 1.f;
    auto monitor = glfwGetPrimaryMonitor();
    if (monitor)
        glfwGetMonitorContentScale(monitor, &xscale, &yscale);
    return 0.5f * (xscale + yscale);
}

struct KeyboardInterruptData
{
    std::mutex mutex;
    std::function<void()> handler;

    static KeyboardInterruptData& get()
    {
        static KeyboardInterruptData data;
        return data;
    }
};

static void signalHandler(int sig)
{
    KeyboardInterruptData& data = KeyboardInterruptData::get();

    if (sig == SIGINT)
    {
        std::lock_guard<std::mutex> lock(data.mutex);
        if (data.handler)
        {
            data.handler();
        }
    }
}

void setKeyboardInterruptHandler(std::function<void()> handler)
{
    KeyboardInterruptData& data = KeyboardInterruptData::get();
    std::lock_guard<std::mutex> lock(data.mutex);

    if (handler && !data.handler)
    {
        struct sigaction action;
        action.sa_handler = signalHandler;
        sigemptyset(&action.sa_mask);
        action.sa_flags = 0;
        if (sigaction(SIGINT, &action, nullptr) != 0)
            FALCOR_THROW("Failed to register keyboard interrupt handler");
    }
    else if (!handler && data.handler)
    {
        struct sigaction action;
        action.sa_handler = SIG_DFL;
        sigemptyset(&action.sa_mask);
        action.sa_flags = 0;
        if (sigaction(SIGINT, &action, nullptr) != 0)
            FALCOR_THROW("Failed to unregister keyboard interrupt handler");
    }
    data.handler = handler;
}

bool isDebuggerPresent()
{
#if 0 // TODO: Implement
        static bool debuggerAttached = false;
        static bool isChecked = false;
        if (isChecked == false)
        {
            if (ptrace(PTRACE_TRACEME, 0, 1, 0) < 0)
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

std::thread::native_handle_type getCurrentThread()
{
    return pthread_self();
}

std::string threadErrorToString(int32_t error)
{
    // Error details can vary depending on what function returned it,
    // just convert error id to string for easy lookup.
    switch (error)
    {
    case EFAULT:
        return "EFAULT";
    case ENOTSUP:
        return "ENOTSUP";
    case EINVAL:
        return "EINVAL";
    case EPERM:
        return "EPERM";
    case ESRCH:
        return "ESRCH";
    default:
        return std::to_string(error);
    }
}

void setThreadAffinity(std::thread::native_handle_type thread, uint32_t affinityMask)
{
    cpu_set_t cpuMask;
    CPU_ZERO(&cpuMask);

    uint32_t bitCount = std::min(sizeof(cpu_set_t), sizeof(uint32_t)) * 8;
    for (uint32_t i = 0; i < bitCount; i++)
    {
        if ((affinityMask & (1 << i)) > 0)
        {
            CPU_SET(i, &cpuMask);
        }
    }

    int32_t result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuMask);
    if (result != 0)
    {
        logError("setThreadAffinity() - pthread_setaffinity_np() failed with error code {}", threadErrorToString(result));
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
        FALCOR_UNREACHABLE();
    }

    if (result != 0)
    {
        logError("setThreadPriority() - pthread_setschedprio() failed with error code {}", threadErrorToString(result));
    }
}

time_t getFileModifiedTime(const std::filesystem::path& path)
{
    struct stat s;
    if (stat(path.c_str(), &s) != 0)
    {
        logError("Can't get file time for '{}'.", path);
        return 0;
    }

    return s.st_mtime;
}

uint32_t bitScanReverse(uint32_t a)
{
    // __builtin_clz counts 0's from the MSB, convert to index from the LSB
    return (sizeof(uint32_t) * 8) - (uint32_t)__builtin_clz(a) - 1;
}

/**
 * Returns index of least significant set bit, or 0 if no bits were set
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

SharedLibraryHandle loadSharedLibrary(const std::filesystem::path& path)
{
    return dlopen(path.c_str(), RTLD_LAZY);
}

void releaseSharedLibrary(SharedLibraryHandle library)
{
    dlclose(library);
}

/**
 * Get a function pointer from a library
 */
void* getProcAddress(SharedLibraryHandle library, const std::string& funcName)
{
    return dlsym(library, funcName.c_str());
}

void OSServices::start() {}

void OSServices::stop() {}

size_t getCurrentRSS()
{
    FALCOR_UNIMPLEMENTED();
    return 0;
}

size_t getPeakRSS()
{
    FALCOR_UNIMPLEMENTED();
    return 0;
}
} // namespace Falcor

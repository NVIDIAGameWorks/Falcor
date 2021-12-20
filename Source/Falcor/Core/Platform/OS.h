/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include <thread>
#include <functional>

namespace Falcor
{
    class Window;

    /*!
    *  \addtogroup Falcor
    *  @{
    */

    /** Utility class to start/stop OS services
    */
    class OSServices
    {
    public:
        static void start();
        static void stop();
    };

    /** Sets the main window handle.
        This is used to set the parent window when showing message boxes.
        \param[in] windowHandle Window handle.
    */
    FALCOR_API void setMainWindowHandle(WindowHandle windowHandle);

    /** Adds an icon to the foreground window.
        \param[in] iconFile Icon file name
        \param[in] windowHandle The api handle of the window for which we need to set the icon to. nullptr will apply the icon to the foreground window
    */
    FALCOR_API void setWindowIcon(const std::string& iconFile, WindowHandle windowHandle);

    /** Retrieves estimated/user-set pixel density of a display.
        \return integer value of number of pixels per inch.
    */
    FALCOR_API int getDisplayDpi();

    /** Get the requested display scale factor
    */
    FALCOR_API float getDisplayScaleFactor();

    /** Message box icons.
    */
    enum class MsgBoxIcon
    {
        None,
        Info,
        Warning,
        Error
    };

    /** Type of message box to display
    */
    enum class MsgBoxType
    {
        Ok,                 ///< Single 'OK' button
        OkCancel,           ///< OK/Cancel buttons
        RetryCancel,        ///< Retry/Cancel buttons
        AbortRetryIgnore,   ///< Abort/Retry/Ignore buttons
        YesNo,              ///< Yes/No buttons
    };

    /** Types of buttons
    */
    enum class MsgBoxButton
    {
        Ok,     ///< 'OK' Button
        Retry,  ///< Retry button
        Cancel, ///< Cancel Button
        Abort,  ///< Abort Button
        Ignore, ///< Ignore Button
        Yes,    ///< Yes Button
        No,     ///< No Button
    };

    /** Display a message box. By default, shows a message box with a single 'OK' button.
        \param[in] msg The message to display.
        \param[in] type Optional. Type of message box to display.
        \param[in] icon Optional. Icon to display.
        \return An enum indicating which button was clicked.
    */
    FALCOR_API MsgBoxButton msgBox(const std::string& msg, MsgBoxType type = MsgBoxType::Ok, MsgBoxIcon icon = MsgBoxIcon::None);

    /** Custom message box button.
    */
    struct MsgBoxCustomButton
    {
        uint32_t id;            ///< Button id used as return code. The id uint32_t(-1) is reserved.
        std::string title;      ///< Button title.
    };

    /** Display a custom message box.
        If no defaultButtonId is specified, the first button in the list is used as the default button.
        Pressing enter closes the dialog and returns the id of the default button.
        If the dialog fails to execute or the user closes the dialog (or presses escape),
        the id of the last button in the list is returned.
        \param[in] msg The message to display.
        \param[in] buttons List of buttons to show.
        \param[in] icon Optional. Icon to display.
        \param[in] defaultButtonId Optional. Button to highlight by default.
        \return The id of the button that was clicked.
    */
    FALCOR_API uint32_t msgBox(const std::string& msg, std::vector<MsgBoxCustomButton> buttons, MsgBoxIcon icon = MsgBoxIcon::None, uint32_t defaultButtonId = uint32_t(-1));

    /** Set the title for message boxes. The default value is "Falcor"
    */
    FALCOR_API void msgBoxTitle(const std::string& title);

    /** Finds a file in one of the data search directories. The arguments must not alias.
        \param[in] filename The file to look for
        \param[in] fullPath If the file was found, the full path to the file. If the file wasn't found, this is invalid.
        \return true if the file was found, otherwise false
    */
    FALCOR_API bool findFileInDataDirectories(const std::string& filename, std::string& fullPath);

    /** Finds a shader file. If in development mode (see isDevelopmentMode()), shaders are searched
        within the source directories. Otherwise, shaders are searched in the Shaders directory
        located besides the executable.
        \param[in] filename The file to look for
        \param[in] fullPath If the file was found, the full path to the file. If the file wasn't found, this is invalid.
        \return true if the file was found, otherwise false
    */
    FALCOR_API bool findFileInShaderDirectories(const std::string& filename, std::string& fullPath);

    /** Get a list of all shader directories.
    */
    FALCOR_API const std::vector<std::string>& getShaderDirectoriesList();

    /** Given a filename, returns the shortest possible path to the file relative to the data directories.
        If the file is not relative to the data directories, return the original filename
    */
    FALCOR_API std::string stripDataDirectories(const std::string& filename);

    /** Structure to help with file dialog file-extension filters
    */
    struct FALCOR_API FileDialogFilter
    {
        FileDialogFilter(const std::string& ext_, const std::string& desc_ = {}) : ext(ext_), desc(desc_) {}
        std::string desc;   // The description ("Portable Network Graphics")
        std::string ext;    // The extension, without the `.` ("png")
    };

    using FileDialogFilterVec = std::vector<FileDialogFilter>;

    /** Creates a 'open file' dialog box.
        \param[in] filters The file extensions filters
        \param[in] filename On successful return, the name of the file selected by the user.
        \return true if a file was selected, otherwise false (if the user clicked 'Cancel').
    */
    FALCOR_API bool openFileDialog(const FileDialogFilterVec& filters, std::string& filename);

    /** Creates a 'save file' dialog box.
        \param[in] filters The file extensions filters
        \param[out] filename On successful return, the name of the file selected by the user.
        \return true if a file was selected, otherwise false (if the user clicked 'Cancel').
    */
    FALCOR_API bool saveFileDialog(const FileDialogFilterVec& filters, std::string& filename);

    /** Creates a dialog box for browsing and selecting folders
        \param[out] folder On successful return, the name of the folder selected by the user.
        \return true if a folder was selected, otherwise false (if the user clicked 'Cancel').
    */
    FALCOR_API bool chooseFolderDialog(std::string& folder);

    /** Checks if a file exists in the file system. This function doesn't look in the search directories.
        \param[in] filename The file to look for
        \return true if the file was found, otherwise false
    */
    FALCOR_API bool doesFileExist(const std::string& filename);

    /** Checks if a directory exists in the file system.
        \param[in] filename The directory to look for
        \return true if the directory was found, otherwise false
    */
    FALCOR_API bool isDirectoryExists(const std::string& filename);

    /** Open watch thread for file changes and call callback when the file is written to.
        \param[in] full path to the file to watch for changes
        \param[in] callback function
    */
    FALCOR_API void monitorFileUpdates(const std::string& filePath, const std::function<void()>& callback = {});

    /** Close watch thread for file changes
        \param[in] full path to the file that was being watched for changes
    */
    FALCOR_API void closeSharedFile(const std::string& filePath);

    /** Creates a file in the temporary directory and returns the path.
        \return pathName Absolute path to unique temp file.
    */
    FALCOR_API std::string getTempFilename();

    /** Create a directory from path.
    */
    FALCOR_API bool createDirectory(const std::string& path);

    /** Create a junction (soft link).
        \param[in] link Link path.
        \param[in] target Target path.
        \return Returns true if successful.
    */
    FALCOR_API bool createJunction(const std::string& link, const std::string& target);

    /** Delete a junction (sof link).
        \param[in] link Link path.
        \return Returns true if successful.
    */
    FALCOR_API bool deleteJunction(const std::string& link);

    /** Given the app name and full command line arguments, begin the process
    */
    FALCOR_API size_t executeProcess(const std::string& appName, const std::string& commandLineArgs);

    /** Check if the given process is still active
     */
    FALCOR_API bool isProcessRunning(size_t processID);

    /** Terminate process
     */
    FALCOR_API void terminateProcess(size_t processID);

    /** Get the current executable directory
        \return The full path of the application directory
    */
    FALCOR_API const std::string& getExecutableDirectory();

    /** Get the current executable name
        \return The name of the executable
    */
    FALCOR_API const std::string& getExecutableName();

    /** Get the working directory. This can be different from the executable directory (for example, by default when you launch an app from Visual Studio, the working the directory is the directory containing the project file).
    */
    FALCOR_API const std::string getWorkingDirectory();

    /** Get the application data directory.
    */
    FALCOR_API const std::string getAppDataDirectory();

    /** Get the content of a system environment variable.
        \param[in] varName Name of the environment variable
        \param[out] value On success, will hold the value of the environment variable.
        \return true if environment variable was found, otherwise false.
    */
    FALCOR_API bool getEnvironmentVariable(const std::string& varName, std::string& value);

    /** Get a list of all recorded data directories.
    */
    FALCOR_API const std::vector<std::string>& getDataDirectoriesList();

    /** Adds a folder to data search directories. Once added, calls to findFileInDataDirectories() will search that directory as well.
        \param[in] dir The new directory to add to the data search directories.
        \param[in] addToFront Add the new directory to the front of the list, making it the highest priority.
    */
    FALCOR_API void addDataDirectory(const std::string& dir, bool addToFront = false);

    /** Removes a folder from the data search directories
        \param[in] dir The directory name to remove from the data search directories.
    */
    FALCOR_API void removeDataDirectory(const std::string& dir);

    /** Find a new filename based on the supplied parameters. This function doesn't actually create the file, just find an available file name.
        \param[in] prefix Requested file prefix.
        \param[in] directory The directory to create the file in.
        \param[in] extension The requested file extension.
        \param[out] filename On success, will hold a valid unused filename in the following format - 'Directory\\Prefix.<index>.Extension'.
        \return true if an available filename was found, otherwise false.
    */
    FALCOR_API bool findAvailableFilename(const std::string& prefix, const std::string& directory, const std::string& extension, std::string& filename);

    /** Check if a debugger session is attached.
        \return true if debugger is attached to the Falcor process.
    */
    FALCOR_API bool isDebuggerPresent();

    /** Check if application is launched in development mode.
        Development mode is enabled by having FALCOR_DEVMODE=1 as an environment variable on launch.
        \return true if application is in development mode.
    */
    FALCOR_API bool isDevelopmentMode();

    /** Remove navigational elements ('.', '..) from a given path/filename and make slash direction consistent.
    */
    FALCOR_API std::string canonicalizeFilename(const std::string& filename);

    /** Breaks in debugger (int 3 functionality)
    */
    FALCOR_API void debugBreak();

    /** Print a message into the debug window
        \param[in] s Text to pring
    */
    FALCOR_API void printToDebugWindow(const std::string& s);

    /** Get directory from filename.
        \param[in] filename File path to strip directory from
        \return Stripped directory path
    */
    FALCOR_API std::string getDirectoryFromFile(const std::string& filename);

    /** Get  extension tag from filename.
        \param[in] filename File path to strip extension name from
        \return Stripped extension name.
    */
    FALCOR_API std::string getExtensionFromFile(const std::string& filename);

    /** Strip path from a full filename
        \param[in] filename File path
        \return Stripped filename
    */
    FALCOR_API std::string getFilenameFromPath(const std::string& filename);

    /** Swap file extension (very simple implementation)
        \param[in] str File name or full path
        \param[in] currentExtension Current extension to look for
        \param[in] newExtension Extension to replace the current with
        \return If end of str matches currentExtension, returns the file name replaced with the new extension, otherwise returns the original file name.
    */
    FALCOR_API std::string swapFileExtension(const std::string& str, const std::string& currentExtension, const std::string& newExtension);

    /** Enumerate files using search string
        \param[in] searchString String to use in file search
        \param[out] filenames Vector of found filenames
    */
    FALCOR_API void enumerateFiles(std::string searchString, std::vector<std::string>& filenames);

    /** Return current thread handle
    */
    FALCOR_API std::thread::native_handle_type getCurrentThread();

    /** Sets thread affinity mask
    */
    FALCOR_API void setThreadAffinity(std::thread::native_handle_type thread, uint32_t affinityMask);

    /** Get the last time a file was modified. If the file is not found will return 0
        \param[in] filename The file to look for
        \return Epoch timestamp of when the file was last modified
    */
    FALCOR_API time_t getFileModifiedTime(const std::string& filename);

    enum class ThreadPriorityType : int32_t
    {
        BackgroundBegin     = -2,   //< Indicates I/O-intense thread
        BackgroundEnd       = -1,   //< Indicates the end of I/O-intense operations in the thread
        Lowest              = 0,    //< Lowest priority
        Low                 = 1,
        Normal              = 2,
        High                = 3,
        Highest             = 4,
    };

    /** Sets thread priority
    */
    FALCOR_API void setThreadPriority(std::thread::native_handle_type thread, ThreadPriorityType priority);

    /** Get the Total Virtual Memory.
    */
    FALCOR_API uint64_t getTotalVirtualMemory();

    /** Get the Used Virtual Memory.
    */
    FALCOR_API uint64_t getUsedVirtualMemory();

    /** Get the Virtual Memory Used by this Process.
    */
    FALCOR_API uint64_t  getProcessUsedVirtualMemory();

    /** Returns index of most significant set bit, or 0 if no bits were set.
    */
    FALCOR_API uint32_t bitScanReverse(uint32_t a);

    /** Returns index of least significant set bit, or 0 if no bits were set.
    */
    FALCOR_API uint32_t bitScanForward(uint32_t a);

    /** Gets the closest power of two to a number, rounded down.
    */
    FALCOR_API uint32_t getLowerPowerOf2(uint32_t a);

    /** Gets the number of set bits.
    */
    FALCOR_API uint32_t popcount(uint32_t a);

    /** Load the content of a file into a string
    */
    FALCOR_API std::string readFile(const std::string& filename);

    /** Load a shared-library
    */
    FALCOR_API DllHandle loadDll(const std::string& libPath);

    /** Release a shared-library
    */
    FALCOR_API void releaseDll(DllHandle dll);

    /** Get a function pointer from a library
    */
    FALCOR_API void* getDllProcAddress(DllHandle dll, const std::string& funcName);

    /** Post a quit message with an exit code
    */
    FALCOR_API void postQuitMessage(int32_t exitCode);
    /*! @} */
};

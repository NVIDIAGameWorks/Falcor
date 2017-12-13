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
#pragma once
#include <string>
#include <vector>
#include <thread>
#include "API/Window.h"

namespace Falcor
{
    class Window;

    /*!
    *  \addtogroup Falcor
    *  @{
    */

    /** Adds an icon to the foreground window.
        \param[in] iconFile Icon file name
        \param[in] windowHandle The api handle of the window for which we need to set the icon to. nullptr will apply the icon to the foreground window
    */
    void setWindowIcon(const std::string& iconFile, Window::ApiHandle windowHandle);

    /** Retrieves estimated/user-set pixel density of a display.
        \return integer value of number of pixels per inch.
    */
    int getDisplayDpi();

    /** Type of message box to display
    */
    enum class MsgBoxType
    {
        Ok,          ///< Single 'OK' button
        RetryCancel, ///< Retry/Cancel buttons
        OkCancel,    ///< OK/Cancel buttons
    };

    /** Types of buttons
    */
    enum class MsgBoxButton
    {
        Ok,     ///< 'OK' Button
        Retry,  ///< Retry button
        Cancel  ///< Cancel Button
    };

    /** Display a message box. By default, shows a message box with a single 'OK' button.
        \param[in] msg The message to display.
        \param[in] mbType Optional. Type of message box to display
        \return An enum indicating which button was clicked
    */
    MsgBoxButton msgBox(const std::string& msg, MsgBoxType mbType = MsgBoxType::Ok);

    /** Finds a file in one of the media directories. The arguments must not alias.
        \param[in] filename The file to look for
        \param[in] fullPath If the file was found, the full path to the file. If the file wasn't found, this is invalid.
        \return true if the file was found, otherwise false
    */
    bool findFileInDataDirectories(const std::string& filename, std::string& fullPath);

    /** Given a filename, returns the shortest possible path to the file relative to the data directories.
        If the file is not relative to the data directories, return the original filename
    */
    std::string stripDataDirectories(const std::string& filename);

    /** Creates a 'open file' dialog box.
        \param[in] pFilters A string containing pairs of null terminating strings. The first string in each pair is the name of the filter, the second string in a pair is a semicolon separated list of file extensions
                   (for example, "*.TXT;*.DOC;*.BAK"). The last pair in the filter list has to end with a 2 null characters.
        \param[in] filename On successful return, the name of the file selected by the user.
        \return true if a file was selected, otherwise false (if the user clicked 'Cancel').
    */
    bool openFileDialog(const char* pFilters, std::string& filename);

    /** Creates a 'save file' dialog box.
        \param[in] pFilters A string containing pairs of null terminating strings. The first string in each pair is the name of the filter, the second string in a pair is a semicolon separated list of file extensions
                   (for example, "*.TXT;*.DOC;*.BAK"). The last pair in the filter list has to end with a 2 null characters.
        \param[in] filename On successful return, the name of the file selected by the user.
        \return true if a file was selected, otherwise false (if the user clicked 'Cancel').
    */
    bool saveFileDialog(const char* pFilters, std::string& filename);

    /** Checks if a file exists in the file system. This function doesn't look in the common directories.
        \param[in] filename The file to look for
        \return true if the file was found, otherwise false
    */
    bool doesFileExist(const std::string& filename);
    
    /** Checks if a directory exists in the file system.
        \param[in] filename The directory to look for
        \return true if the directory was found, otherwise false
    */
    bool isDirectoryExists(const std::string& filename);
    
    /** Create a directory from path.
    */
    bool createDirectory(const std::string& path);

    /** Get the current executable directory
        \return The full path of the application directory
    */
    const std::string& getExecutableDirectory();
    /** Get the current executable name
        \return The name of the executable
    */
    const std::string& getExecutableName();

    /** Get the working directory. This can be different from the executable directory (for example, by default when you launch an app from Visual Studio, the working the directory is the directory containing the project file).
    */ 
    const std::string getWorkingDirectory();

    /** Get the content of a system environment variable.
        \param[in] varName Name of the environment variable
        \param[out] value On success, will hold the value of the environment variable.
        \return true if environment variable was found, otherwise false.
    */
    bool getEnvironmentVariable(const std::string& varName, std::string& value);

    /** Get a list of all recorded data directories.
    */
    const std::vector<std::string>& getDataDirectoriesList();

    /** Read a file into a string. The function expects a full path to the file, and will not look in the common directories.
        \param[in] fullpath The path to the requested file
        \param[in] str On successful return, the content of the file
        \return true if the was read successfully, false if an error occurred (usually file not found)
    */
    bool readFileToString(const std::string& fullpath, std::string& str);

    /** Adds a folder into the search directory. Once added, calls to FindFileInCommonDirs() will seach that directory as well
        \param[in] dir The new directory to add to the common directories.
    */
    void addDataDirectory(const std::string& dir);

    /** Find a new filename based on the supplied parameters. This function doesn't actually create the file, just find an available file name.
        \param[in] prefix Requested file prefix.
        \param[in] directory The directory to create the file in.
        \param[in] extension The requested file extension.
        \param[out] filename On success, will hold a valid unused filename in the following format - 'Directory\\Prefix.<index>.Extension'.
        \return true if an available filename was found, otherwise false.
    */
    bool findAvailableFilename(const std::string& prefix, const std::string& directory, const std::string& extension, std::string& filename);

    /** Check if a debugger session is attached.
        \return true if debugger is attached to the Falcor process.
    */
    bool isDebuggerPresent();
    
    /** Remove navigational elements ('.', '..) from a given path/filename and make slash direction consistent.
    */
    std::string canonicalizeFilename(const std::string& filename);

    /** Breaks in debugger (int 3 functionality)
    */
    void debugBreak();

    /** Print a message into the debug window
        \param[in] s Text to pring
    */
    void printToDebugWindow(const std::string& s);

    /** Get directory from filename.
        \param[in] filename File path to strip directory from
        \return Stripped directory path
    */
    std::string getDirectoryFromFile(const std::string& filename);

    /** Strip path from a full filename
        \param[in] filename File path
        \return Stripped filename
    */
    std::string getFilenameFromPath(const std::string& filename);

    /** Swap file extension (very simple implementation)
        \param[in] str File name or full path
        \param[in] currentExtension Current extension to look for
        \param[in] newExtension Extension to replace the current with
        \return If end of str matches currentExtension, returns the file name replaced with the new extension, otherwise returns the original file name.
    */
    std::string swapFileExtension(const std::string& str, const std::string& currentExtension, const std::string& newExtension);

    /** Enumerate files using search string
        \param[in] searchString String to use in file search
        \param[out] filenames Vector of found filenames
    */
    void enumerateFiles(std::string searchString, std::vector<std::string>& filenames);
    
    /** Return current thread handle
    */
    std::thread::native_handle_type getCurrentThread();
        
    /** Sets thread affinity mask
    */
    void setThreadAffinity(std::thread::native_handle_type thread, uint32_t affinityMask);

    /** Get the last time a file was modified. If the file is not found will return 0
        \param[in] filename The file to look for
        \return Epoch timestamp of when the file was last modified
    */
    time_t getFileModifiedTime(const std::string& filename);

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
    void setThreadPriority(std::thread::native_handle_type thread, ThreadPriorityType priority);

    /** Get the Total Virtual Memory.
    */
    uint64_t getTotalVirtualMemory();

    /** Get the Used Virtual Memory.
    */
    uint64_t getUsedVirtualMemory();

    /** Get the Virtual Memory Used by this Process.
    */
    uint64_t  getProcessUsedVirtualMemory();

    /** Returns index of most significant set bit, or 0 if no bits were set.
    */
    uint32_t bitScanReverse(uint32_t a);

    /** Returns index of least significant set bit, or 0 if no bits were set.
    */
    uint32_t bitScanForward(uint32_t a);

    /** Gets the closest power of two to a number, rounded down.
    */
    uint32_t getLowerPowerOf2(uint32_t a);

    /** Gets the number of set bits.
    */
    uint32_t popcount(uint32_t a);

    /*! @} */
};
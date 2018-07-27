/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "Falcor.h"
#include "RenderGraphLiveEditor.h"
#include "RenderGraphLoader.h"
#include <string>

namespace Falcor
{
    const char* kEditorExecutableName =
#ifdef _WIN32
        "RenderGraphEditor.exe";
#else
        "./RenderGraphEditor";
#endif
    const char* kViewerExecutableName =
#ifdef _WIN32
        "RenderGraphViewer.exe";
#else
        "./RenderGraphViewer";
#endif

    const float kCheckFileInterval = 0.5f;

    RenderGraphLiveEditor::RenderGraphLiveEditor()
    {

    }

    void RenderGraphLiveEditor::openUpdatesFile(const std::string filePath)
    {
        mUpdatesFile.open(filePath);
        if (!mUpdatesFile.is_open())
        {
            logError("Failed to open temporary file for render graph viewer.");
        }

        if ((mUpdatesFile.rdstate() & std::ifstream::failbit) != 0)
        {
            logError("Unable to read temporary file for render graph viewer.");
        }

        mTempFileName = filePath;

        mIsOpen = true;
    }

    bool RenderGraphLiveEditor::createUpdateFile(const RenderGraph& renderGraph)
    {
        // create a new temporary file through windows
        mTempFileNameW.resize(510, '0');
        mTempFilePathW.resize(510, '0');
        mTempFileName.resize(255, '0');
        mTempFilePath.resize(255, '0');
        
        std::string renderGraphScript = RenderGraphLoader::saveRenderGraphAsScriptBuffer(renderGraph);
        if (!renderGraphScript.size())
        {
            logError("No graph data to display in editor.");
            return false;
        }

#ifdef _WIN32
        GetTempPath(255, (LPWSTR)(&mTempFilePathW.front()));
        GetTempFileName((LPCWSTR)mTempFilePathW.c_str(), L"PW", 0, (LPWSTR)&mTempFileNameW.front());
        wcstombs(&mTempFileName.front(), (wchar_t*)mTempFileNameW.c_str(), mTempFileNameW.size());
        wcstombs(&mTempFilePath.front(), (wchar_t*)mTempFilePathW.c_str(), mTempFilePathW.size());
#endif
        
        std::ofstream updatesFileOut(mTempFileName);
        assert(updatesFileOut.is_open());
        updatesFileOut.write(renderGraphScript.c_str(), renderGraphScript.size());
        updatesFileOut.close();

        openUpdatesFile(mTempFileName);
        mLastWriteTime = getFileModifiedTime(mTempFileName);

        return true;
    }

    bool RenderGraphLiveEditor::open(const std::string& commandLine)
    {
#ifdef _WIN32
        STARTUPINFOA startupInfo{}; PROCESS_INFORMATION processInformation{};
        if (!CreateProcessA(nullptr, (LPSTR)commandLine.c_str(), nullptr, nullptr, TRUE, NORMAL_PRIORITY_CLASS, nullptr, nullptr, &startupInfo, &processInformation))
        {
            logError("Unable to execute the render graph editor");
            close();
            return false;
        }

        mProcess = processInformation.hProcess;
#endif
        mIsOpen = true;
        return true;
    }

    void RenderGraphLiveEditor::openViewer(const RenderGraph& renderGraph)
    {
        if (mIsOpen)
        {
            logWarning("Render Graph Editor is already open for this graph!");
            return;
        }

        if (!createUpdateFile(renderGraph)) return;

        std::string commandLine = kViewerExecutableName;
        commandLine += std::string(" ") + mTempFileName;
        
        if (!open(commandLine)) return;
    }

    void RenderGraphLiveEditor::openEditor(const RenderGraph& renderGraph)
    {
        if (mIsOpen)
        {
            logWarning("Render Graph Editor is already open for this graph!");
            return;
        }

        // create mapped memory and launch editor process
        if (!createUpdateFile(renderGraph)) return;

        // load application for the editor given it the name of the mapped file
        std::string commandLine = kEditorExecutableName;
        commandLine += std::string(" ") + mTempFileName;
        
        if (!open(commandLine)) return;
    }

    void RenderGraphLiveEditor::forceUpdateGraph(RenderGraph& renderGraph)
    {
        // load changes from the modified graph file
        mSharedMemoryStage = std::string((std::istreambuf_iterator<char>(mUpdatesFile)), std::istreambuf_iterator<char>());
        mUpdatesFile.seekg(0, std::ios::beg);
        
        RenderGraphLoader::runScript(mSharedMemoryStage.data() + sizeof(size_t), *reinterpret_cast<const size_t*>(mSharedMemoryStage.data()), renderGraph);
    }

    void RenderGraphLiveEditor::updateGraph(RenderGraph& renderGraph, float lastFrameTime)
    {
        bool status = false;

        // check to see if the editor has closed and react accordingly
#ifdef _WIN32
        uint32_t exitCode = 0;
        if (GetExitCodeProcess(mProcess, (LPDWORD)&exitCode))
        {
            if (exitCode != STILL_ACTIVE)
            {
                CloseHandle(mProcess);
                mProcess = nullptr;
                mIsOpen = false;
                mUpdatesFile.close();
                return;
            }
        }
#endif
        if ((mTimeSinceLastCheck += lastFrameTime) > kCheckFileInterval)
        {
            time_t lastWriteTime = getFileModifiedTime(mTempFileName);
            mTimeSinceLastCheck = 0.0f;
            if (mLastWriteTime < lastWriteTime)
            {
                mLastWriteTime = lastWriteTime;
                forceUpdateGraph(renderGraph);
            }
        }
    }

    RenderGraphLiveEditor::~RenderGraphLiveEditor()
    {
        if (mIsOpen) { close(); }
    }

    void RenderGraphLiveEditor::close()
    {
#ifdef _WIN32
        if (mProcess)
        {
            TerminateProcess(mProcess, 0);
            CloseHandle(mProcess);
            mProcess = nullptr;
        }
#endif
        mSharedMemoryStage.clear();
        mIsOpen = false;
        // delete temporary file
    }
}

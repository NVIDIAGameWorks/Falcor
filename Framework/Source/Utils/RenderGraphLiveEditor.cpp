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
    const char* kEditorExecutableName = "RenderGraphEditor";
    const char* kViewerExecutableName = "RenderGraphViewer";
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

        mTempFilePath = filePath;
        mIsOpen = true;
        mLastWriteTime = getFileModifiedTime(mTempFilePath);
    }

    bool RenderGraphLiveEditor::createUpdateFile(const RenderGraph& renderGraph)
    {
        std::string renderGraphScript = RenderGraphLoader::saveRenderGraphAsScriptBuffer(renderGraph);
        if (!renderGraphScript.size())
        {
            logError("No graph data to display in editor.");
            return false;
        }

        mTempFilePath = getNewTempFilePath();

        std::ofstream updatesFileOut(mTempFilePath);
        assert(updatesFileOut.is_open());
        updatesFileOut.write(renderGraphScript.c_str(), renderGraphScript.size());
        updatesFileOut.close();

        openUpdatesFile(mTempFilePath);
        mLastWriteTime = getFileModifiedTime(mTempFilePath);

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

        std::string commandLine = std::string("-tempFile ") + mTempFilePath;
        
        mProcess = executeProcess(kViewerExecutableName, commandLine);
        if(mProcess) mIsOpen = true;
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
        std::string commandLine = std::string("-tempFile ") + mTempFilePath;

        mProcess = executeProcess(kEditorExecutableName, commandLine);
        if (mProcess) mIsOpen = true;
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
        if(!isProcessRunning(mProcess))
        {
            CloseHandle((HANDLE)mProcess);
            mProcess = 0;
            mIsOpen = false;
            mUpdatesFile.close();
            return;
        }

        if ((mTimeSinceLastCheck += lastFrameTime) > kCheckFileInterval)
        {
            time_t lastWriteTime = getFileModifiedTime(mTempFilePath);
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
        // terminate process
        if (mProcess)
        {
            terminateProcess(mProcess);
            mProcess = 0;
        }

        mSharedMemoryStage.clear();
        mIsOpen = false;
        // delete temporary file
    }
}

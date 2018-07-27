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
#pragma once
#include "Falcor.h"
#include <fstream>

namespace Falcor
{
    class RenderGraphLiveEditor
    {
    public:
        RenderGraphLiveEditor();
        ~RenderGraphLiveEditor();

        bool isOpen() { return mIsOpen; }
        void openUpdatesFile(const std::string filePath);
        void openEditor(const RenderGraph& renderGraph);
        void openViewer(const RenderGraph& renderGraph);
        void close();
        void updateGraph(RenderGraph& renderGraph, float lastFrameTime = 1.0f / 60.0f);
        void forceUpdateGraph(RenderGraph& renderGraph);
        const std::string& getTempFilePath() { return mTempFileName; }

    private:
        bool createUpdateFile(const RenderGraph& renderGraph);
        bool open(const std::string& commandLine);

        bool mIsOpen = false;
        std::string mSharedMemoryStage;
        std::ifstream mUpdatesFile;
#ifdef _WIN32
        HANDLE mProcess;
        std::string mTempFileNameW;
        std::string mTempFilePathW;
#endif
        time_t mLastWriteTime;
        std::string mTempFileName;
        std::string mTempFilePath;
        float mTimeSinceLastCheck = 0.0f;
    };
}

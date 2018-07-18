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
#include "Falcor.h"
#include "RenderGraphLiveEditor.h"
#include <string>

namespace Falcor
{
    const char* kEditorExecutableName =
#ifdef _WIN32
        "RenderGraphRenderer.exe";
#else
        "";
#endif

    RenderGraphLiveEditor::RenderGraphLiveEditor()
    {

    }

    void RenderGraphLiveEditor::openEditor()
    {
        // create mapped memory
#ifdef _WIN32

        // create a new temporary file through windows
        std::string tempFileName; tempFileName.resize(255);
        std::string tempFilePath; tempFilePath.resize(255);

        GetTempPath(255, (LPWSTR)(&tempFilePath.front()));
        GetTempFileName((LPCWSTR)tempFilePath.c_str(), L"PW", 0, (LPWSTR)&tempFileName.front());
        HANDLE tempFileHndl = CreateFile((LPCWSTR)tempFileName.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, NULL);
        if (!tempFileHndl)
        {
            logError("Unable to create temporary file for graph editor");
            return;
        }

        // map file so we can do things with it
        HANDLE tempFileMappingHndl = CreateFileMapping(tempFileHndl, NULL, PAGE_READWRITE, 0, 0x01400000, NULL);

        if (!tempFileMappingHndl)
        {
            logError("Unable to map temporary file for graph editor");
            if (tempFileHndl) { CloseHandle(tempFileHndl); }
            return;
        }

        // load application for the editor given it the name of the mapped file
        std::string commandLine = kEditorExecutableName;
        commandLine += " " + tempFileName;
        if (WinExec((LPSTR)commandLine.c_str(), 1) < 31)
        {
            logError("Unable to execute the render graph editor");
            return;
        }

#endif
    }

    void RenderGraphLiveEditor::closeEditor()
    {

    }
}

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
#include "Utils/Platform/ProgressBar.h"
#include <CommCtrl.h>
#include <random>

#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

namespace Falcor
{
    struct ProgressBarData
    {
        HWND hwnd;
        std::random_device rd;
        std::mt19937 rng;
        std::uniform_int_distribution<int> dist;
        std::thread thread;
        bool running = true;
    };

    ProgressBar::~ProgressBar()
    {
        mpData->running = false;
        mpData->thread.join();
        DestroyWindow(mpData->hwnd);
        safe_delete(mpData);
    }

    void progressBarThread(ProgressBarData* pData, const ProgressBar::MessageList& msgList, uint32_t delayInMs)
    {
        if(delayInMs)
        {
            Sleep(delayInMs);
        }
        if (pData->running == false) return;

        // Create the window
        int w = 200;
        int h = 60;
        int x = (GetSystemMetrics(SM_CXSCREEN) - w) / 2;
        int y = (GetSystemMetrics(SM_CYSCREEN) - h) / 2;
        pData->hwnd = CreateWindowEx(0, PROGRESS_CLASS, nullptr, WS_VISIBLE | PBS_MARQUEE, x, y, w, h, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);

        if (msgList.size())
        {
            SetWindowTextA(pData->hwnd, msgList[0].c_str());
            // Initialize the random-number generator
            pData->rng = std::mt19937(pData->rd());
            pData->dist = std::uniform_int_distribution<int>(0, (int)msgList.size() - 1);
        }
        else
        {
            SetWindowTextA(pData->hwnd, "Loading...");
        }
        SetForegroundWindow(pData->hwnd);
        setWindowIcon("Framework\\Nvidia.ico", pData->hwnd);

        // Execute
        int j = 0;
        while (pData->running)
        {
            SendMessage(pData->hwnd, PBM_STEPIT, 0, 0);
            SendMessage(pData->hwnd, WM_PAINT, 0, 0);
            Sleep(50);
            if (j == 50 && msgList.size())
            {
                j = 0;
                SetWindowTextA(pData->hwnd, msgList[pData->dist(pData->rng)].c_str());
            }
            MSG msg;
            while (PeekMessage(&msg, pData->hwnd, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }

            j++;
        }
    }

    void ProgressBar::platformInit(const MessageList& list, uint32_t delayInMs)
    {
        mpData = new ProgressBarData;

        // Initialize the common controls
        INITCOMMONCONTROLSEX init;
        init.dwSize = sizeof(INITCOMMONCONTROLSEX);
        init.dwICC = ICC_PROGRESS_CLASS;
        InitCommonControlsEx(&init);

        // Start the thread
        mpData->thread = std::thread(progressBarThread, mpData, list, delayInMs);

    }
}

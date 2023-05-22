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
#include "../ProgressBar.h"
#include "../OS.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <CommCtrl.h>
#include <random>
#include <thread>

namespace Falcor
{
struct ProgressBar::Window
{
    bool running;
    std::thread thread;

    Window(const std::string& msg)
    {
        running = true;
        thread = std::thread(threadFunc, this, msg);
    }

    ~Window()
    {
        running = false;
        thread.join();
    }

    static void threadFunc(ProgressBar::Window* pThis, std::string msgText)
    {
        // Create the window
        int w = 200;
        int h = 60;
        int x = (GetSystemMetrics(SM_CXSCREEN) - w) / 2;
        int y = (GetSystemMetrics(SM_CYSCREEN) - h) / 2;
        HWND hwnd = CreateWindowEx(
            0, PROGRESS_CLASS, nullptr, WS_VISIBLE | PBS_MARQUEE, x, y, w, h, nullptr, nullptr, GetModuleHandle(nullptr), nullptr
        );

        SetWindowTextA(hwnd, msgText.c_str());
        SetForegroundWindow(hwnd);
        setWindowIcon(getRuntimeDirectory() / "data/framework/nvidia.ico", hwnd);

        // Execute
        while (pThis->running)
        {
            SendMessage(hwnd, PBM_STEPIT, 0, 0);
            SendMessage(hwnd, WM_PAINT, 0, 0);
            Sleep(50);
            MSG msg;
            while (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }

        DestroyWindow(hwnd);
    }
};

ProgressBar::ProgressBar()
{
    // Initialize the common controls
    INITCOMMONCONTROLSEX init;
    init.dwSize = sizeof(INITCOMMONCONTROLSEX);
    init.dwICC = ICC_PROGRESS_CLASS;
    InitCommonControlsEx(&init);
}

ProgressBar::~ProgressBar()
{
    close();
}

void ProgressBar::show(const std::string& msg)
{
    close();
    mpWindow = std::make_unique<Window>(msg);
}

void ProgressBar::close()
{
    mpWindow.reset();
}
} // namespace Falcor

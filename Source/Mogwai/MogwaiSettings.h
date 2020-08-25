/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Mogwai.h"

namespace Mogwai
{
    class Renderer;

    class MogwaiSettings : public Extension
    {
    public:
        using UniquePtr = std::unique_ptr<Extension>;
        static UniquePtr create(Renderer* pRenderer);

        void renderUI(Gui* pGui) override;
        bool mouseEvent(const MouseEvent& e) override;
        bool keyboardEvent(const KeyboardEvent& e) override;

    private:
        MogwaiSettings(Renderer* pRenderer) : Extension(pRenderer, "Settings") {}

        void mainMenu(Gui* pGui);
        void graphs(Gui* pGui);
        void timeSettings(Gui* pGui);
        void windowSettings(Gui* pGui);
        void exitIfNeeded();

        bool mAutoHideMenu = false;
        bool mShowFps = true;
        bool mShowGraphUI = true;
        bool mShowConsole = false;
        bool mShowTime = false;
        bool mShowWinSize = false;
        uint2 mMousePosition;
    };
}

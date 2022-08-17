/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/Macros.h"
#include <memory>
#include <string>
#include <vector>

namespace Falcor
{
    struct ProgressBarData;

    /** Creates a progress bar visual and manages a new thread for it.
    */
    class FALCOR_API ProgressBar
    {
    public:
        using SharedPtr = std::shared_ptr<ProgressBar>;
        using MessageList = std::vector<std::string>;
        ~ProgressBar();

        /** Creates a progress bar.
            \param[in] list List of messages to display on the progress bar
            \param[in] delayInMs Time between updates in milliseconds
        */
        static SharedPtr show(const MessageList& list, uint32_t delayInMs = 1000);

        /** Creates a progress bar.
            \param[in] pMsg Message to display on the progress bar
            \param[in] delayInMs Time between updates in milliseconds
        */
        static SharedPtr show(const char* pMsg = nullptr, uint32_t delayInMs = 1000);

        /** Close the progress bar
        */
        static void close();

        /** Check if the progress bar is currently active.
            \return Returns true if progress bar is active.
        */
        static bool isActive();

    private:
        static std::weak_ptr<ProgressBar> spBar;
        static std::unique_ptr<ProgressBarData> spData;
        ProgressBar();
        void platformInit(const MessageList& list, uint32_t delayInMs);
    };
}

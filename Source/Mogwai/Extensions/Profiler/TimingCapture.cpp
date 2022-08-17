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
#include "Falcor.h"
#include "TimingCapture.h"

namespace Mogwai
{
    namespace
    {
        const std::string kScriptVar = "timingCapture";
        const std::string kCaptureFrameTime = "captureFrameTime";
    }

    MOGWAI_EXTENSION(TimingCapture);

    TimingCapture::UniquePtr TimingCapture::create(Renderer* pRenderer)
    {
        return UniquePtr(new TimingCapture(pRenderer));
    }

    void TimingCapture::registerScriptBindings(pybind11::module& m)
    {
        using namespace pybind11::literals;

        pybind11::class_<TimingCapture> timingCapture(m, "TimingCapture");

        // Members
        timingCapture.def(kCaptureFrameTime.c_str(), &TimingCapture::captureFrameTime, "path"_a);
    }

    std::string TimingCapture::getScriptVar() const
    {
        return kScriptVar;
    }

    void TimingCapture::beginFrame(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
    {
        recordPreviousFrameTime();
    }

    void TimingCapture::captureFrameTime(std::filesystem::path path)
    {
        if (mFrameTimeFile.is_open())
            mFrameTimeFile.close();

        if (!path.empty())
        {
            if (std::filesystem::exists(path))
            {
                logWarning("Frame times in file '{}' will be overwritten.", path);
            }

            mFrameTimeFile.open(path, std::ofstream::trunc);
            if (!mFrameTimeFile.is_open())
            {
                logError("Failed to open file '{}' for writing. Ignoring call.", path);
            }
        }
    }

    void TimingCapture::recordPreviousFrameTime()
    {
        if (!mFrameTimeFile.is_open()) return;

        // The FrameRate object is updated at the start of each frame, the first valid time is available on the second frame.
        auto& frameRate = gpFramework->getFrameRate();
        if (frameRate.getFrameCount() > 1)
            mFrameTimeFile << frameRate.getLastFrameTime() << std::endl;
    }
}

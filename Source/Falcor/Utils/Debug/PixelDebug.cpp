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
#include "PixelDebug.h"
#include "Core/API/RenderContext.h"
#include "Core/Program/ComputeProgram.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Logger.h"
#include "Utils/UI/InputTypes.h"
#include <fstd/bit.h> // TODO: Replace with C++20 <bit> when available on all targets
#include <sstream>
#include <iomanip>

namespace Falcor
{
    namespace
    {
        static_assert(sizeof(PixelLogValue) % 16 == 0, "PixelLogValue size should be a multiple of 16B");

        const char kReflectPixelDebugTypesFile[] = "Utils/Debug/ReflectPixelDebugTypes.cs.slang";
    }

    PixelDebug::SharedPtr PixelDebug::create(uint32_t logSize)
    {
        return SharedPtr(new PixelDebug(logSize));
    }

    void PixelDebug::beginFrame(RenderContext* pRenderContext, const uint2& frameDim)
    {
        mFrameDim = frameDim;
        if (mRunning)
        {
            throw RuntimeError("PixelDebug::beginFrame() - Logging is already running, did you forget to call endFrame()?");
        }
        mRunning = true;

        // Reset previous data.
        mPixelLogData.clear();
        mAssertLogData.clear();
        mDataValid = false;
        mWaitingForData = false;

        if (mEnabled)
        {
            // Prepare log buffers.
            if (!mpPixelLog || mpPixelLog->getElementCount() != mLogSize)
            {
                // Create program for type reflection.
                if (!mpReflectProgram) mpReflectProgram = ComputeProgram::createFromFile(kReflectPixelDebugTypesFile, "main");

                // Allocate GPU buffers.
                mpPixelLog = Buffer::createStructured(mpReflectProgram.get(), "gPixelLog", mLogSize);
                if (mpPixelLog->getStructSize() != sizeof(PixelLogValue)) throw RuntimeError("Struct PixelLogValue size mismatch between CPU/GPU");

                mpAssertLog = Buffer::createStructured(mpReflectProgram.get(), "gAssertLog", mLogSize);
                if (mpAssertLog->getStructSize() != sizeof(AssertLogValue)) throw RuntimeError("Struct AssertLogValue size mismatch between CPU/GPU");

                // Allocate staging buffers for readback. These are shared, the data is stored consecutively.
                mpCounterBuffer = Buffer::create(2 * sizeof(uint32_t), ResourceBindFlags::None, Buffer::CpuAccess::Read);
                mpDataBuffer = Buffer::create(mpPixelLog->getSize() + mpAssertLog->getSize(), ResourceBindFlags::None, Buffer::CpuAccess::Read);
            }

            pRenderContext->clearUAVCounter(mpPixelLog, 0);
            pRenderContext->clearUAVCounter(mpAssertLog, 0);
        }
    }

    void PixelDebug::endFrame(RenderContext* pRenderContext)
    {
        if (!mRunning)
        {
            throw RuntimeError("PixelDebug::endFrame() - Logging is not running, did you forget to call beginFrame()?");
        }
        mRunning = false;

        if (mEnabled)
        {
            // Copy logged data to staging buffers.
            pRenderContext->copyBufferRegion(mpCounterBuffer.get(), 0, mpPixelLog->getUAVCounter().get(), 0, 4);
            pRenderContext->copyBufferRegion(mpCounterBuffer.get(), 4, mpAssertLog->getUAVCounter().get(), 0, 4);
            pRenderContext->copyBufferRegion(mpDataBuffer.get(), 0, mpPixelLog.get(), 0, mpPixelLog->getSize());
            pRenderContext->copyBufferRegion(mpDataBuffer.get(), mpPixelLog->getSize(), mpAssertLog.get(), 0, mpAssertLog->getSize());

            // Create fence first time we need it.
            if (!mpFence) mpFence = GpuFence::create();

            // Submit command list and insert signal.
            pRenderContext->flush(false);
            mpFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

            mWaitingForData = true;
        }
    }

    void PixelDebug::prepareProgram(const Program::SharedPtr& pProgram, const ShaderVar& var)
    {
        FALCOR_ASSERT(mRunning);

        if (mEnabled)
        {
            pProgram->addDefine("_PIXEL_DEBUG_ENABLED");
            var["gPixelLog"] = mpPixelLog;
            var["gAssertLog"] = mpAssertLog;
            var["PixelDebugCB"]["gPixelLogSelected"] = mSelectedPixel;
            var["PixelDebugCB"]["gPixelLogSize"] = mLogSize;
            var["PixelDebugCB"]["gAssertLogSize"] = mLogSize;

            const auto &hashedStrings = pProgram->getReflector()->getHashedStrings();
            for (const auto& hashedString : hashedStrings)
            {
                mHashToString.insert(std::make_pair(hashedString.hash, hashedString.string));
            }
        }
        else
        {
            pProgram->removeDefine("_PIXEL_DEBUG_ENABLED");
        }
    }

    void PixelDebug::renderUI(Gui::Widgets* widget)
    {
        if (mRunning)
        {
            throw RuntimeError("PixelDebug::renderUI() - Logging is running, call end() before renderUI().");
        }

        if (widget)
        {
            // Configure logging.
            widget->checkbox("Pixel debug", mEnabled);
            widget->tooltip("Enables shader debugging.\n\n"
                "Left-mouse click on a pixel to select it.\n"
                "Use print(value) or print(msg, value) in the shader to print values of basic types (int, float2, etc.) for the selected pixel.\n"
                "Use assert(condition) or assert(condition, msg) in the shader to test a condition.", true);
            if (mEnabled)
            {
                widget->var("Selected pixel", mSelectedPixel);
            }
        }

        // Fetch stats and show log if available.
        bool isNewData = copyDataToCPU();
        if (mDataValid)
        {
            std::ostringstream oss;

            // Print list of printed values.
            oss << "Pixel log:" << (mPixelLogData.empty() ? " <empty>\n" : "\n");
            for (auto v : mPixelLogData)
            {
                // Print message.
                auto it = mHashToString.find(v.msgHash);
                if (it != mHashToString.end() && !it->second.empty()) oss << it->second << " ";

                // Parse value and convert to string.
                if (v.count > 1) oss << "(";
                for (uint32_t i = 0; i < v.count; i++)
                {
                    uint32_t bits = v.data[i];
                    switch ((PixelLogValueType)v.type)
                    {
                    case PixelLogValueType::Bool:
                        oss << (bits != 0 ? "true" : "false");
                        break;
                    case PixelLogValueType::Int:
                        oss << (int32_t)bits;
                        break;
                    case PixelLogValueType::Uint:
                        oss << bits;
                        break;
                    case PixelLogValueType::Float:
                        oss << fstd::bit_cast<float>(bits);
                        break;
                    default:
                        oss << "INVALID VALUE";
                        break;
                    }
                    if (i + 1 < v.count) oss << ", ";
                }
                if (v.count > 1) oss << ")";
                oss << "\n";
            }

            // Print list of asserts.
            if (!mAssertLogData.empty())
            {
                oss << "\n";
                for (auto v : mAssertLogData)
                {
                    oss << "Assert at (" << v.launchIndex.x << ", " << v.launchIndex.y << ", " << v.launchIndex.z << ")";
                    auto it = mHashToString.find(v.msgHash);
                    if (it != mHashToString.end() && !it->second.empty()) oss << " " << it->second;
                    oss << "\n";
                }
            }

            if( widget )
                widget->text(oss.str());

            bool isEmpty = mPixelLogData.empty() && mAssertLogData.empty();
            if (isNewData && !isEmpty) logInfo("\n" + oss.str());
        }
    }

    bool PixelDebug::onMouseEvent(const MouseEvent& mouseEvent)
    {
        if (mEnabled)
        {
            if (mouseEvent.type == MouseEvent::Type::ButtonDown && mouseEvent.button == Input::MouseButton::Left)
            {
                mSelectedPixel = uint2(mouseEvent.pos * float2(mFrameDim));
                return true;
            }
        }
        return false;
    }

    bool PixelDebug::copyDataToCPU()
    {
        FALCOR_ASSERT(!mRunning);
        if (mWaitingForData)
        {
            // Wait for signal.
            mpFence->syncCpu();
            mWaitingForData = false;

            if (mEnabled)
            {
                // Map counter buffer. This tells us how many print() and assert() calls were made.
                uint32_t* uavCounters = (uint32_t*)mpCounterBuffer->map(Buffer::MapType::Read);
                const uint32_t printCount = std::min(mpPixelLog->getElementCount(), uavCounters[0]);
                const uint32_t assertCount = std::min(mpAssertLog->getElementCount(), uavCounters[1]);
                mpCounterBuffer->unmap();

                // Map the data buffer and copy the relevant sections.
                uint8_t* pLog = (uint8_t*)mpDataBuffer->map(Buffer::MapType::Read);

                mPixelLogData.resize(printCount);
                for (uint32_t i = 0; i < printCount; i++) mPixelLogData[i] = ((PixelLogValue*)pLog)[i];
                pLog += mpPixelLog->getSize();

                mAssertLogData.resize(assertCount);
                for (uint32_t i = 0; i < assertCount; i++) mAssertLogData[i] = ((AssertLogValue*)pLog)[i];

                mpDataBuffer->unmap();
                mDataValid = true;
                return true;
            }
        }

        return false;
    }

}

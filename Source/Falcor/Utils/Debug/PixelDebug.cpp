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
#include "PixelDebug.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Core/Program/Program.h"
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
static_assert(sizeof(PrintRecord) % 16 == 0, "PrintRecord size should be a multiple of 16B");
} // namespace

PixelDebug::PixelDebug(ref<Device> pDevice, uint32_t printCapacity, uint32_t assertCapacity)
    : mpDevice(pDevice), mPrintCapacity(printCapacity), mAssertCapacity(assertCapacity)
{}

void PixelDebug::beginFrame(RenderContext* pRenderContext, const uint2& frameDim)
{
    FALCOR_CHECK(!mRunning, "Logging is already running, did you forget to call endFrame()?");

    mFrameDim = frameDim;
    mRunning = true;

    // Reset previous data.
    mPrintData.clear();
    mAssertData.clear();
    mDataValid = false;
    mWaitingForData = false;

    if (mEnabled)
    {
        // Prepare buffers.
        if (!mpPrintBuffer)
        {
            // Allocate GPU buffers.
            const ref<Device>& pDevice = pRenderContext->getDevice();
            mpCounterBuffer = pDevice->createBuffer(sizeof(uint32_t) * 2);
            mpPrintBuffer = pDevice->createStructuredBuffer(sizeof(PrintRecord), mPrintCapacity);
            mpAssertBuffer = pDevice->createStructuredBuffer(sizeof(AssertRecord), mAssertCapacity);

            // Allocate readback buffer. This buffer is shared for copying all the above buffers to the CPU
            mpReadbackBuffer = pDevice->createBuffer(
                mpCounterBuffer->getSize() + mpPrintBuffer->getSize() + mpAssertBuffer->getSize(),
                ResourceBindFlags::None,
                MemoryType::ReadBack
            );
        }

        pRenderContext->clearUAV(mpCounterBuffer->getUAV().get(), uint4(0));
    }
}

void PixelDebug::endFrame(RenderContext* pRenderContext)
{
    FALCOR_CHECK(mRunning, "Logging is not running, did you forget to call beginFrame()?");

    mRunning = false;

    if (mEnabled)
    {
        // Copy logged data to staging buffers.
        uint32_t dst = 0;
        pRenderContext->copyBufferRegion(mpReadbackBuffer.get(), dst, mpCounterBuffer.get(), 0, mpCounterBuffer->getSize());
        dst += mpCounterBuffer->getSize();
        pRenderContext->copyBufferRegion(mpReadbackBuffer.get(), dst, mpPrintBuffer.get(), 0, mpPrintBuffer->getSize());
        dst += mpPrintBuffer->getSize();
        pRenderContext->copyBufferRegion(mpReadbackBuffer.get(), dst, mpAssertBuffer.get(), 0, mpAssertBuffer->getSize());
        dst += mpAssertBuffer->getSize();
        FALCOR_ASSERT(dst == mpReadbackBuffer->getSize());

        // Create fence first time we need it.
        if (!mpFence)
            mpFence = mpDevice->createFence();

        // Submit command list and insert signal.
        pRenderContext->submit(false);
        pRenderContext->signal(mpFence.get());

        mWaitingForData = true;
    }
}

void PixelDebug::prepareProgram(const ref<Program>& pProgram, const ShaderVar& var)
{
    FALCOR_CHECK(mRunning, "Logging is not running, did you forget to call beginFrame()?");

    if (mEnabled)
    {
        pProgram->addDefine("_PIXEL_DEBUG_ENABLED");

        ShaderVar pixelDebug = var["gPixelDebug"];
        pixelDebug["counterBuffer"] = mpCounterBuffer;
        pixelDebug["printBuffer"] = mpPrintBuffer;
        pixelDebug["assertBuffer"] = mpAssertBuffer;
        pixelDebug["printBufferCapacity"] = mPrintCapacity;
        pixelDebug["assertBufferCapacity"] = mAssertCapacity;
        pixelDebug["selectedPixel"] = mSelectedPixel;

        const auto& hashedStrings = pProgram->getReflector()->getHashedStrings();
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
    FALCOR_CHECK(!mRunning, "Logging is running, call endFrame() before renderUI().");

    if (widget)
    {
        // Configure logging.
        widget->checkbox("Pixel debug", mEnabled);
        widget->tooltip(
            "Enables shader debugging.\n\n"
            "Left-mouse click on a pixel to select it.\n"
            "Use print(value) or print(msg, value) in the shader to print values for the selected pixel.\n"
            "All basic types such as int, float2, etc. are supported.\n"
            "Use assert(condition) or assert(condition, msg) in the shader to test a condition.",
            true
        );
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
        oss << "Pixel log:" << (mPrintData.empty() ? " <empty>\n" : "\n");
        for (auto v : mPrintData)
        {
            // Print message.
            auto it = mHashToString.find(v.msgHash);
            if (it != mHashToString.end() && !it->second.empty())
                oss << it->second << " ";

            // Parse value and convert to string.
            if (v.count > 1)
                oss << "(";
            for (uint32_t i = 0; i < v.count; i++)
            {
                uint32_t bits = v.data[i];
                switch ((PrintValueType)v.type)
                {
                case PrintValueType::Bool:
                    oss << (bits != 0 ? "true" : "false");
                    break;
                case PrintValueType::Int:
                    oss << (int32_t)bits;
                    break;
                case PrintValueType::Uint:
                    oss << bits;
                    break;
                case PrintValueType::Float:
                    oss << fstd::bit_cast<float>(bits);
                    break;
                default:
                    oss << "INVALID VALUE";
                    break;
                }
                if (i + 1 < v.count)
                    oss << ", ";
            }
            if (v.count > 1)
                oss << ")";
            oss << "\n";
        }

        // Print list of asserts.
        if (!mAssertData.empty())
        {
            oss << "\n";
            for (auto v : mAssertData)
            {
                oss << "Assert at (" << v.launchIndex.x << ", " << v.launchIndex.y << ", " << v.launchIndex.z << ")";
                auto it = mHashToString.find(v.msgHash);
                if (it != mHashToString.end() && !it->second.empty())
                    oss << " " << it->second;
                oss << "\n";
            }
        }

        if (widget)
            widget->text(oss.str());

        bool isEmpty = mPrintData.empty() && mAssertData.empty();
        if (isNewData && !isEmpty)
            logInfo("\n" + oss.str());
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
        mpFence->wait();
        mWaitingForData = false;

        if (mEnabled)
        {
            // Copy data from readback buffer to CPU buffers.
            const uint8_t* data = reinterpret_cast<const uint8_t*>(mpReadbackBuffer->map(Buffer::MapType::Read));
            const uint32_t* counterData = reinterpret_cast<const uint32_t*>(data);
            data += mpCounterBuffer->getSize();
            const PrintRecord* printData = reinterpret_cast<const PrintRecord*>(data);
            data += mpPrintBuffer->getSize();
            const AssertRecord* assertData = reinterpret_cast<const AssertRecord*>(data);

            const uint32_t printCount = std::min(mpPrintBuffer->getElementCount(), counterData[0]);
            const uint32_t assertCount = std::min(mpAssertBuffer->getElementCount(), counterData[1]);

            mPrintData.assign(printData, printData + printCount);
            mAssertData.assign(assertData, assertData + assertCount);

            mpReadbackBuffer->unmap();
            mDataValid = true;
            return true;
        }
    }

    return false;
}

} // namespace Falcor

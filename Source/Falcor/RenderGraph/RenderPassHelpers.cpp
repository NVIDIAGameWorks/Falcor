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
#include "RenderPassHelpers.h"
#include "RenderPass.h"
#include "Core/Assert.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    uint2 RenderPassHelpers::calculateIOSize(const IOSize selection, const uint2 fixedSize, const uint2 windowSize)
    {
        uint2 sz = {};
        if (selection == RenderPassHelpers::IOSize::Fixed) sz = fixedSize;
        else if (selection == RenderPassHelpers::IOSize::Full) sz = windowSize;
        else if (selection == RenderPassHelpers::IOSize::Half) sz = windowSize / uint2(2);
        else if (selection == RenderPassHelpers::IOSize::Quarter) sz = windowSize / uint2(4);
        else if (selection == RenderPassHelpers::IOSize::Double) sz = windowSize * uint2(2);
        else FALCOR_ASSERT(selection == RenderPassHelpers::IOSize::Default);
        return sz;
    }

    FALCOR_SCRIPT_BINDING(RenderPassHelpers)
    {
        pybind11::enum_<RenderPassHelpers::IOSize> sz(m, "IOSize");
        sz.value("Default", RenderPassHelpers::IOSize::Default);
        sz.value("Fixed", RenderPassHelpers::IOSize::Fixed);
        sz.value("Full", RenderPassHelpers::IOSize::Full);
        sz.value("Half", RenderPassHelpers::IOSize::Half);
        sz.value("Quarter", RenderPassHelpers::IOSize::Quarter);
        sz.value("Double", RenderPassHelpers::IOSize::Double);
    }
}

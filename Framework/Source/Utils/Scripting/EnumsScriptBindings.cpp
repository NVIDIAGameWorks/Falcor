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
#include "Framework.h"
#include "EnumsScriptBindings.h"
#include "Scripting.h"
#include "API/Sampler.h"
#include "Effects/ToneMapping/ToneMapping.h"

namespace Falcor
{
#define val(a) value(to_string(a).c_str(), a)

    static void globalEnums(pybind11::module& m)
    {
        // Resource formats
        auto formats = pybind11::enum_<ResourceFormat>(m, "Format");
        for (uint32_t i = 0; i < (uint32_t)ResourceFormat::Count; i++)
        {
            formats.val(ResourceFormat(i));
        }

        // Comparison mode
        auto comparison = pybind11::enum_<ComparisonFunc>(m, "Comparison");
        comparison.val(ComparisonFunc::Disabled).val(ComparisonFunc::LessEqual).val(ComparisonFunc::GreaterEqual).val(ComparisonFunc::Less).val(ComparisonFunc::Greater);
        comparison.val(ComparisonFunc::Equal).val(ComparisonFunc::NotEqual).val(ComparisonFunc::Always).val(ComparisonFunc::Never);
    }

    static void samplerState(pybind11::module& m)
    {
        auto filter = pybind11::enum_<Sampler::Filter>(m, "Filter");
        filter.val(Sampler::Filter::Linear).val(Sampler::Filter::Point);

        auto addressing = pybind11::enum_<Sampler::AddressMode>(m, "AddressMode");
        addressing.val(Sampler::AddressMode::Wrap).val(Sampler::AddressMode::Mirror).val(Sampler::AddressMode::Clamp).val(Sampler::AddressMode::Border).val(Sampler::AddressMode::MirrorOnce);
    }

    static void toneMapping(pybind11::module& m)
    {
        auto op = pybind11::enum_<ToneMapping::Operator>(m, "ToneMapOp");
        op.val(ToneMapping::Operator::Clamp).val(ToneMapping::Operator::Linear).val(ToneMapping::Operator::Reinhard).val(ToneMapping::Operator::ReinhardModified).val(ToneMapping::Operator::HejiHableAlu);
        op.val(ToneMapping::Operator::HableUc2).val(ToneMapping::Operator::Aces);
    }

    void EnumsScriptBindings::registerScriptingObjects(pybind11::module& m)
    {
        globalEnums(m);
        samplerState(m);
        toneMapping(m);
    }
}
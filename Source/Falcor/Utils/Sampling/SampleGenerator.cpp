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
#include "SampleGenerator.h"

namespace Falcor
{
static std::map<uint32_t, std::function<ref<SampleGenerator>(ref<Device>)>> sFactory;
static Gui::DropdownList sGuiDropdownList;

ref<SampleGenerator> SampleGenerator::create(ref<Device> pDevice, uint32_t type)
{
    if (auto it = sFactory.find(type); it != sFactory.end())
    {
        return it->second(pDevice);
    }
    else
    {
        FALCOR_THROW("Can't create SampleGenerator. Unknown type");
    }
}

DefineList SampleGenerator::getDefines() const
{
    DefineList defines;
    defines.add("SAMPLE_GENERATOR_TYPE", std::to_string(mType));
    return defines;
}

const Gui::DropdownList& SampleGenerator::getGuiDropdownList()
{
    return sGuiDropdownList;
}

void SampleGenerator::registerType(uint32_t type, const std::string& name, std::function<ref<SampleGenerator>(ref<Device>)> createFunc)
{
    sGuiDropdownList.push_back({type, name});
    sFactory[type] = createFunc;
}

void SampleGenerator::registerAll()
{
    registerType(
        SAMPLE_GENERATOR_TINY_UNIFORM,
        "Tiny uniform (32-bit)",
        [](ref<Device> pDevice) { return ref<SampleGenerator>(new SampleGenerator(pDevice, SAMPLE_GENERATOR_TINY_UNIFORM)); }
    );
    registerType(
        SAMPLE_GENERATOR_UNIFORM,
        "Uniform (128-bit)",
        [](ref<Device> pDevice) { return ref<SampleGenerator>(new SampleGenerator(pDevice, SAMPLE_GENERATOR_UNIFORM)); }
    );
}

// Automatically register basic sampler types.
static struct RegisterSampleGenerators
{
    RegisterSampleGenerators() { SampleGenerator::registerAll(); }
} sRegisterSampleGenerators;
} // namespace Falcor

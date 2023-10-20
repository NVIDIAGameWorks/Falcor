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
#include "ComputePass.h"
#include "Core/API/ComputeContext.h"
#include "Core/API/PythonHelpers.h"
#include "Utils/Math/Common.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
ComputePass::ComputePass(ref<Device> pDevice, const ProgramDesc& desc, const DefineList& defines, bool createVars) : mpDevice(pDevice)
{
    auto pProg = Program::create(mpDevice, desc, defines);
    mpState = ComputeState::create(mpDevice);
    mpState->setProgram(pProg);
    if (createVars)
        mpVars = ProgramVars::create(mpDevice, pProg.get());
    FALCOR_ASSERT(pProg && mpState && (!createVars || mpVars));
}

ref<ComputePass> ComputePass::create(
    ref<Device> pDevice,
    const std::filesystem::path& path,
    const std::string& csEntry,
    const DefineList& defines,
    bool createVars
)
{
    ProgramDesc desc;
    desc.addShaderLibrary(path).csEntry(csEntry);
    return create(pDevice, desc, defines, createVars);
}

ref<ComputePass> ComputePass::create(ref<Device> pDevice, const ProgramDesc& desc, const DefineList& defines, bool createVars)
{
    return ref<ComputePass>(new ComputePass(pDevice, desc, defines, createVars));
}

void ComputePass::execute(ComputeContext* pContext, uint32_t nThreadX, uint32_t nThreadY, uint32_t nThreadZ)
{
    FALCOR_ASSERT(mpVars);
    uint3 threadGroupSize = mpState->getProgram()->getReflector()->getThreadGroupSize();
    uint3 groups = div_round_up(uint3(nThreadX, nThreadY, nThreadZ), threadGroupSize);
    pContext->dispatch(mpState.get(), mpVars.get(), groups);
}

void ComputePass::executeIndirect(ComputeContext* pContext, const Buffer* pArgBuffer, uint64_t argBufferOffset)
{
    FALCOR_ASSERT(mpVars);
    pContext->dispatchIndirect(mpState.get(), mpVars.get(), pArgBuffer, argBufferOffset);
}

void ComputePass::addDefine(const std::string& name, const std::string& value, bool updateVars)
{
    mpState->getProgram()->addDefine(name, value);
    if (updateVars)
        mpVars = ProgramVars::create(mpDevice, mpState->getProgram().get());
}

void ComputePass::removeDefine(const std::string& name, bool updateVars)
{
    mpState->getProgram()->removeDefine(name);
    if (updateVars)
        mpVars = ProgramVars::create(mpDevice, mpState->getProgram().get());
}

void ComputePass::setVars(const ref<ProgramVars>& pVars)
{
    mpVars = pVars ? pVars : ProgramVars::create(mpDevice, mpState->getProgram().get());
    FALCOR_ASSERT(mpVars);
}

FALCOR_SCRIPT_BINDING(ComputePass)
{
    using namespace pybind11::literals;

    FALCOR_SCRIPT_BINDING_DEPENDENCY(Device)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(ShaderVar)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(ComputeContext)

    pybind11::class_<ComputePass, ref<ComputePass>> computePass(m, "ComputePass");
    computePass.def(
        pybind11::init(
            [](ref<Device> device, std::optional<ProgramDesc> desc, pybind11::dict defines, const pybind11::kwargs& kwargs)
            {
                if (desc)
                {
                    FALCOR_CHECK(kwargs.empty(), "Either provide a 'desc' or kwargs, but not both.");
                    return ComputePass::create(device, *desc, defineListFromPython(defines));
                }
                else
                {
                    return ComputePass::create(device, programDescFromPython(kwargs), defineListFromPython(defines));
                }
            }
        ),
        "device"_a,
        "desc"_a = std::optional<ProgramDesc>(),
        "defines"_a = pybind11::dict(),
        pybind11::kw_only()
    );

    computePass.def_property_readonly("program", &ComputePass::getProgram);
    computePass.def_property_readonly("root_var", &ComputePass::getRootVar);
    computePass.def_property_readonly("globals", &ComputePass::getRootVar);

    computePass.def(
        "execute",
        [](ComputePass& pass, uint32_t threads_x, uint32_t threads_y, uint32_t threads_z, ComputeContext* compute_context)
        {
            if (compute_context == nullptr)
                compute_context = pass.getDevice()->getRenderContext();
            pass.execute(compute_context, threads_x, threads_y, threads_z);
        },
        "threads_x"_a,
        "threads_y"_a = 1,
        "threads_z"_a = 1,
        "compute_context"_a = nullptr
    );
}

} // namespace Falcor

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
#include "FullScreenPass.h"
#include "Core/API/RenderContext.h"
#include "Utils/SharedCache.h"

namespace Falcor
{
namespace
{
struct Vertex
{
    float2 screenPos;
    float2 texCoord;
};

const Vertex kVertices[] = {
    {float2(-1, 1), float2(0, 0)},
    {float2(-1, -1), float2(0, 1)},
    {float2(1, 1), float2(1, 0)},
    {float2(1, -1), float2(1, 1)},
};
} // namespace

struct FullScreenPass::SharedData
{
    ref<Buffer> pVertexBuffer;
    ref<Vao> pVao;
    uint64_t objectCount = 0;

    SharedData(ref<Device> pDevice)
    {
        const uint32_t vbSize = (uint32_t)(sizeof(Vertex) * std::size(kVertices));
        pVertexBuffer = pDevice->createBuffer(vbSize, ResourceBindFlags::Vertex, MemoryType::Upload, (void*)kVertices);
        pVertexBuffer->breakStrongReferenceToDevice();

        ref<VertexLayout> pLayout = VertexLayout::create();
        ref<VertexBufferLayout> pBufLayout = VertexBufferLayout::create();
        pBufLayout->addElement("POSITION", 0, ResourceFormat::RG32Float, 1, 0);
        pBufLayout->addElement("TEXCOORD", 8, ResourceFormat::RG32Float, 1, 1);
        pLayout->addBufferLayout(0, pBufLayout);

        Vao::BufferVec buffers{pVertexBuffer};
        pVao = Vao::create(Vao::Topology::TriangleStrip, pLayout, buffers);
    }
};

static SharedCache<FullScreenPass::SharedData, Device*> sSharedCache;

FullScreenPass::FullScreenPass(ref<Device> pDevice, const ProgramDesc& progDesc, const DefineList& programDefines)
    : BaseGraphicsPass(pDevice, progDesc, programDefines)
{
    // Get shared VB and VAO.
    mpSharedData = sSharedCache.acquire(mpDevice, [this]() { return std::make_shared<SharedData>(mpDevice); });

    // Create depth stencil state
    FALCOR_ASSERT(mpState);
    auto pDsState = DepthStencilState::create(DepthStencilState::Desc().setDepthEnabled(false));
    mpState->setDepthStencilState(pDsState);

    mpState->setVao(mpSharedData->pVao);
}

FullScreenPass::~FullScreenPass() = default;

ref<FullScreenPass> FullScreenPass::create(ref<Device> pDevice, const ProgramDesc& desc, const DefineList& defines, uint32_t viewportMask)
{
    ProgramDesc d = desc;
    DefineList defs = defines;
    std::string gs;

    if (viewportMask)
    {
        defs.add("_VIEWPORT_MASK", std::to_string(viewportMask));
        defs.add("_OUTPUT_VERTEX_COUNT", std::to_string(3 * popcount(viewportMask)));
        d.addShaderLibrary("Core/Pass/FullScreenPass.gs.slang").gsEntry("main");
    }
    if (!d.hasEntryPoint(ShaderType::Vertex))
        d.addShaderLibrary("Core/Pass/FullScreenPass.vs.slang").vsEntry("main");

    return ref<FullScreenPass>(new FullScreenPass(pDevice, d, defs));
}

ref<FullScreenPass> FullScreenPass::create(
    ref<Device> pDevice,
    const std::filesystem::path& path,
    const DefineList& defines,
    uint32_t viewportMask
)
{
    ProgramDesc desc;
    desc.addShaderLibrary(path).psEntry("main");
    return create(pDevice, desc, defines, viewportMask);
}

void FullScreenPass::execute(RenderContext* pRenderContext, const ref<Fbo>& pFbo, bool autoSetVpSc) const
{
    mpState->setFbo(pFbo, autoSetVpSc);
    pRenderContext->draw(mpState.get(), mpVars.get(), (uint32_t)std::size(kVertices), 0);
}
} // namespace Falcor

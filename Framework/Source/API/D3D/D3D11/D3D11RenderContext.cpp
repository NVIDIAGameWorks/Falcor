/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "API/RenderContext.h"
#include "API/RasterizerState.h"
#include "API/BlendState.h"
#include "API/FBO.h"
#include "API/ProgramVersion.h"
#include "API/VAO.h"
#include "API/ConstantBuffer.h"
#include "API/Buffer.h"
#include "API/StructuredBuffer.h"
#include "glm/gtc/type_ptr.hpp"

namespace Falcor
{
    RenderContext::~RenderContext() = default;

    RenderContext::SharedPtr RenderContext::create()
    {
        SharedPtr pCtx = SharedPtr(new RenderContext(D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE));
        pCtx->mState.pUniformBuffers.assign(D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, nullptr);
        pCtx->mState.pShaderStorageBuffers.assign(D3D11_1_UAV_SLOT_COUNT, nullptr);
        return pCtx;
    }

    void RenderContext::applyDepthStencilState() const
    {
        getD3D11ImmediateContext()->OMSetDepthStencilState(mState.pDsState->getApiHandle(), mState.stencilRef);
    }

    void RenderContext::applyRasterizerState() const
    {
        getD3D11ImmediateContext()->RSSetState(mState.pRastState->getApiHandle());
    }

    void RenderContext::applyBlendState() const
    {
        const auto pBlendState = mState.pBlendState;
        const glm::vec4& blendFactor = pBlendState->getBlendFactor();
        getD3D11ImmediateContext()->OMSetBlendState(pBlendState->getApiHandle(), glm::value_ptr(blendFactor), mState.sampleMask);
    }

    void RenderContext::applyProgram() const
    {
        const auto pProgram = mState.pProgram;
        const Shader* pShader[(uint32_t)ShaderType::Count] = {nullptr};
        if(pProgram)
        {
            for(uint32_t i = 0 ; i < arraysize(pShader) ; i++)
            {
                pShader[i] = pProgram->getShader(ShaderType(i));
            }
        }

        auto pCtx = getD3D11ImmediateContext();
#define set_shader(handleType_, shaderID_, dxFunc_) \
        {                                           \
            handleType_ handle = pShader[(uint32_t)shaderID_] ? pShader[(uint32_t)shaderID_]->getApiHandle<handleType_>() : nullptr;    \
            pCtx->dxFunc_(handle, nullptr, 0);                                                                   \
        }

        set_shader(VertexShaderHandle,   ShaderType::Vertex,   VSSetShader);
        set_shader(FragmentShaderHandle, ShaderType::Pixel, PSSetShader);
        set_shader(HullShaderHandle,     ShaderType::Hull,     HSSetShader);
        set_shader(DomainShaderHandle,   ShaderType::Domain,   DSSetShader);
        set_shader(GeometryShaderHandle, ShaderType::Geometry, GSSetShader);
#undef set_shader
    }

    void RenderContext::applyVao() const
    {
        auto pCtx = getD3D11ImmediateContext();
        uint32_t offsets[D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT] = {0};
        uint32_t strides[D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT] = {0};
        ID3D11Buffer* pVB[D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT] = {nullptr};
        ID3D11Buffer* pIB = nullptr;
        ID3D11InputLayout* pLayout = nullptr;
        
        const auto pVao = mState.pVao;
        if(pVao)
        {
            // Get the vertex buffers
            for(uint32_t i = 0; i < pVao->getVertexBuffersCount() ; i++)
            {
                pVB[i] = pVao->getVertexBuffer(i)->getApiHandle();
                strides[i] = pVao->getVertexBufferStride(i);
            }

            // Get the index buffer
            pIB = pVao->getIndexBuffer() ? pVao->getIndexBuffer()->getApiHandle() : nullptr;

        }

        pCtx->IASetIndexBuffer(pIB, DXGI_FORMAT_R32_UINT, 0);
        pCtx->IASetVertexBuffers(0, arraysize(pVB), pVB, strides, offsets);
    }

    void RenderContext::applyFbo() const
    {
        std::vector<ID3D11RenderTargetView*> pRTV;
        uint32_t colorTargets = Fbo::getMaxColorTargetCount();
        pRTV.assign(colorTargets, nullptr);
        ID3D11DepthStencilView* pDSV = nullptr;

        if(mState.pFbo)
        {
            for(uint32_t i = 0; i < colorTargets; i++)
                pRTV[i] = mState.pFbo->getRenderTargetView(i);

            pDSV = mState.pFbo->getDepthStencilView();
        }

        auto pCtx = getD3D11ImmediateContext();
        pCtx->OMSetRenderTargets(colorTargets, pRTV.data(), pDSV);
    }

    void RenderContext::blitFbo(const Fbo* pSource, const Fbo* pTarget, const glm::ivec4& srcRegion, const glm::ivec4& dstRegion, bool useLinearFiltering, FboAttachmentType copyFlags, uint32_t srcIdx, uint32_t dstIdx)
	{
        UNSUPPORTED_IN_D3D11("BlitFbo");
	}

    void RenderContext::applyConstantBuffer(uint32_t Index) const
    {
        ID3D11Buffer* pBuffer = nullptr;
        if(mState.pUniformBuffers[Index] && mState.pUniformBuffers[Index]->getBuffer())
        {
            pBuffer = mState.pUniformBuffers[Index]->getBuffer()->getApiHandle();
        }

        auto pCtx = getD3D11ImmediateContext();
        pCtx->VSSetConstantBuffers(Index, 1, &pBuffer);
        pCtx->PSSetConstantBuffers(Index, 1, &pBuffer);
        pCtx->DSSetConstantBuffers(Index, 1, &pBuffer);
        pCtx->HSSetConstantBuffers(Index, 1, &pBuffer);
        pCtx->GSSetConstantBuffers(Index, 1, &pBuffer);        
    }

    void RenderContext::applyShaderStorageBuffer(uint32_t index) const
    {
        UNSUPPORTED_IN_D3D11("RenderContext::ApplyShaderStorageBuffer()");
    }

    void RenderContext::applyTopology() const
    {
        D3D11_PRIMITIVE_TOPOLOGY topology;
        switch (mState.topology)
        {
        case Topology::PointList:
            topology = D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;
            break;
        case Topology::LineList:
            topology = D3D11_PRIMITIVE_TOPOLOGY_LINELIST;
            break;
        case Topology::TriangleList:
            topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            break;
        case Topology::TriangleStrip:
            topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
            break;
        default:
            should_not_get_here();
        }

        getD3D11ImmediateContext()->IASetPrimitiveTopology(topology);
    }

    void ApplyShaderResources(ID3D11DeviceContextPtr pCtx, const std::map<uint32_t, ID3D11ShaderResourceViewPtr>& pSrvMap, const std::map<uint32_t, ID3D11SamplerStatePtr>& pSamplerMap)
    {
        for(const auto& tex : pSrvMap)
        {
            uint32_t index = tex.first;
            ID3D11ShaderResourceView* pSRV = tex.second;
            pCtx->VSSetShaderResources(index, 1, &pSRV);
            pCtx->PSSetShaderResources(index, 1, &pSRV);
            pCtx->GSSetShaderResources(index, 1, &pSRV);
            pCtx->DSSetShaderResources(index, 1, &pSRV);
            pCtx->HSSetShaderResources(index, 1, &pSRV);
        }

        for(const auto& Sampler : pSamplerMap)
        {
            uint32_t index = Sampler.first;
            ID3D11SamplerState* pSampler = Sampler.second;
            pCtx->VSSetSamplers(index, 1, &pSampler);
            pCtx->PSSetSamplers(index, 1, &pSampler);
            pCtx->HSSetSamplers(index, 1, &pSampler);
            pCtx->DSSetSamplers(index, 1, &pSampler);
            pCtx->GSSetSamplers(index, 1, &pSampler);
        }
    }

    void RenderContext::prepareForDrawApi() const
    {
        // Set the input layout
        const Shader* pShader = mState.pProgram ? mState.pProgram->getShader(ShaderType::Vertex) : nullptr;
        ID3D11InputLayoutPtr pLayout = nullptr;
        if(mState.pVao && pShader)
        {
            pLayout = mState.pVao->getInputLayout(pShader->getD3DBlob());
        }

        ID3D11DeviceContextPtr pCtx = getD3D11ImmediateContext();
        pCtx->IASetInputLayout(pLayout);

        // Set the shader resources
        // All constant buffers for the current program holds the entire resource information, so just need to find the first uniform buffer and use it
        ConstantBuffer::SharedConstPtr pBuffer;
        for(size_t i = 0; i < mState.pUniformBuffers.size(); i++)
        {
            pBuffer = mState.pUniformBuffers[i];
            if(pBuffer)
            {
                ApplyShaderResources(pCtx, pBuffer->getAssignedResourcesMap(), pBuffer->getAssignedSamplersMap());
                break;
            }
        }
    }

    void RenderContext::draw(uint32_t vertexCount, uint32_t startVertexLocation)
    {
        prepareForDraw();
        getD3D11ImmediateContext()->Draw(vertexCount, startVertexLocation);
    }

    void RenderContext::drawIndexed(uint32_t indexCount, uint32_t startIndexLocation, int baseVertexLocation)
    {
        prepareForDraw();
        getD3D11ImmediateContext()->DrawIndexed(indexCount, startIndexLocation, baseVertexLocation);
    }

    void RenderContext::drawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t startIndexLocation, int baseVertexLocation, uint32_t startInstanceLocation)
    {
        prepareForDraw();
        getD3D11ImmediateContext()->DrawIndexedInstanced(indexCount, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation);
    }

    void RenderContext::applyViewport(uint32_t index) const
    {
        static_assert(offsetof(Viewport, originX) == offsetof(D3D11_VIEWPORT, TopLeftX), "VP TopLeftX offset");
        static_assert(offsetof(Viewport, originY) == offsetof(D3D11_VIEWPORT, TopLeftY), "VP TopLeftY offset");
        static_assert(offsetof(Viewport, width) == offsetof(D3D11_VIEWPORT, Width), "VP Width offset");
        static_assert(offsetof(Viewport, height) == offsetof(D3D11_VIEWPORT, Height), "VP Height offset");
        static_assert(offsetof(Viewport, minDepth) == offsetof(D3D11_VIEWPORT, MinDepth), "VP MinDepth offset");
        static_assert(offsetof(Viewport, maxDepth) == offsetof(D3D11_VIEWPORT, MaxDepth), "VP TopLeftX offset");
                
        getD3D11ImmediateContext()->RSSetViewports(D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE, (D3D11_VIEWPORT*)mState.viewports.data());
    }

    void RenderContext::applyScissor(uint32_t index) const
    {
        std::vector<D3D11_RECT> scRects;
        scRects.resize(mState.scissors.size());

        for(uint32_t si = 0u; si < mState.scissors.size(); ++si)
        {
            scRects[si].left   = mState.scissors[si].originX;
            scRects[si].top    = mState.scissors[si].originY;
            scRects[si].right  = mState.scissors[si].originX + mState.scissors[si].width;
            scRects[si].bottom = mState.scissors[si].originY + mState.scissors[si].height;
        }

        getD3D11ImmediateContext()->RSSetScissorRects(D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE, scRects.data());
    }
}

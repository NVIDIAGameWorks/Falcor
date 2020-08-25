/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#define NOMINMAX
#include <d3d12.h>
#include "Core/API/Formats.h"
#include <comdef.h>
#include <dxgi1_4.h>
#include <dxgiformat.h>

#define MAKE_SMART_COM_PTR(_a) _COM_SMARTPTR_TYPEDEF(_a, __uuidof(_a))

__forceinline BOOL dxBool(bool b) { return b ? TRUE : FALSE; }

#define d3d_call(a) {HRESULT hr_ = a; if(FAILED(hr_)) { d3dTraceHR( #a, hr_); }}

#define GET_COM_INTERFACE(base, type, var) MAKE_SMART_COM_PTR(type); concat_strings(type, Ptr) var; d3d_call(base->QueryInterface(IID_PPV_ARGS(&var)));

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")

#define UNSUPPORTED_IN_D3D(msg_) {logWarning(msg_ + std::string(" is not supported in D3D. Ignoring call."));}

namespace Falcor
{
    class DescriptorSet;
    /*!
    *  \addtogroup Falcor
    *  @{
    */

    struct DxgiFormatDesc
    {
        ResourceFormat falcorFormat;
        DXGI_FORMAT dxgiFormat;
    };

    extern const dlldecl DxgiFormatDesc kDxgiFormatDesc[];

    /** Convert from Falcor to DXGI format
    */
    inline DXGI_FORMAT getDxgiFormat(ResourceFormat format)
    {
        assert(kDxgiFormatDesc[(uint32_t)format].falcorFormat == format);
        return kDxgiFormatDesc[(uint32_t)format].dxgiFormat;
    }

    /** Convert from DXGI to Falcor format
    */
    inline ResourceFormat getResourceFormat(DXGI_FORMAT format)
    {
        for (size_t i = 0; i < (size_t)ResourceFormat::Count; ++i)
        {
            const auto& desc = kDxgiFormatDesc[i];
            if (desc.dxgiFormat == format) return desc.falcorFormat;
        }
        return ResourceFormat::Unknown;
    }

    inline DXGI_FORMAT getTypelessFormatFromDepthFormat(ResourceFormat format)
    {
        switch (format)
        {
        case ResourceFormat::D16Unorm:
            return DXGI_FORMAT_R16_TYPELESS;
        case ResourceFormat::D32FloatS8X24:
            return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        case ResourceFormat::D24UnormS8:
            return DXGI_FORMAT_R24G8_TYPELESS;
        case ResourceFormat::D32Float:
            return DXGI_FORMAT_R32_TYPELESS;
        default:
            assert(isDepthFormat(format) == false);
            return kDxgiFormatDesc[(uint32_t)format].dxgiFormat;
        }
    }

#define to_string_case(a) case a: return #a;
    inline std::string to_string(D3D_FEATURE_LEVEL featureLevel)
    {
        switch (featureLevel)
        {
            to_string_case(D3D_FEATURE_LEVEL_9_1)
            to_string_case(D3D_FEATURE_LEVEL_9_2)
            to_string_case(D3D_FEATURE_LEVEL_9_3)
            to_string_case(D3D_FEATURE_LEVEL_10_0)
            to_string_case(D3D_FEATURE_LEVEL_10_1)
            to_string_case(D3D_FEATURE_LEVEL_11_0)
            to_string_case(D3D_FEATURE_LEVEL_11_1)
            to_string_case(D3D_FEATURE_LEVEL_12_0)
            to_string_case(D3D_FEATURE_LEVEL_12_1)
        default: should_not_get_here(); return "";
        }
    }
#undef to_string_case

    /** Get D3D_FEATURE_LEVEL
    */
    D3D_FEATURE_LEVEL getD3DFeatureLevel(uint32_t majorVersion, uint32_t minorVersion);

    /** Log a message if hr indicates an error
    */
    void dlldecl d3dTraceHR(const std::string& Msg, HRESULT hr);

    template<typename BlobType>
    inline std::string convertBlobToString(BlobType* pBlob)
    {
        std::vector<char> infoLog(pBlob->GetBufferSize() + 1);
        memcpy(infoLog.data(), pBlob->GetBufferPointer(), pBlob->GetBufferSize());
        infoLog[pBlob->GetBufferSize()] = 0;
        return std::string(infoLog.data());
    }

    enum class RtBuildFlags
    {
        None = 0,
        AllowUpdate = 0x1,
        AllowCompaction = 0x2,
        FastTrace = 0x4,
        FastBuild = 0x8,
        MinimizeMemory = 0x10,
        PerformUpdate = 0x20,
    };
    enum_class_operators(RtBuildFlags);

#define rt_flags(a) case RtBuildFlags::a: return #a
    inline std::string to_string(RtBuildFlags flags)
    {
        switch (flags)
        {
            rt_flags(None);
            rt_flags(AllowUpdate);
            rt_flags(AllowCompaction);
            rt_flags(FastTrace);
            rt_flags(FastBuild);
            rt_flags(MinimizeMemory);
            rt_flags(PerformUpdate);
        default:
            should_not_get_here();
            return "";
        }
    }
#undef rt_flags

    inline D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS getDxrBuildFlags(RtBuildFlags buildFlags)
    {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS dxr = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;

        if (is_set(buildFlags, RtBuildFlags::AllowUpdate)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
        if (is_set(buildFlags, RtBuildFlags::AllowCompaction)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
        if (is_set(buildFlags, RtBuildFlags::FastTrace)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        if (is_set(buildFlags, RtBuildFlags::FastBuild)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        if (is_set(buildFlags, RtBuildFlags::MinimizeMemory)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY;
        if (is_set(buildFlags, RtBuildFlags::PerformUpdate)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;

        return dxr;
    }

    // The max scalars supported by our driver
    #define FALCOR_RT_MAX_PAYLOAD_SIZE_IN_BYTES (14 * sizeof(float))

    // DXGI
    MAKE_SMART_COM_PTR(IDXGISwapChain3);
    MAKE_SMART_COM_PTR(IDXGIDevice);
    MAKE_SMART_COM_PTR(IDXGIAdapter1);
    MAKE_SMART_COM_PTR(IDXGIFactory4);
    MAKE_SMART_COM_PTR(ID3DBlob);

    MAKE_SMART_COM_PTR(ID3D12StateObject);
    MAKE_SMART_COM_PTR(ID3D12Device);
    MAKE_SMART_COM_PTR(ID3D12GraphicsCommandList);
    MAKE_SMART_COM_PTR(ID3D12Debug);
    MAKE_SMART_COM_PTR(ID3D12CommandQueue);
    MAKE_SMART_COM_PTR(ID3D12CommandAllocator);
    MAKE_SMART_COM_PTR(ID3D12DescriptorHeap);
    MAKE_SMART_COM_PTR(ID3D12Resource);
    MAKE_SMART_COM_PTR(ID3D12Fence);
    MAKE_SMART_COM_PTR(ID3D12PipelineState);
    MAKE_SMART_COM_PTR(ID3D12RootSignature);
    MAKE_SMART_COM_PTR(ID3D12QueryHeap);
    MAKE_SMART_COM_PTR(ID3D12CommandSignature);
    MAKE_SMART_COM_PTR(IUnknown);

    using ApiObjectHandle = IUnknownPtr;

    using HeapCpuHandle = D3D12_CPU_DESCRIPTOR_HANDLE;
    using HeapGpuHandle = D3D12_GPU_DESCRIPTOR_HANDLE;

    class DescriptorHeapEntry;

    using WindowHandle = HWND;
    using DeviceHandle = ID3D12DevicePtr;
    using CommandListHandle = ID3D12GraphicsCommandListPtr;
    using CommandQueueHandle = ID3D12CommandQueuePtr;
    using ApiCommandQueueType = D3D12_COMMAND_LIST_TYPE;
    using CommandAllocatorHandle = ID3D12CommandAllocatorPtr;
    using CommandSignatureHandle = ID3D12CommandSignaturePtr;
    using FenceHandle = ID3D12FencePtr;
    using ResourceHandle = ID3D12ResourcePtr;
    using RtvHandle = std::shared_ptr<DescriptorSet>;
    using DsvHandle = std::shared_ptr<DescriptorSet>;
    using SrvHandle = std::shared_ptr<DescriptorSet>;
    using SamplerHandle = std::shared_ptr<DescriptorSet>;
    using UavHandle = std::shared_ptr<DescriptorSet>;
    using CbvHandle = std::shared_ptr<DescriptorSet>;
    using FboHandle = void*;
    using GpuAddress = D3D12_GPU_VIRTUAL_ADDRESS;
    using QueryHeapHandle = ID3D12QueryHeapPtr;
    using SharedResourceApiHandle = HANDLE;

    using GraphicsStateHandle = ID3D12PipelineStatePtr;
    using ComputeStateHandle = ID3D12PipelineStatePtr;
    using ShaderHandle = D3D12_SHADER_BYTECODE;
    using RootSignatureHandle = ID3D12RootSignaturePtr;
    using DescriptorHeapHandle = ID3D12DescriptorHeapPtr;

    using VaoHandle = void*;
    using VertexShaderHandle = void*;
    using FragmentShaderHandle = void*;
    using DomainShaderHandle = void*;
    using HullShaderHandle = void*;
    using GeometryShaderHandle = void*;
    using ComputeShaderHandle = void*;
    using ProgramHandle = void*;
    using DepthStencilStateHandle = void*;
    using RasterizerStateHandle = void*;
    using BlendStateHandle = void*;
    using DescriptorSetApiHandle = void*;

    inline constexpr uint32_t getMaxViewportCount() { return D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE; }

    /*! @} */
}

#define UNSUPPORTED_IN_D3D12(msg_) {Falcor::logWarning(msg_ + std::string(" is not supported in D3D12. Ignoring call."));}

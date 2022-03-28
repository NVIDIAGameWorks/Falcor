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
#pragma once
#define FALCOR_D3D12_AVAILABLE 1
#define NOMINMAX
#include <d3d12.h>
#include "Core/FalcorConfig.h"
#include "Core/API/Formats.h"
#include "Core/API/Shared/D3D12Handles.h"
#include <dxgiformat.h>


#if FALCOR_ENABLE_D3D12_AGILITY_SDK
// To enable the D3D12 Agility SDK, this macro needs to be added to the main source file of the executable.
#define FALCOR_EXPORT_D3D12_AGILITY_SDK                                                     \
    extern "C" { FALCOR_API_EXPORT extern const UINT D3D12SDKVersion = 4;}              \
    extern "C" { FALCOR_API_EXPORT extern const char* D3D12SDKPath = u8".\\D3D12\\"; }
#else
#define FALCOR_EXPORT_D3D12_AGILITY_SDK
#endif

#define FALCOR_D3D_CALL(_a) {HRESULT hr_ = _a; if (FAILED(hr_)) { Falcor::d3dTraceHR( #_a, hr_); }}

#define FALCOR_GET_COM_INTERFACE(_base, _type, _var) FALCOR_MAKE_SMART_COM_PTR(_type); FALCOR_CONCAT_STRINGS(_type, Ptr) _var; FALCOR_D3D_CALL(_base->QueryInterface(IID_PPV_ARGS(&_var)));

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")

#define FALCOR_UNSUPPORTED_IN_D3D(_msg) {Falcor::logWarning("{}  is not supported in D3D. Ignoring call.", _msg);}

#define FALCOR_NVAPI_AVAILABLE FALCOR_ENABLE_NVAPI

namespace Falcor
{
    class D3D12DescriptorSet;
    class ShaderResourceView;
    /*!
    *  \addtogroup Falcor
    *  @{
    */

    inline BOOL dxBool(bool b) { return b ? TRUE : FALSE; }

    /** Get D3D_FEATURE_LEVEL
    */
    D3D_FEATURE_LEVEL getD3DFeatureLevel(uint32_t majorVersion, uint32_t minorVersion);

    /** Log a message if hr indicates an error
    */
    void FALCOR_API d3dTraceHR(const std::string& Msg, HRESULT hr);

    template<typename BlobType>
    inline std::string convertBlobToString(BlobType* pBlob)
    {
        std::vector<char> infoLog(pBlob->GetBufferSize() + 1);
        memcpy(infoLog.data(), pBlob->GetBufferPointer(), pBlob->GetBufferSize());
        infoLog[pBlob->GetBufferSize()] = 0;
        return std::string(infoLog.data());
    }

    /** Flags passed to TraceRay(). These must match the device side.
    */
    enum class RayFlags : uint32_t
    {
        None = D3D12_RAY_FLAG_NONE,
        ForceOpaque = D3D12_RAY_FLAG_FORCE_OPAQUE,
        ForceNonOpaque = D3D12_RAY_FLAG_FORCE_NON_OPAQUE,
        AcceptFirstHitAndEndSearch = D3D12_RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        SkipClosestHitShader = D3D12_RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        CullBackFacingTriangles = D3D12_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        CullFrontFacingTriangles = D3D12_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
        CullOpaque = D3D12_RAY_FLAG_CULL_OPAQUE,
        CullNonOpaque = D3D12_RAY_FLAG_CULL_NON_OPAQUE,
        SkipTriangles = D3D12_RAY_FLAG_SKIP_TRIANGLES,
        SkipProceduralPrimitives = D3D12_RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
    };
    FALCOR_ENUM_CLASS_OPERATORS(RayFlags);

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
    FALCOR_ENUM_CLASS_OPERATORS(RtBuildFlags);

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
            FALCOR_UNREACHABLE();
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

    // Maximum raytracing attribute size.
    inline constexpr uint32_t getRaytracingMaxAttributeSize() { return D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES; }

    // DXGI

    FALCOR_MAKE_SMART_COM_PTR(IUnknown);

    using ApiObjectHandle = IUnknownPtr;

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
    using RtvHandle = std::shared_ptr<D3D12DescriptorSet>;
    using DsvHandle = std::shared_ptr<D3D12DescriptorSet>;
    using SrvHandle = std::shared_ptr<D3D12DescriptorSet>;
    using SamplerHandle = std::shared_ptr<D3D12DescriptorSet>;
    using UavHandle = std::shared_ptr<D3D12DescriptorSet>;
    using CbvHandle = std::shared_ptr<D3D12DescriptorSet>;
    using AccelerationStructureHandle = std::shared_ptr<ShaderResourceView>;
    using FboHandle = void*;
    using GpuAddress = D3D12_GPU_VIRTUAL_ADDRESS;
    using QueryHeapHandle = ID3D12QueryHeapPtr;
    using SharedResourceApiHandle = HANDLE;
    using SharedFenceApiHandle = HANDLE;

    using GraphicsStateHandle = ID3D12PipelineStatePtr;
    using ComputeStateHandle = ID3D12PipelineStatePtr;
    using RaytracingStateHandle = ID3D12StateObjectPtr;
    
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

#define FALCOR_UNSUPPORTED_IN_D3D12(_msg) {Falcor::logWarning("{} is not supported in D3D12. Ignoring call.", _msg);}

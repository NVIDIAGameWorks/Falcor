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
#include "GpuFence.h"
#include "Handles.h"
#include "Shared/D3D12Handles.h"
#if FALCOR_HAS_D3D12
#include "Shared/D3D12DescriptorPool.h"
#endif
#include "Core/Macros.h"
#include <memory>


namespace Falcor
{
    struct LowLevelContextApiData;

    class FALCOR_API LowLevelContextData
    {
    public:
        using SharedPtr = std::shared_ptr<LowLevelContextData>;
        using SharedConstPtr = std::shared_ptr<const LowLevelContextData>;

        enum class CommandQueueType
        {
            Copy,
            Compute,
            Direct,
            Count
        };

        ~LowLevelContextData();

        /** Create a new low-level context data object.
            \param[in] type Command queue type.
            \param[in] queue Command queue handle. Can be nullptr.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(CommandQueueType type, CommandQueueHandle queue);

        void flush();

        const CommandListHandle& getCommandList() const { return mpList; }
        const CommandQueueHandle& getCommandQueue() const { return mpQueue; }
        const D3D12CommandListHandle& getD3D12CommandList() const;
        const D3D12CommandQueueHandle& getD3D12CommandQueue() const;
        const CommandAllocatorHandle& getCommandAllocator() const { return mpAllocator; }
        const GpuFence::SharedPtr& getFence() const { return mpFence; }
        LowLevelContextApiData* getApiData() const { return mpApiData.get(); }

#ifdef FALCOR_D3D12
        // Used in DXR
        void setCommandList(CommandListHandle pList) { mpList = pList; }
#endif
#ifdef FALCOR_GFX
        void closeCommandBuffer();
        void openCommandBuffer();
        void beginDebugEvent(const char* name);
        void endDebugEvent();
#endif
    protected:
        LowLevelContextData(CommandQueueType type, CommandQueueHandle queue);
        std::unique_ptr<LowLevelContextApiData> mpApiData;
        CommandQueueType mType;
        CommandListHandle mpList;
        CommandQueueHandle mpQueue; // Can be nullptr
        CommandAllocatorHandle mpAllocator;
        GpuFence::SharedPtr mpFence;
    };
}

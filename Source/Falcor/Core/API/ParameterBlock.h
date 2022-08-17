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
#include "Buffer.h"
#include "Texture.h"
#include "Sampler.h"
#include "RtAccelerationStructure.h"
#include "Core/Macros.h"
#include "Core/Program/ProgramReflection.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/UI/Gui.h"

#include <slang.h>

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <cstddef>

namespace Falcor
{
    class ProgramVersion;
    class CopyContext;

    /** Shared pointer class for `ParameterBlock` and derived classes.
        This smart pointer type adds syntax sugar for `operator[]` so that it implicitly operates on a `ShaderVar` derived from the contents of the buffer.
    */
    template<typename T>
    class ParameterBlockSharedPtr : public std::shared_ptr<T>
    {
    public:
        ParameterBlockSharedPtr() : std::shared_ptr<T>() {}
        explicit ParameterBlockSharedPtr(T* pObject) : std::shared_ptr<T>(pObject) {}
        constexpr ParameterBlockSharedPtr(std::nullptr_t) : std::shared_ptr<T>(nullptr) {}
        ParameterBlockSharedPtr(const std::shared_ptr<T>& pObject) : std::shared_ptr<T>(pObject) {}

        /** Implicitly convert a `ShaderVar` to a `ParameterBlock` pointer.
        */
        ParameterBlockSharedPtr(const ShaderVar& var) : std::shared_ptr<T>(var.getParameterBlock()) {}

        /** Get a shader variable that points to the root/contents of the parameter block.
        */
        ShaderVar getRootVar() const
        {
            return std::shared_ptr<T>::get()->getRootVar();
        }

        /** Get a shader variable that points at the field with the given `name`.
            This is an alias for `getRootVar()[name]`.
        */
        ShaderVar operator[](const std::string& name) const
        {
            return getRootVar()[name];
        }

        /** Get a shader variable that points at the field with the given `name`.
            This is an alias for `getRootVar()[name]`.
        */
        ShaderVar operator[](const char* name) const
        {
            return getRootVar()[name];
        }

        /** Get a shader variable that points at the field/element with the given `index`.
            This is an alias for `getRootVar()[index]`.
        */
        ShaderVar operator[](size_t index) const
        {
            return getRootVar()[index];
        }

        /** Get a shader variable that points at the field/element with the given `offset`.
            This is an alias for `getRootVar()[offset]`.
        */
        ShaderVar operator[](UniformShaderVarOffset offset) const
        {
            return getRootVar()[offset];
        }
    };

    /** A parameter block. This block stores all the parameter data associated with a specific type in shader code
    */
    class FALCOR_API ParameterBlock
    {
    public:
        using SharedPtr = ParameterBlockSharedPtr<ParameterBlock>;
        using SharedConstPtr = std::shared_ptr<const ParameterBlock>;
        ~ParameterBlock();

        using BindLocation = ParameterBlockReflection::BindLocation;

        /** Create a new object that holds a value of the given type.
        */
        static SharedPtr create(const std::shared_ptr<const ProgramVersion>& pProgramVersion, const ReflectionType::SharedConstPtr& pType);

        /** Create a new object that holds a value described by the given reflector.
        */
        static SharedPtr create(const ParameterBlockReflection::SharedConstPtr& pReflection);

        /** Create a new object that holds a value of the type with the given name in the given program.
            \param[in] pProgramVersion Program version object.
            \param[in] typeName Name of the type. If the type does not exist an exception is thrown.
        */
        static SharedPtr create(const std::shared_ptr<const ProgramVersion>& pProgramVersion, const std::string& typeName);

#ifdef FALCOR_GFX
        gfx::IShaderObject* getShaderObject() const { return mpShaderObject.get(); }
#endif

        /** Set a variable into the block.
            The function will validate that the value Type matches the declaration in the shader. If there's a mismatch, an error will be logged and the call will be ignored.
            \param[in] name The variable name. See notes about naming in the ConstantBuffer class description.
            \param[in] value Value to set
        */
        template<typename T>
        bool setVariable(const std::string& name, const T& value)
        {
            return getRootVar()[name].set(value);
        }

        /** Set a variable into the block.
            The function will validate that the value Type matches the declaration in the shader. If there's a mismatch, an error will be logged and the call will be ignored.
            \param[in] offset The variable byte offset inside the buffer
            \param[in] value Value to set
        */
        template<typename T>
        bool setVariable(UniformShaderVarOffset offset, const T& value);

        template<typename T>
        bool setBlob(UniformShaderVarOffset bindLocation, const T& blob) const
        {
            return setBlob(bindLocation, &blob, sizeof(blob));
        }

        bool setBlob(UniformShaderVarOffset offset, const void* pSrc, size_t size)
        {
            return setBlob(pSrc, offset, size);
        }

        bool setBlob(const void* pSrc, UniformShaderVarOffset offset, size_t size);
        bool setBlob(const void* pSrc, size_t offset, size_t size);

        /** Bind a buffer by name.
            If the name doesn't exists, the bind flags don't match the shader requirements or the size doesn't match the required size, the call will fail.
            \param[in] name The name of the buffer in the program
            \param[in] pBuffer The buffer object
            \return false is the call failed, otherwise true
        */
        bool setBuffer(const std::string& name, const Buffer::SharedPtr& pBuffer);

        /** Bind a buffer object by index
            If the no buffer exists in the specified index or the bind flags don't match the shader requirements or the size doesn't match the required size, the call will fail.
            \param[in] bindLocation The bind-location in the block
            \param[in] pBuffer The buffer object
            \return false is the call failed, otherwise true
        */
        bool setBuffer(const BindLocation& bindLocation, const Buffer::SharedPtr& pBuffer);

        /** Get a buffer
            \param[in] name The name of the buffer
            \return If the name is valid, a shared pointer to the buffer. Otherwise returns nullptr
        */
        Buffer::SharedPtr getBuffer(const std::string& name) const;

        /** Get a buffer
            \param[in] bindLocation The bind location of the buffer
            \return If the name is valid, a shared pointer to the buffer. Otherwise returns nullptr
        */
        Buffer::SharedPtr getBuffer(const BindLocation& bindLocation) const;

        /** Bind a parameter block by name.
            If the name doesn't exists or the size doesn't match the required size, the call will fail.
            \param[in] name The name of the parameter block in the program
            \param[in] pBlock The parameter block
            \return false is the call failed, otherwise true
        */
        bool setParameterBlock(const std::string& name, const ParameterBlock::SharedPtr& pBlock);

        /** Bind a parameter block by index.
            If the no parameter block exists in the specified index or the parameter block size doesn't match the required size, the call will fail.
            \param[in] bindLocation The location of the object
            \param[in] pBlock The parameter block
            \return false is the call failed, otherwise true
        */
        bool setParameterBlock(const BindLocation& bindLocation, const ParameterBlock::SharedPtr& pBlock);

        /** Get a parameter block.
            \param[in] name The name of the parameter block
            \return If the name is valid, a shared pointer to the parameter block. Otherwise returns nullptr
        */
        ParameterBlock::SharedPtr getParameterBlock(const std::string& name) const;

        /** Get a parameter block.
            \param[in] bindLocation The location of the block
            \return If the indices is valid, a shared pointer to the parameter block. Otherwise returns nullptr
        */
        ParameterBlock::SharedPtr getParameterBlock(const BindLocation& bindLocation) const;

        /** Bind a texture. Based on the shader reflection, it will be bound as either an SRV or a UAV
            \param[in] name The name of the texture object in the shader
            \param[in] pTexture The texture object to bind
        */
        bool setTexture(const std::string& name, const Texture::SharedPtr& pTexture);
        bool setTexture(const BindLocation& bindLocation, const Texture::SharedPtr& pTexture);

        /** Get a texture object.
            \param[in] name The name of the texture
            \return If the name is valid, a shared pointer to the texture object. Otherwise returns nullptr
        */
        Texture::SharedPtr getTexture(const std::string& name) const;
        Texture::SharedPtr getTexture(const BindLocation& bindLocation) const;

        /** Bind an SRV.
            \param[in] bindLocation The bind-location in the block
            \param[in] pSrv The shader-resource-view object to bind
        */
        bool setSrv(const BindLocation& bindLocation, const ShaderResourceView::SharedPtr& pSrv);

        /** Bind a UAV.
            \param[in] bindLocation The bind-location in the block
            \param[in] pSrv The unordered-access-view object to bind
        */
        bool setUav(const BindLocation& bindLocation, const UnorderedAccessView::SharedPtr& pUav);

        /** Bind an acceleration structure.
            \param[in] bindLocation The bind-location in the block
            \param[in] pAccl The acceleration structure object to bind
            \return false if the binding location does not accept an acceleration structure, true otherwise.
        */
        bool setAccelerationStructure(const BindLocation& bindLocation, const RtAccelerationStructure::SharedPtr& pAccl);

        /** Get an SRV object.
            \param[in] bindLocation The bind-location in the block
            \return If the bind-location is valid, a shared pointer to the SRV. Otherwise returns nullptr
        */
        ShaderResourceView::SharedPtr getSrv(const BindLocation& bindLocation) const;

        /** Get a UAV object
            \param[in] bindLocation The bind-location in the block
            \return If the bind-location is valid, a shared pointer to the UAV. Otherwise returns nullptr
        */
        UnorderedAccessView::SharedPtr getUav(const BindLocation& bindLocation) const;

        /** Get an acceleration structure object.
            \param[in] bindLocation The bind-location in the block
            \return If the bind-location is valid, a shared pointer to the acceleration structure. Otherwise returns nullptr
        */
        RtAccelerationStructure::SharedPtr getAccelerationStructure(const BindLocation& bindLocation) const;

        /** Bind a sampler to the program in the global namespace.
            \param[in] name The name of the sampler object in the shader
            \param[in] pSampler The sampler object to bind
            \return false if the sampler was not found in the program, otherwise true
        */
        bool setSampler(const std::string& name, const Sampler::SharedPtr& pSampler);

        /** Bind a sampler to the program in the global namespace.
            \param[in] bindLocation The bind-location in the block
            \param[in] pSampler The sampler object to bind
            \return false if the sampler was not found in the program, otherwise true
        */
        bool setSampler(const BindLocation& bindLocation, const Sampler::SharedPtr& pSampler);

        /** Gets a sampler object.
            \param[in] bindLocation The bind-location in the block
            \return If the bind-location is valid, a shared pointer to the sampler. Otherwise returns nullptr
        */
        const Sampler::SharedPtr& getSampler(const BindLocation& bindLocation) const;

        /** Gets a sampler object.
            \return If the name is valid, a shared pointer to the sampler. Otherwise returns nullptr
        */
        Sampler::SharedPtr getSampler(const std::string& name) const;

        /** Get the parameter block's reflection interface
        */
        ParameterBlockReflection::SharedConstPtr getReflection() const { return mpReflector; }

        /** Get the block reflection type
        */
        ReflectionType::SharedConstPtr getElementType() const { return mpReflector->getElementType(); }

        /** Get the size of the reflection type
        */
        size_t getElementSize() const;

        /** Get offset of a uniform variable inside the block, given its name.
        */
        UniformShaderVarOffset getVariableOffset(const std::string& varName) const;

        /** Get an initial var to the contents of this block.
        */
        ShaderVar getRootVar() const;

        /** Try to find a shader var for a member of the block.

            Returns an invalid shader var if no such member is found.
        */
        ShaderVar findMember(const std::string& varName) const;

        /** Try to find a shader var for a member of the block by index.

            Returns an invalid shader var if no such member is found.
        */
        ShaderVar findMember(uint32_t index) const;

        /** Get the size of the parameter-block's buffer
        */
        size_t getSize() const;

        bool updateSpecialization() const;
        ParameterBlockReflection::SharedConstPtr getSpecializedReflector() const { return mpSpecializedReflector; }

        bool prepareDescriptorSets(CopyContext* pCopyContext);

#ifdef FALCOR_D3D12
        uint32_t getD3D12DescriptorSetCount() const { return mpReflector->getD3D12DescriptorSetCount(); }
        D3D12DescriptorSet::SharedPtr const& getD3D12DescriptorSet(uint32_t index) const { return mSets[index].pSet; }

        std::pair<Resource::SharedPtr, bool> getRootDescriptor(uint32_t resourceRangeIndex, uint32_t arrayIndex) const;
#endif // FALCOR_D3D12

        void renderUI(Gui::Widgets& widget);
        const ParameterBlock::SharedPtr& getParameterBlock(uint32_t resourceRangeIndex, uint32_t arrayIndex) const;

        // Delete some functions. If they are not deleted, the compiler will try to convert the uints to string, resulting in runtime error
        Sampler::SharedPtr getSampler(uint32_t) = delete;
        bool setSampler(uint32_t, const Sampler::SharedPtr&) = delete;

        using SpecializationArgs = std::vector<slang::SpecializationArg>;
        void collectSpecializationArgs(SpecializationArgs& ioArgs) const;

        void markUniformDataDirty() const;

        void const* getRawData() const;

        /** Get the underlying constant buffer that holds the ordinary/uniform data for this block.
            Be cautious with the returned buffer as it can be invalidated any time you set/bind something
            to the parameter block (or one if its internal sub-blocks).
        */
        const Buffer::SharedPtr& getUnderlyingConstantBuffer() const;

#if FALCOR_HAS_CUDA
        /** Get a host-memory pointer that represents the contents of this shader object
            as a CUDA-compatible buffer.

            In the case where this parameter block represents a `ProgramVars`, the resulting
            buffer can be passed as argument data for a kernel launch.

            \return Host-memory pointer to a copy of the block, or throws on error
            (e.g., parameter types that are unsupported on CUDA).

            The lifetime of the returned pointer is tied to the `ParameterBlock`,
            and does not need to be explicitly deleted by the caller. The pointer
            may become invalid if:

            * The parameter block is deleted
            * A call to `getCUDADeviceBuffer()` is made on the same parameter block
            * Another call it made to `getCUDAHostBuffer()` after changes have been made to parameters in the block
        */
        void* getCUDAHostBuffer(size_t& outSize);

        /** Get a device-memory pointer that represents the contents of this shader object
            as a CUDA-compatible buffer.

            The resulting buffer can be used to represent this shader object when it
            is used as a constant buffer or parameter block.

            \return Device-memory pointer to a copy of the block, or throws on error
            (e.g., parameter types that are unsupported on CUDA).

            The lifetime of the returned pointer is tied to the `ParameterBlock`,
            and does not need to be explicitly deleted by the caller. The pointer
            may become invalid if:

            * The parameter block is deleted
            * A call to `getCUDAHostBuffer()` is made on the same parameter block
            * Another call it made to `getCUDADeviceBuffer()` after changes have been made to parameters in the block
        */
        void* getCUDADeviceBuffer(size_t& outSize);
#endif

        typedef uint64_t ChangeEpoch;

    protected:
        friend class VariablesBufferUI;

        ParameterBlock(
            const std::shared_ptr<const ProgramVersion>& pProgramVersion,
            const ParameterBlockReflection::SharedConstPtr& pReflection);

#ifdef FALCOR_GFX
        ParameterBlock(const ProgramReflection::SharedConstPtr& pReflector);
#endif

        std::shared_ptr<const ProgramVersion> mpProgramVersion;
        ParameterBlockReflection::SharedConstPtr mpReflector;
        mutable ParameterBlockReflection::SharedConstPtr mpSpecializedReflector;

        void createConstantBuffers(const ShaderVar& var);

        static void prepareResource(
            CopyContext* pContext,
            Resource* pResource,
            bool isUav);

#ifdef FALCOR_D3D12
        std::vector<uint8_t> mData;

        virtual bool updateSpecializationImpl() const;

        /** Get a constant buffer view for the underlying constant buffer for ordinary/uniform data.
        */
        ConstantBufferView::SharedPtr getUnderlyingConstantBufferView();

        void validateUnderlyingConstantBuffer(
            ParameterBlockReflection const*   pReflector);

        void writeIntoBuffer(
            ParameterBlockReflection const*   pReflector,
            char*                           pBuffer,
            size_t                          bufferSize);

        bool prepareDescriptorSets(
            CopyContext*                    pCopyContext,
            const ParameterBlockReflection* pReflector);
        bool prepareDefaultConstantBufferAndResources(
            CopyContext*                        pContext,
            ParameterBlockReflection const*     pReflector);
        bool prepareResources(
            CopyContext*                    pContext,
            ParameterBlockReflection const* pReflector);

        bool bindIntoD3D12DescriptorSet(
            const ParameterBlockReflection*   pReflector,
            D3D12DescriptorSet::SharedPtr        pDescSet,
            uint32_t                        setIndex,
            uint32_t&                       destRangeIndex);

        bool bindResourcesIntoD3D12DescriptorSet(
            const ParameterBlockReflection*   pReflector,
            D3D12DescriptorSet::SharedPtr        pDescSet,
            uint32_t                        setIndex,
            uint32_t&                       destRangeIndex);

        struct AssignedSRV
        {
            ShaderResourceView::SharedPtr pView; // Can be a null view even when a valid resource is assigned, if the bind location is a root descriptor.
            Resource::SharedPtr pResource;
        };

        struct AssignedUAV
        {
            UnorderedAccessView::SharedPtr pView; // Can be a null view even when a valid resource is assigned, if the bind location is a root descriptor.
            Resource::SharedPtr pResource;
        };

        struct AssignedParameterBlock
        {
            ParameterBlock::SharedPtr pBlock;
            mutable ChangeEpoch epochOfLastObservedChange = 0;
        };

        std::vector<AssignedParameterBlock>     mParameterBlocks;
        std::vector<AssignedSRV>                mSRVs;              ///< All SRVs bound to descriptor sets or root descriptors.
        std::vector<AssignedUAV>                mUAVs;              ///< All UAVs bound to descriptor sets or root descriptors.
        std::vector<Sampler::SharedPtr>         mSamplers;

        // Map from a flat Srv binding index to a bound Acceleration Structure object.
        // Used by getAccelerationStructure().
        std::map<size_t, RtAccelerationStructure::SharedPtr> mAccelerationStructures;

        AssignedParameterBlock const& getAssignedParameterBlock(uint32_t resourceRangeIndex, uint32_t arrayIndex) const;
        const AssignedParameterBlock& getAssignedParameterBlock(size_t index) const;
        AssignedParameterBlock& getAssignedParameterBlock(size_t index);

        size_t getFlatIndex(const BindLocation& bindLocation) const;

        bool checkResourceIndices(const BindLocation& bindLocation, const char* funcName) const;
        template<size_t N>
        bool checkDescriptorType(const BindLocation& bindLocation, const std::array<ShaderResourceType, N>& allowedTypes, const char* funcName) const;
        bool checkDescriptorSrvUavCommon(
            const BindLocation& bindLocation,
            const Resource::SharedPtr& pResource,
            const std::variant<ShaderResourceView::SharedPtr, UnorderedAccessView::SharedPtr>& pView,
            const char* funcName) const;
        bool checkRootDescriptorResourceCompatibility(const Resource::SharedPtr& pResource, const std::string& funcName) const;

        bool setResourceSrvUavCommon(const BindLocation& bindLoc, const Resource::SharedPtr& pResource, const char* funcName);
        Resource::SharedPtr getResourceSrvUavCommon(const BindLocation& bindLoc, const char* funcName) const;

        mutable ChangeEpoch mEpochOfLastUniformDataChange = 1;
        mutable ChangeEpoch mEpochOfLastChange = 1;

        static ChangeEpoch getEpochOfLastChange(ParameterBlock* pBlock) { return pBlock->mEpochOfLastChange; }
        static ChangeEpoch computeEpochOfLastChange(ParameterBlock* pBlock);

        void checkForIndirectChanges(ParameterBlockReflection const* pReflector) const;

        mutable uint32_t mDescriptorSetResourceDataVersion = 0;

        uint32_t getDescriptorSetIndex(const BindLocation& bindLocation);
        void markDescriptorSetDirty(uint32_t index) const;
        void markDescriptorSetDirty(const BindLocation& bindLocation);

        struct UnderlyingConstantBuffer
        {
            Buffer::SharedPtr pBuffer;
            ConstantBufferView::SharedPtr pCBV;

            mutable ChangeEpoch epochOfLastObservedChange = 0;
        };
        mutable UnderlyingConstantBuffer mUnderlyingConstantBuffer;

        struct DescriptorSetInfo
        {
            D3D12DescriptorSet::SharedPtr pSet;
            ChangeEpoch epochOfLastChange;
        };
        mutable std::vector<DescriptorSetInfo> mSets;
#endif // FALCOR_D3D12

#ifdef FALCOR_GFX
        Slang::ComPtr<gfx::IShaderObject> mpShaderObject;
        std::map<gfx::ShaderOffset, ParameterBlock::SharedPtr> mParameterBlocks;
        std::map<gfx::ShaderOffset, ShaderResourceView::SharedPtr> mSRVs;
        std::map<gfx::ShaderOffset, UnorderedAccessView::SharedPtr> mUAVs;
        std::map<gfx::ShaderOffset, Resource::SharedPtr> mResources;
        std::map<gfx::ShaderOffset, Sampler::SharedPtr> mSamplers;
        std::map<gfx::ShaderOffset, RtAccelerationStructure::SharedPtr> mAccelerationStructures;
#endif // FALCOR_GFX

#if FALCOR_HAS_CUDA

        // The following members pertain to the issue of exposing the
        // current state/contents of a shader object to CUDA kernels.

        /** A kind of data buffer used for communicating with CUDA.
        */
        enum class CUDABufferKind
        {
            Host,   ///< A buffer in host memory
            Device, ///< A buffer in device memory
        };

        /** Get a CUDA-compatible buffer that represents the contents of this shader object.
        */
        void* getCUDABuffer(
            CUDABufferKind  bufferKind,
            size_t& outSize);

        /** Get a CUDA-compatible buffer that represents the contents of this shader object.
        */
        void* getCUDABuffer(
            const ParameterBlockReflection* pReflector,
            CUDABufferKind                  bufferKind,
            size_t& outSize);

        /** Update the CUDA-compatible buffer stored on this parameter block to reflect
            the current state of the shader object.
        */
        void updateCUDABuffer(
            const ParameterBlockReflection* pReflector,
            CUDABufferKind                  bufferKind);

        /** Information about the CUDA buffer (if any) used to represnet the state of
            this shader object
        */
        struct UnderlyingCUDABuffer
        {
            Buffer::SharedPtr   pBuffer;
            void*               pData                       = nullptr;
            ChangeEpoch         epochOfLastObservedChange   = 0;
            size_t              size                        = 0;
            CUDABufferKind      kind                        = CUDABufferKind::Host;
        };
        UnderlyingCUDABuffer mUnderlyingCUDABuffer;
#endif
    };

    template<typename T> bool ShaderVar::setImpl(const T& val) const
    {
        return mpBlock->setVariable(mOffset, val);
    }
}

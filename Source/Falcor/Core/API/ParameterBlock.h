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
#pragma once
#include "fwd.h"
#include "Buffer.h"
#include "Texture.h"
#include "Sampler.h"
#include "RtAccelerationStructure.h"
#include "Core/Macros.h"
#include "Core/Object.h"
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

/**
 * A parameter block. This block stores all the parameter data associated with a specific type in shader code
 */
class FALCOR_API ParameterBlock : public Object
{
    FALCOR_OBJECT(ParameterBlock)
public:
    ~ParameterBlock();

    using BindLocation = ParameterBlockReflection::BindLocation;

    /**
     * Create a new object that holds a value of the given type.
     */
    static ref<ParameterBlock> create(
        ref<Device> pDevice,
        const ref<const ProgramVersion>& pProgramVersion,
        const ref<const ReflectionType>& pType
    );

    /**
     * Create a new object that holds a value described by the given reflector.
     */
    static ref<ParameterBlock> create(ref<Device> pDevice, const ref<const ParameterBlockReflection>& pReflection);

    /**
     * Create a new object that holds a value of the type with the given name in the given program.
     * @param[in] pProgramVersion Program version object.
     * @param[in] typeName Name of the type. If the type does not exist an exception is thrown.
     */
    static ref<ParameterBlock> create(ref<Device> pDevice, const ref<const ProgramVersion>& pProgramVersion, const std::string& typeName);

    gfx::IShaderObject* getShaderObject() const { return mpShaderObject.get(); }

    /**
     * Set a variable into the block.
     * The function will validate that the value Type matches the declaration in the shader. If there's a mismatch, an error will be logged
     * and the call will be ignored.
     * @param[in] name The variable name. See notes about naming in the ConstantBuffer class description.
     * @param[in] value Value to set
     */
    template<typename T>
    bool setVariable(const std::string& name, const T& value)
    {
        return getRootVar()[name].set(value);
    }

    /**
     * Set a variable into the block.
     * The function will validate that the value Type matches the declaration in the shader. If there's a mismatch, an error will be logged
     * and the call will be ignored.
     * @param[in] offset The variable byte offset inside the buffer
     * @param[in] value Value to set
     */
    template<typename T>
    bool setVariable(UniformShaderVarOffset offset, const T& value);

    template<typename T>
    bool setBlob(UniformShaderVarOffset bindLocation, const T& blob) const
    {
        return setBlob(bindLocation, &blob, sizeof(blob));
    }

    bool setBlob(UniformShaderVarOffset offset, const void* pSrc, size_t size) { return setBlob(pSrc, offset, size); }

    bool setBlob(const void* pSrc, UniformShaderVarOffset offset, size_t size);
    bool setBlob(const void* pSrc, size_t offset, size_t size);

    /**
     * Bind a buffer by name.
     * If the name doesn't exists, the bind flags don't match the shader requirements or the size doesn't match the required size, the call
     * will fail.
     * @param[in] name The name of the buffer in the program
     * @param[in] pBuffer The buffer object
     * @return false is the call failed, otherwise true
     */
    bool setBuffer(const std::string& name, const ref<Buffer>& pBuffer);

    /**
     * Bind a buffer object by index
     * If the no buffer exists in the specified index or the bind flags don't match the shader requirements or the size doesn't match the
     * required size, the call will fail.
     * @param[in] bindLocation The bind-location in the block
     * @param[in] pBuffer The buffer object
     * @return false is the call failed, otherwise true
     */
    bool setBuffer(const BindLocation& bindLocation, const ref<Buffer>& pBuffer);

    /**
     * Get a buffer
     * @param[in] name The name of the buffer
     * @return If the name is valid, a shared pointer to the buffer. Otherwise returns nullptr
     */
    ref<Buffer> getBuffer(const std::string& name) const;

    /**
     * Get a buffer
     * @param[in] bindLocation The bind location of the buffer
     * @return If the name is valid, a shared pointer to the buffer. Otherwise returns nullptr
     */
    ref<Buffer> getBuffer(const BindLocation& bindLocation) const;

    /**
     * Bind a parameter block by name.
     * If the name doesn't exists or the size doesn't match the required size, the call will fail.
     * @param[in] name The name of the parameter block in the program
     * @param[in] pBlock The parameter block
     * @return false is the call failed, otherwise true
     */
    bool setParameterBlock(const std::string& name, const ref<ParameterBlock>& pBlock);

    /**
     * Bind a parameter block by index.
     * If the no parameter block exists in the specified index or the parameter block size doesn't match the required size, the call will
     * fail.
     * @param[in] bindLocation The location of the object
     * @param[in] pBlock The parameter block
     * @return false is the call failed, otherwise true
     */
    bool setParameterBlock(const BindLocation& bindLocation, const ref<ParameterBlock>& pBlock);

    /**
     * Get a parameter block.
     * @param[in] name The name of the parameter block
     * @return If the name is valid, a shared pointer to the parameter block. Otherwise returns nullptr
     */
    ref<ParameterBlock> getParameterBlock(const std::string& name) const;

    /**
     * Get a parameter block.
     * @param[in] bindLocation The location of the block
     * @return If the indices is valid, a shared pointer to the parameter block. Otherwise returns nullptr
     */
    ref<ParameterBlock> getParameterBlock(const BindLocation& bindLocation) const;

    /**
     * Bind a texture. Based on the shader reflection, it will be bound as either an SRV or a UAV
     * @param[in] name The name of the texture object in the shader
     * @param[in] pTexture The texture object to bind
     */
    bool setTexture(const std::string& name, const ref<Texture>& pTexture);
    bool setTexture(const BindLocation& bindLocation, const ref<Texture>& pTexture);

    /**
     * Get a texture object.
     * @param[in] name The name of the texture
     * @return If the name is valid, a shared pointer to the texture object. Otherwise returns nullptr
     */
    ref<Texture> getTexture(const std::string& name) const;
    ref<Texture> getTexture(const BindLocation& bindLocation) const;

    /**
     * Bind an SRV.
     * @param[in] bindLocation The bind-location in the block
     * @param[in] pSrv The shader-resource-view object to bind
     */
    bool setSrv(const BindLocation& bindLocation, const ref<ShaderResourceView>& pSrv);

    /**
     * Bind a UAV.
     * @param[in] bindLocation The bind-location in the block
     * @param[in] pSrv The unordered-access-view object to bind
     */
    bool setUav(const BindLocation& bindLocation, const ref<UnorderedAccessView>& pUav);

    /**
     * Bind an acceleration structure.
     * @param[in] bindLocation The bind-location in the block
     * @param[in] pAccl The acceleration structure object to bind
     * @return false if the binding location does not accept an acceleration structure, true otherwise.
     */
    bool setAccelerationStructure(const BindLocation& bindLocation, const ref<RtAccelerationStructure>& pAccl);

    /**
     * Get an SRV object.
     * @param[in] bindLocation The bind-location in the block
     * @return If the bind-location is valid, a shared pointer to the SRV. Otherwise returns nullptr
     */
    ref<ShaderResourceView> getSrv(const BindLocation& bindLocation) const;

    /**
     * Get a UAV object
     * @param[in] bindLocation The bind-location in the block
     * @return If the bind-location is valid, a shared pointer to the UAV. Otherwise returns nullptr
     */
    ref<UnorderedAccessView> getUav(const BindLocation& bindLocation) const;

    /**
     * Get an acceleration structure object.
     * @param[in] bindLocation The bind-location in the block
     * @return If the bind-location is valid, a shared pointer to the acceleration structure. Otherwise returns nullptr
     */
    ref<RtAccelerationStructure> getAccelerationStructure(const BindLocation& bindLocation) const;

    /**
     * Bind a sampler to the program in the global namespace.
     * @param[in] name The name of the sampler object in the shader
     * @param[in] pSampler The sampler object to bind
     * @return false if the sampler was not found in the program, otherwise true
     */
    bool setSampler(const std::string& name, const ref<Sampler>& pSampler);

    /**
     * Bind a sampler to the program in the global namespace.
     * @param[in] bindLocation The bind-location in the block
     * @param[in] pSampler The sampler object to bind
     * @return false if the sampler was not found in the program, otherwise true
     */
    bool setSampler(const BindLocation& bindLocation, const ref<Sampler>& pSampler);

    /**
     * Gets a sampler object.
     * @param[in] bindLocation The bind-location in the block
     * @return If the bind-location is valid, a shared pointer to the sampler. Otherwise returns nullptr
     */
    ref<Sampler> getSampler(const BindLocation& bindLocation) const;

    /**
     * Gets a sampler object.
     * @return If the name is valid, a shared pointer to the sampler. Otherwise returns nullptr
     */
    ref<Sampler> getSampler(const std::string& name) const;

    /**
     * Get the parameter block's reflection interface
     */
    ref<const ParameterBlockReflection> getReflection() const { return mpReflector; }

    /**
     * Get the block reflection type
     */
    ref<const ReflectionType> getElementType() const { return mpReflector->getElementType(); }

    /**
     * Get the size of the reflection type
     */
    size_t getElementSize() const;

    /**
     * Get offset of a uniform variable inside the block, given its name.
     */
    UniformShaderVarOffset getVariableOffset(const std::string& varName) const;

    /**
     * Get an initial var to the contents of this block.
     */
    ShaderVar getRootVar() const;

    /**
     * Try to find a shader var for a member of the block.
     *
     * Returns an invalid shader var if no such member is found.
     */
    ShaderVar findMember(const std::string& varName) const;

    /**
     * Try to find a shader var for a member of the block by index.
     *
     * Returns an invalid shader var if no such member is found.
     */
    ShaderVar findMember(uint32_t index) const;

    /**
     * Get the size of the parameter-block's buffer
     */
    size_t getSize() const;

    bool updateSpecialization() const;
    ref<const ParameterBlockReflection> getSpecializedReflector() const { return mpSpecializedReflector; }

    bool prepareDescriptorSets(CopyContext* pCopyContext);

    const ref<ParameterBlock>& getParameterBlock(uint32_t resourceRangeIndex, uint32_t arrayIndex) const;

    // Delete some functions. If they are not deleted, the compiler will try to convert the uints to string, resulting in runtime error
    ref<Sampler> getSampler(uint32_t) = delete;
    bool setSampler(uint32_t, ref<Sampler>) = delete;

    using SpecializationArgs = std::vector<slang::SpecializationArg>;
    void collectSpecializationArgs(SpecializationArgs& ioArgs) const;

    void const* getRawData() const;

    /**
     * Get the underlying constant buffer that holds the ordinary/uniform data for this block.
     * Be cautious with the returned buffer as it can be invalidated any time you set/bind something
     * to the parameter block (or one if its internal sub-blocks).
     */
    ref<Buffer> getUnderlyingConstantBuffer() const;

protected:
    ParameterBlock(
        ref<Device> pDevice,
        const ref<const ProgramVersion>& pProgramVersion,
        const ref<const ParameterBlockReflection>& pReflection
    );

    ParameterBlock(ref<Device> pDevice, const ref<const ProgramReflection>& pReflector);

    void initializeResourceBindings();
    void createConstantBuffers(const ShaderVar& var);
    static void prepareResource(CopyContext* pContext, Resource* pResource, bool isUav);

    /// Note: We hold an unowned pointer to the device but a strong pointer to the program version.
    /// We tie the lifetime of the program version to the lifetime of the parameter block.
    /// This is because the program version holds the reflection data for the parameter block.

    Device* mpDevice;
    ref<const ProgramVersion> mpProgramVersion;
    ref<const ParameterBlockReflection> mpReflector;
    mutable ref<const ParameterBlockReflection> mpSpecializedReflector;

    Slang::ComPtr<gfx::IShaderObject> mpShaderObject;
    std::map<gfx::ShaderOffset, ref<ParameterBlock>> mParameterBlocks;
    std::map<gfx::ShaderOffset, ref<ShaderResourceView>> mSRVs;
    std::map<gfx::ShaderOffset, ref<UnorderedAccessView>> mUAVs;
    std::map<gfx::ShaderOffset, ref<Resource>> mResources;
    std::map<gfx::ShaderOffset, ref<Sampler>> mSamplers;
    std::map<gfx::ShaderOffset, ref<RtAccelerationStructure>> mAccelerationStructures;
};

template<typename T>
bool ShaderVar::setImpl(const T& val) const
{
    return mpBlock->setVariable(mOffset, val);
}
} // namespace Falcor

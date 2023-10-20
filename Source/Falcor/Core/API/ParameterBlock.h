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
#include <string_view>
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

    //
    // Uniforms
    //

    /**
     * Set a variable into the block.
     * The function will validate that the value Type matches the declaration in the shader. If there's a mismatch, an error will be logged
     * and the call will be ignored.
     * @param[in] name The variable name. See notes about naming in the ConstantBuffer class description.
     * @param[in] value Value to set
     */
    template<typename T>
    void setVariable(std::string_view name, const T& value)
    {
        getRootVar()[name].set(value);
    }

    /**
     * Set a variable into the block.
     * The function will validate that the value Type matches the declaration in the shader. If there's a mismatch, an error will be logged
     * and the call will be ignored.
     * @param[in] offset The variable byte offset inside the buffer
     * @param[in] value Value to set
     */
    template<typename T>
    void setVariable(const BindLocation& bindLocation, const T& value);

    template<typename T>
    void setBlob(const BindLocation& bindLocation, const T& blob) const
    {
        setBlob(bindLocation, &blob, sizeof(blob));
    }

    void setBlob(const BindLocation& bindLocation, const void* pSrc, size_t size) { return setBlob(pSrc, bindLocation, size); }

    void setBlob(const void* pSrc, const BindLocation& bindLocation, size_t size);
    void setBlob(const void* pSrc, size_t offset, size_t size);

    //
    // Buffer
    //

    /**
     * Bind a buffer to a variable by name.
     * Throws an exception if the variable doesn't exist, there is a type-mismatch or the bind flags don't match the shader requirements.
     * @param[in] name The name of the variable to bind to.
     * @param[in] pBuffer The buffer object.
     */
    void setBuffer(std::string_view name, const ref<Buffer>& pBuffer);

    /**
     * Bind a buffer to a variable by bind location.
     * Throws an exception if the variable doesn't exist, there is a type-mismatch or the bind flags don't match the shader requirements.
     * @param[in] bindLocation The bind location of the variable to bind to.
     * @param[in] pBuffer The buffer object.
     */
    void setBuffer(const BindLocation& bindLocation, const ref<Buffer>& pBuffer);

    /**
     * Get the buffer bound to a variable by name.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] name The name of the variable.
     * @return The bound buffer or nullptr if none is bound.
     */
    ref<Buffer> getBuffer(std::string_view name) const;

    /**
     * Get the buffer bound to a variable by bind location.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] bindLocation The bind location of the variable.
     * @return The bound buffer or nullptr if none is bound.
     */
    ref<Buffer> getBuffer(const BindLocation& bindLocation) const;

    //
    // Texture
    //

    /**
     * Bind a texture to a variable by name.
     * Throws an exception if the variable doesn't exist, there is a type-mismatch or the bind flags don't match the shader requirements.
     * @param[in] name The name of the variable to bind to.
     * @param[in] pTexture The texture object.
     */
    void setTexture(std::string_view name, const ref<Texture>& pTexture);

    /**
     * Bind a texture to a variable by bind location.
     * Throws an exception if the variable doesn't exist, there is a type-mismatch or the bind flags don't match the shader requirements.
     * @param[in] bindLocation The bind location of the variable to bind to.
     * @param[in] pTexture The texture object.
     */
    void setTexture(const BindLocation& bindLocation, const ref<Texture>& pTexture);

    /**
     * Get the texture bound to a variable by name.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] name The name of the variable.
     * @return The bound texture or nullptr if none is bound.
     */
    ref<Texture> getTexture(std::string_view name) const;

    /**
     * Get the texture bound to a variable by bind location.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] bindLocation The bind location of the variable.
     * @return The bound texture or nullptr if none is bound.
     */
    ref<Texture> getTexture(const BindLocation& bindLocation) const;

    //
    // ResourceView
    //

    /**
     * Bind an SRV to a variable by name.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] name The name of the variable to bind to.
     * @param[in] pSrv The shader-resource-view object.
     */
    void setSrv(const BindLocation& bindLocation, const ref<ShaderResourceView>& pSrv);

    /**
     * Get the SRV bound to a variable by bind location.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] bindLocation The bind location of the variable.
     * @return The bound SRV or nullptr if none is bound.
     */
    ref<ShaderResourceView> getSrv(const BindLocation& bindLocation) const;

    /**
     * Bind a UAV to a variable by name.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] name The name of the variable to bind to.
     * @param[in] pSrv The unordered-access-view object.
     */
    void setUav(const BindLocation& bindLocation, const ref<UnorderedAccessView>& pUav);

    /**
     * Get the UAV bound to a variable by bind location.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] bindLocation The bind location of the variable.
     * @return The bound UAV or nullptr if none is bound.
     */
    ref<UnorderedAccessView> getUav(const BindLocation& bindLocation) const;

    /**
     * Bind an acceleration strcture to a variable by name.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] name The name of the variable to bind to.
     * @param[in] pAccel The acceleration structure object.
     */
    void setAccelerationStructure(const BindLocation& bindLocation, const ref<RtAccelerationStructure>& pAccl);

    /**
     * Get the acceleration structure bound to a variable by bind location.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] bindLocation The bind location of the variable.
     * @return The bound acceleration structure or nullptr if none is bound.
     */
    ref<RtAccelerationStructure> getAccelerationStructure(const BindLocation& bindLocation) const;

    //
    // Sampler
    //

    /**
     * Bind a sampler to a variable by name.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] name The name of the variable to bind to.
     * @param[in] pSampler The sampler object.
     */
    void setSampler(std::string_view name, const ref<Sampler>& pSampler);

    /**
     * Bind a sampler to a variable by bind location.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] bindLocation The bind location of the variable to bind to.
     * @param[in] pSampler The sampler object.
     */
    void setSampler(const BindLocation& bindLocation, const ref<Sampler>& pSampler);

    /**
     * Get the sampler bound to a variable by bind location.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] bindLocation The bind location of the variable.
     * @return The bound sampler or nullptr if none is bound.
     */
    ref<Sampler> getSampler(const BindLocation& bindLocation) const;

    /**
     * Get the sampler bound to a variable by name.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] name The name of the variable.
     * @return The bound sampler or nullptr if none is bound.
     */
    ref<Sampler> getSampler(std::string_view name) const;

    //
    // ParameterBlock
    //

    /**
     * Bind a parameter block to a variable by name.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] name The name of the variable to bind to.
     * @param[in] pBlock The parameter block.
     */
    void setParameterBlock(std::string_view name, const ref<ParameterBlock>& pBlock);

    /**
     * Bind a parameter block to a variable by bind location.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] bindLocation The bind location of the variable to bind to.
     * @param[in] pBlock The parameter block object.
     */
    void setParameterBlock(const BindLocation& bindLocation, const ref<ParameterBlock>& pBlock);

    /**
     * Get the parameter block bound to a variable by name.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] name The name of the variable.
     * @return The bound parameter block or nullptr if none is bound.
     */
    ref<ParameterBlock> getParameterBlock(std::string_view name) const;

    /**
     * Get the parameter block bound to a variable by bind location.
     * Throws an exception if the variable doesn't exist or there is a type-mismatch.
     * @param[in] bindLocation The bind location of the variable.
     * @return The bound parameter block or nullptr if none is bound.
     */
    ref<ParameterBlock> getParameterBlock(const BindLocation& bindLocation) const;

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
    TypedShaderVarOffset getVariableOffset(std::string_view varName) const;

    /**
     * Get an initial var to the contents of this block.
     */
    ShaderVar getRootVar() const;

    /**
     * Try to find a shader var for a member of the block.
     *
     * Returns an invalid shader var if no such member is found.
     */
    ShaderVar findMember(std::string_view varName) const;

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

    using SpecializationArgs = std::vector<slang::SpecializationArg>;
    void collectSpecializationArgs(SpecializationArgs& ioArgs) const;

    void const* getRawData() const;

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
void ShaderVar::setImpl(const T& val) const
{
    mpBlock->setVariable(mOffset, val);
}
} // namespace Falcor

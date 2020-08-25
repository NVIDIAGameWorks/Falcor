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
#include "ProgramReflection.h"
#include "Core/API/Texture.h"
#include "Core/API/Sampler.h"
#include "Core/API/Buffer.h"

namespace Falcor
{
    class ParameterBlock;
    template<typename T>
    class ParameterBlockSharedPtr;

    /** A "pointer" to a shader variable stored in some parameter block.

    A `ShaderVar` works like a pointer to the data "inside" a `ParameterBlock`.
    It keeps track of three things:

    1. The parameter block that is being pointed into
    2. An offset into the data of that parameter block
    3. The type of the data at that offset

    Typically a `ShaderVar` is created using the `getRootVar()` operation
    on `ParameterBlock`, which yields a shader variable that points to
    the entire "contents" of the parameter block.

    Given a `ShaderVar` that represents a value with `struct` or
    array type, we can use `operator[]` to get a shader variable
    that points to a single `struct` field or array element:

        // Shader code has `MyStruct myVar[10];`

        ShaderVar myVar = pObj["myVar"];                // works like &myVar
        ShaderVar arrayElement = myVar[2];              // works like &myVar[2]
        ShaderVar someField = arrayElement["someField"] // works like &myVar[2].someField

    Once you have a `ShaderVar` that refers to a simple value you
    want to set, you can do so with either an explicit `set*()` function
    or an overload of `operator=`:

        someField = float3(0);

        pObj["someTexture"].setTexture(pMyTexture);
    */
    struct dlldecl ShaderVar
    {
        /** Create a null/invalid shader variable pointer.
        */
        ShaderVar();

        /** Create a null/invalid shader variable pointer.
        */
        ShaderVar(nullptr_t) : ShaderVar() {};

        /** Copy constructor.
        */
        ShaderVar(const ShaderVar& other);

        /** Create a shader variable pointer into `pObject` at the given `offset`.
        */
        ShaderVar(ParameterBlock* pObject, const TypedShaderVarOffset& offset);

        /** Create a shader variable pointer to the content of `pObject`.
        */
        explicit ShaderVar(ParameterBlock* pObject);

        /** Check if this shader variable pointer is valid/non-null.
        */
        bool isValid() const;

        /** Get the type data this shader variable points at.

        For an invalid/null shader variable the result will be null.

        */
        ReflectionType::SharedConstPtr getType() const;

        /** Get the offset that this shader variable points to inside the parameter block.
        */
        TypedShaderVarOffset getOffset() const { return mOffset; }

        /** Get the byte offset that this shader variable points to inside the parameter block.
            Note: If the type of the value being pointed at includes anything other than ordinary/uniform data, then this byte offset will not provide
            complete enough information to re-create the same `ShaderVar` later.
        */
        size_t getByteOffset() const;

        /** Get a shader variable pointer to a sub-field.

            This shader variable must point to a value of `struct` type, with a field matching the given `name`.
            If this shader variable points at a constant buffer or parameter block, then the lookup will proceed in the contents of that block.
            Otherwise an error is logged and an invalid `ShaderVar` is returned.
        */
        ShaderVar operator[](const char* name) const;

        /** Get a shader variable pointer to a sub-field.

            This shader variable must point to a value of `struct` type, with a field matching the given `name`.
            If this shader variable points at a constant buffer or parameter block, then the lookup will proceed in the contents of that block.
            Otherwise an error is logged and an invalid `ShaderVar` is returned.
        */
        ShaderVar operator[](const std::string& name) const;

        /** Get a shader variable pointer to an element or sub-field.

            This operation is valid in two cases:
            1) This shader variable points at a value of array type, and the `index` is in range for the array. The result
               is a shader variable that points to a single array element.
            2) This shader variable points at a value of `struct` type, and `index` is in range for the number of fields in
               the `struct`. The result is a shader variable that points to a single `struct` field.
            If this shader variable points at a constant buffer or parameter block, then the lookup will proceed in the contents of that block.

            Otherwise an error is logged and an invalid `ShaderVar` is returned.
        */
        ShaderVar operator[](size_t index) const;

        /** Try to get a variable for a member/field.

        Unlike `operator[]`, a `findMember` operation does not
        log an error if a member of the given name cannot be found.
        */
        ShaderVar findMember(const std::string& name) const;

        /** Try to get a variable for a member/field, by index.

        Unlike `operator[]`, a `findMember` operation does not
        log an error if a member cannot be found at the given index.
        */
        ShaderVar findMember(uint32_t index) const;

        /** Set the value of the data pointed to by this shader variable.
            Returns `true` if successful. Logs and error and returns `false` if the given `val` does not have a suitable type for the value
            pointed to by this shader variable.
        */
        template<typename T> bool set(const T& val) const
        {
            return setImpl<T>(val);
        }

        /** Assign raw binary data to the pointed-to value.

            This operation will only assign to the ordinary/"uniform" data pointed to by this shader variable, and will not affect any
            nested variables of texture/buffer/sampler types.
        */
        bool setBlob(void const* data, size_t size) const;

        /** Assign raw binary data to the pointed-to value.
            This is a convenience form for `setBlob(&val, sizeof(val)`.
        */
        template<typename T>
        bool setBlob(const T& val) const
        {
            return setBlob(&val, sizeof(val));
        }

        /** Set a buffer into this variable
            Logs an error and returns `false` if this variable doesn't point at a buffer
        */
        bool setBuffer(const Buffer::SharedPtr& pBuffer) const;

        /** Set the texture that this variable points to.
            Logs an error and returns `false` if this variable doesn't point at a texture.
        */
        bool setTexture(const Texture::SharedPtr& pTexture) const;

        /** Get the texture that this variable points to.
            Logs an error and returns null if this variable doesn't point at a texture.
        */
        Texture::SharedPtr getTexture() const;

        /** Set the sampler that this variable points to.
            Logs an error and returns `false` if this variable doesn't point at a sampler.
        */
        bool setSampler(const Sampler::SharedPtr& pSampler) const;

        /** Get the sampler that this variable points to.
            Logs an error and returns null if this variable doesn't point at a sampler.
        */
        Sampler::SharedPtr getSampler() const;

        /** Get the buffer that this variable points to.
            Logs an error and returns nullptr if this variable doesn't point at a buffer.
        */
        Buffer::SharedPtr getBuffer() const;

        /** Set the shader resource view that this variable points to.
            Logs an error and returns `false` if this variable doesn't point at a shader resource view.
        */
        bool setSrv(const ShaderResourceView::SharedPtr& pSrv) const;

        /** Get the shader resource view that this variable points to.
            Logs an error and returns null if this variable doesn't point at a shader resource view.
        */
        ShaderResourceView::SharedPtr getSrv() const;

        /** Set the unordered access view that this variable points to.
            Logs an error and returns `false` if this variable doesn't point at an unordered access view.
        */
        bool setUav(const UnorderedAccessView::SharedPtr& pUav) const;

        /** Get the unordered access view that this variable points to.
            Logs an error and returns null if this variable doesn't point at an unordered access view.
        */
        UnorderedAccessView::SharedPtr getUav() const;

        /** Set the parameter block that this variable points to.
            Logs an error and returns `false` if this variable doesn't point at a parameter block.
        */
        bool setParameterBlock(const std::shared_ptr<ParameterBlock>& pBlock) const;

        /** Get the parameter block that this variable points to.
            Logs an error and returns null if this variable doesn't point at a parameter block.
        */
        std::shared_ptr<ParameterBlock> getParameterBlock() const;

        /** Set the value of the data pointed to by this shader variable.

            This operator allows assignment syntax to be used in place of
            the `set()` method. The following two statements are equivalent:
                myShaderVar["someField"].set(float4(0));
                myShaderVar["someField"] = float4(0);
        */
        template<typename T>
        void operator=(const T& val)
        {
            setImpl(val);
        }

        /** Set the value of the data pointed to by this shader variable.

            This operator allows assignment syntax to be used in place of the `set()` method. The following two statements are equivalent:

                myShaderVar["someField"].set(float4(0));
                myShaderVar["someField"] = float4(0);

            Note: because this operation modified the data "pointed at" by the shader variable, rather than the shader variable itself, assignment is allowed on a `const ShaderVar`.
        */
        template<typename T>
        void operator=(const T& val) const
        {
            setImpl(val);
        }

        /** Implicit conversion from a shader variable to its offset information.

            This operation allows the offset information for a shader variable to be queried easily using the `[]` sugar:
                TypedShaderVarOffset myVarLoc = pVars["myVar"];
                ...
                pVars[myVarLoc] = someValue

            Note that the returned offset information only retains the offset into the leaf-most parameter block (constant buffer or parameter block).
            Users must take care when using an offset that they apply the offset to the correct object:

                auto pPerFrameCB = pVars["PerFrameCB"];
                TypedShaderVarOffset myVarLoc = pPerFrameCB["myVar"];
                ...
                pVars[myVarLoc] = someValue; // CRASH!
        */
        operator TypedShaderVarOffset() const { return mOffset; }

        /** Implicit conversion from a shader variable to its offset information.

            This operation allows the offset information for a shader variable to be queried easily using the `[]` sugar:

                UniformShaderVarOffset myVarLoc = pVars["myVar"];
                ...
                pVars[myVarLoc] = someValue

            Note that the returned offset information only retains the offset into the leaf-most parameter block (constant buffer or parameter block).
            Users must take care when using an offset that they apply the offset to the correct object:

                auto pPerFrameCB = pVars["PerFrameCB"];
                UniformShaderVarOffset myVarLoc = pPerFrameCB["myVar"];
                ...
                pVars[myVarLoc] = someValue; // CRASH!
        */
        operator UniformShaderVarOffset() const;

        /** Create a shader variable that points to some pre-computed `offset` from this one.

        This operation assumes that the provided `offset` has been appropriately computed based on a type that matches what this shader variable points to.
        The resulting shader variable will have the type encoded in `offset`, and will have an offset that is the sum of this variables offset and the provided `offset`.
        */
        ShaderVar operator[](TypedShaderVarOffset const& offset) const;

        /** Create a shader variable that points to some pre-computed `offset` from this one.

            This operation assumes that the provided `offset` has been appropriately computed based on a type that matches what this shader variable points to.

            Because a `UniformShaderVarOffset` does not encode type information, this operation will search for a field/element matching the given `offset` and use its type
            information in the resulting shader variable. If no appropriate field/element can be found, an error will be logged.
        */
        ShaderVar operator[](UniformShaderVarOffset const& offset) const;

        /** Implicit conversion from a shader variable to a texture.
            This operation allows a bound texture to be queried using the `[]` syntax:
                pTexture = pVars["someTexture"];
        */
        operator Texture::SharedPtr() const;

        /** Implicit conversion from a shader variable to a sampler.
            This operation allows a bound sampler to be queried using the `[]` syntax:
                pSampler = pVars["someSampler"];
        */
        operator Sampler::SharedPtr() const;

        /** Implicit conversion from a shader variable to a buffer.
            This operation allows a bound buffer to be queried using the `[]` syntax:
                pBuffer = pVars["someBuffer"];
        */
        operator Buffer::SharedPtr() const;

        /** Get access to the underlying bytes of the variable.

            This operation must be used with caution; the caller takes all responsibility for validation.

            Note: if a caller uses the resulting pointer to write to the variable (e.g.,
            by casting away the `const`-ness, then the underlying `ParameterBlock` will
            not automatically be marked dirty, and it is possible that the effects of that
            write will not be visible.
        */
        void const* getRawData() const;

    private:
        friend class VariablesBufferUI;
        /** The parameter block that is being pointed into.

            Note: this is an unowned pointer, so it is *not* safe to hold onto a `ShaderVar` for long periods
            of time where the object it points into might get released. This is a concession to performance,
            since we do not want to perform reference-counting each and every time a `ShaderVar` gets created or destroyed.
        */
        ParameterBlock*   mpBlock;

        /** The offset into the object where this variable points.

            This field encodes both the offset information and the type of the variable.

            TODO(tfoley): This field technically retains a reference count on the type, which shouldn't be
            needed because the original object should keep its type (and thus its field types) alive.
        */
        TypedShaderVarOffset    mOffset;

#if _LOG_ENABLED
        /** A string representation of the path into the object.

            This field is retained for debugging purposes only, so that better error messages can be reported on things like missing
            or incorrectly typed fields.
        */
        std::string mDebugName;
#endif

        bool setImpl(const Texture::SharedPtr& pTexture) const;
        bool setImpl(const Sampler::SharedPtr& pSampler) const;
        bool setImpl(const Buffer::SharedPtr& pBuffer) const;
        bool setImpl(const std::shared_ptr<ParameterBlock>& pBlock) const;

        template<typename T>
        bool setImpl(const ParameterBlockSharedPtr<T>& pBlock) const
        {
            return setImpl(std::static_pointer_cast<ParameterBlock>(pBlock));
        }

        template<typename T> bool setImpl(const T& val) const;
    };
}

#include "Core/BufferTypes/ParameterBlock.h"

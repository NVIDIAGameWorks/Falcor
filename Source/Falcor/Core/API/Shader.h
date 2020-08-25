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
#include <map>
#include <initializer_list>

struct ISlangBlob;

namespace Falcor
{
    /** Minimal smart pointer for working with COM objects.
    */
    template<typename T>
    struct dlldecl ComPtr
    {
    public:
        /// Type of the smart pointer itself
        typedef ComPtr ThisType;

        /// Initialize to a null pointer.
        ComPtr() : mpObject(nullptr) {}
        ComPtr(nullptr_t) : mpObject(nullptr) {}

        /// Release reference to the pointed-to object.
        ~ComPtr() { if (mpObject) (mpObject)->Release(); }

        /// Add a new reference to an existing object.
        explicit ComPtr(T* pObject) : mpObject(pObject) { if (pObject) (pObject)->AddRef(); }

        /// Add a new reference to an existing object.
        ComPtr(const ThisType& rhs) : mpObject(rhs.mpObject) { if (mpObject) (mpObject)->AddRef(); }

        /// Add a new reference to an existing object
        T* operator=(T* in)
        {
            if(in) in->AddRef();
            if(mpObject) mpObject->Release();
            mpObject = in;
            return in;
        }

        /// Add a new reference to an existing object
        const ThisType& operator=(const ThisType& rhs)
        {
            if(rhs.mpObject) rhs.mpObject->AddRef();
            if(mpObject) mpObject->Release();
            mpObject = rhs.mpObject;
            return *this;
        }

        /// Transfer ownership of a reference.
        ComPtr(ThisType&& rhs) : mpObject(rhs.mpObject) { rhs.mpObject = nullptr; }

        /// Transfer ownership of a reference.
        ComPtr& operator=(ThisType&& rhs) { T* swap = mpObject; mpObject = rhs.mpObject; rhs.mpObject = swap; return *this; }

        /// Clear out object pointer.
        void setNull()
        {
            if( mpObject )
            {
                mpObject->Release();
                mpObject = nullptr;
            }
        }

        /// Swap pointers with another reference.
        void swap(ThisType& rhs)
        {
            T* tmp = mpObject;
            mpObject = rhs.mpObject;
            rhs.mpObject = tmp;
        }

        /// Get the underlying object pointer.
        T* get() const { return mpObject; }

        /// Cast to object pointer type.
        operator T*() const { return mpObject; }

        /// Access members of underlying object.
        T* operator->() const { return mpObject; }

        /// Dereference underlying pointer.
        T& operator*() { return *mpObject; }

        /// Transfer ownership of reference to the caller.
        T* detach() { T* ptr = mpObject; mpObject = nullptr; return ptr; }

        /// Transfer ownership of reference from the caller.
        void attach(T* in) { T* old = mpObject; mpObject = in; if(old) old->Release(); }

        /// Get a writable reference suitable for use as a function output argument.
        T** writeRef() { setNull(); return &mpObject; }

        /// Get a readable reference suitable for use as a function input argument.
        T*const* readRef() const { return &mpObject; }

    protected:
        // Disabled: do not take the address of a smart pointer.
        T** operator&();

        /// The underlying raw object pointer
        T* mpObject;
    };

    /** Low-level shader object
        This class abstracts the API's shader creation and management
    */
    class dlldecl Shader : public std::enable_shared_from_this<Shader>
    {
    public:
        using SharedPtr = std::shared_ptr<Shader>;
        using SharedConstPtr = std::shared_ptr<const Shader>;
        using ApiHandle = ShaderHandle;

        typedef ComPtr<ISlangBlob> Blob;

        enum class CompilerFlags
        {
            None                        = 0x0,
            TreatWarningsAsErrors       = 0x1,
            DumpIntermediates           = 0x2,
            FloatingPointModeFast       = 0x4,
            FloatingPointModePrecise    = 0x8,
            GenerateDebugInfo           = 0x10,
        };

        class DefineList : public std::map<std::string, std::string>
        {
        public:
            /** Adds a macro definition. If the macro already exists, it will be replaced.
                \param[in] name The name of macro.
                \param[in] value Optional. The value of the macro.
                \return The updated list of macro definitions.
            */
            DefineList& add(const std::string& name, const std::string& val = "") { (*this)[name] = val; return *this; }

            /** Removes a macro definition. If the macro doesn't exist, the call will be silently ignored.
                \param[in] name The name of macro.
                \return The updated list of macro definitions.
            */
            DefineList& remove(const std::string& name) { (*this).erase(name); return *this; }

            /** Add a define list to the current list
            */
            DefineList& add(const DefineList& dl) { for (const auto& p : dl) add(p.first, p.second); return *this; }

            /** Remove a define list from the current list
            */
            DefineList& remove(const DefineList& dl) { for (const auto& p : dl) remove(p.first); return *this; }

            DefineList() = default;
            DefineList(std::initializer_list<std::pair<const std::string, std::string>> il) : std::map<std::string, std::string>(il) {}
        };

        /** Create a shader object
            \param[in] shaderBlog A blob containing the shader code
            \param[in] Type The Type of the shader
            \param[out] log This string will contain the error log message in case shader compilation failed
            \return If success, a new shader object, otherwise nullptr
        */
        static SharedPtr create(const Blob& shaderBlob, ShaderType type, std::string const&  entryPointName, CompilerFlags flags, std::string& log)
        {
            SharedPtr pShader = SharedPtr(new Shader(type));
            pShader->mEntryPointName = entryPointName;
            return pShader->init(shaderBlob, entryPointName, flags, log) ? pShader : nullptr;
        }

        virtual ~Shader();

        /** Get the API handle.
        */
        const ApiHandle& getApiHandle() const { return mApiHandle; }

        /** Get the shader Type
        */
        ShaderType getType() const { return mType; }

        /** Get the name of the entry point.
        */
        const std::string& getEntryPoint() const { return mEntryPointName; }

#ifdef FALCOR_D3D12
        ID3DBlobPtr getD3DBlob() const;
#endif

    protected:
        // API handle depends on the shader Type, so it stored be stored as part of the private data
        bool init(const Blob& shaderBlob, const std::string&  entryPointName, CompilerFlags flags, std::string& log);
        Shader(ShaderType Type);
        ShaderType mType;
        std::string mEntryPointName;
        ApiHandle mApiHandle;
        void* mpPrivateData = nullptr;
    };
    enum_class_operators(Shader::CompilerFlags);
}

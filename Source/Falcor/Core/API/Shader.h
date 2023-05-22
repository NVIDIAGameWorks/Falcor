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
#include "Handles.h"
#include "Core/Macros.h"
#include "Core/Assert.h"
#include "Core/Object.h"

#include <initializer_list>
#include <memory>
#include <map>
#include <string>
#include <cstddef> // std::nullptr_t

struct ISlangBlob;

namespace Falcor
{
/**
 * Falcor shader types
 */
enum class ShaderType
{
    Vertex,   ///< Vertex shader
    Pixel,    ///< Pixel shader
    Geometry, ///< Geometry shader
    Hull,     ///< Hull shader (AKA Tessellation control shader)
    Domain,   ///< Domain shader (AKA Tessellation evaluation shader)
    Compute,  ///< Compute shader

    RayGeneration, ///< Ray generation shader
    Intersection,  ///< Intersection shader
    AnyHit,        ///< Any hit shader
    ClosestHit,    ///< Closest hit shader
    Miss,          ///< Miss shader
    Callable,      ///< Callable shader
    Count          ///< Shader Type count
};

/**
 * Converts ShaderType enum elements to a string.
 * @param[in] type Type to convert to string
 * @return Shader type as a string
 */
inline const std::string to_string(ShaderType type)
{
    switch (type)
    {
    case ShaderType::Vertex:
        return "vertex";
    case ShaderType::Pixel:
        return "pixel";
    case ShaderType::Hull:
        return "hull";
    case ShaderType::Domain:
        return "domain";
    case ShaderType::Geometry:
        return "geometry";
    case ShaderType::Compute:
        return "compute";
    case ShaderType::RayGeneration:
        return "raygeneration";
    case ShaderType::Intersection:
        return "intersection";
    case ShaderType::AnyHit:
        return "anyhit";
    case ShaderType::ClosestHit:
        return "closesthit";
    case ShaderType::Miss:
        return "miss";
    case ShaderType::Callable:
        return "callable";
    default:
        FALCOR_UNREACHABLE();
        return "";
    }
}

/**
 * Forward declaration of backend implementation-specific Shader data.
 */
struct ShaderData;

/**
 * Low-level shader object
 * This class abstracts the API's shader creation and management
 */
class FALCOR_API Shader : public Object
{
public:
    typedef Slang::ComPtr<ISlangBlob> Blob;

    enum class CompilerFlags
    {
        None = 0x0,
        TreatWarningsAsErrors = 0x1,
        DumpIntermediates = 0x2,
        FloatingPointModeFast = 0x4,
        FloatingPointModePrecise = 0x8,
        GenerateDebugInfo = 0x10,
        MatrixLayoutColumnMajor = 0x20, // Falcor is using row-major, use this only to compile stand-alone external shaders.
    };

    struct BlobData
    {
        const void* data;
        size_t size;
    };

    class DefineList : public std::map<std::string, std::string>
    {
    public:
        /**
         * Adds a macro definition. If the macro already exists, it will be replaced.
         * @param[in] name The name of macro.
         * @param[in] value Optional. The value of the macro.
         * @return The updated list of macro definitions.
         */
        DefineList& add(const std::string& name, const std::string& val = "")
        {
            (*this)[name] = val;
            return *this;
        }

        /**
         * Removes a macro definition. If the macro doesn't exist, the call will be silently ignored.
         * @param[in] name The name of macro.
         * @return The updated list of macro definitions.
         */
        DefineList& remove(const std::string& name)
        {
            (*this).erase(name);
            return *this;
        }

        /**
         * Add a define list to the current list
         */
        DefineList& add(const DefineList& dl)
        {
            for (const auto& p : dl)
                add(p.first, p.second);
            return *this;
        }

        /**
         * Remove a define list from the current list
         */
        DefineList& remove(const DefineList& dl)
        {
            for (const auto& p : dl)
                remove(p.first);
            return *this;
        }

        DefineList() = default;
        DefineList(std::initializer_list<std::pair<const std::string, std::string>> il) : std::map<std::string, std::string>(il) {}
    };

    /**
     * Representing a shader implementation of an interface.
     * When linked into a `ProgramVersion`, the specialized shader will contain
     * the implementation of the specified type in a dynamic dispatch function.
     */
    struct TypeConformance
    {
        std::string mTypeName;
        std::string mInterfaceName;
        TypeConformance() = default;
        TypeConformance(const std::string& typeName, const std::string& interfaceName) : mTypeName(typeName), mInterfaceName(interfaceName)
        {}
        bool operator<(const TypeConformance& other) const
        {
            return mTypeName < other.mTypeName || (mTypeName == other.mTypeName && mInterfaceName < other.mInterfaceName);
        }
        bool operator==(const TypeConformance& other) const
        {
            return mTypeName == other.mTypeName && mInterfaceName == other.mInterfaceName;
        }
        struct HashFunction
        {
            size_t operator()(const TypeConformance& conformance) const
            {
                size_t hash = std::hash<std::string>()(conformance.mTypeName);
                hash = hash ^ std::hash<std::string>()(conformance.mInterfaceName);
                return hash;
            }
        };
    };

    class TypeConformanceList : public std::map<TypeConformance, uint32_t>
    {
    public:
        /**
         * Adds a type conformance. If the type conformance exists, it will be replaced.
         * @param[in] typeName The name of the implementation type.
         * @param[in] interfaceName The name of the interface type.
         * @param[in] id Optional. The id representing the implementation type for this interface. If it is -1, Slang will automatically
         * assign a unique Id for the type.
         * @return The updated list of type conformances.
         */
        TypeConformanceList& add(const std::string& typeName, const std::string& interfaceName, uint32_t id = -1)
        {
            (*this)[TypeConformance(typeName, interfaceName)] = id;
            return *this;
        }

        /**
         * Removes a type conformance. If the type conformance doesn't exist, the call will be silently ignored.
         * @param[in] typeName The name of the implementation type.
         * @param[in] interfaceName The name of the interface type.
         * @return The updated list of type conformances.
         */
        TypeConformanceList& remove(const std::string& typeName, const std::string& interfaceName)
        {
            (*this).erase(TypeConformance(typeName, interfaceName));
            return *this;
        }

        /**
         * Add a type conformance list to the current list
         */
        TypeConformanceList& add(const TypeConformanceList& cl)
        {
            for (const auto& p : cl)
                add(p.first.mTypeName, p.first.mInterfaceName, p.second);
            return *this;
        }

        /**
         * Remove a type conformance list from the current list
         */
        TypeConformanceList& remove(const TypeConformanceList& cl)
        {
            for (const auto& p : cl)
                remove(p.first.mTypeName, p.first.mInterfaceName);
            return *this;
        }

        TypeConformanceList() = default;
        TypeConformanceList(std::initializer_list<std::pair<const TypeConformance, uint32_t>> il) : std::map<TypeConformance, uint32_t>(il)
        {}
    };

    /**
     * Create a shader object
     * @param[in] linkedSlangEntryPoint The Slang IComponentType that defines the shader entry point.
     * @param[in] type The Type of the shader
     * @param[out] log This string will contain the error log message in case shader compilation failed
     * @return If success, a new shader object, otherwise nullptr
     */
    static ref<Shader> create(
        Slang::ComPtr<slang::IComponentType> linkedSlangEntryPoint,
        ShaderType type,
        const std::string& entryPointName,
        CompilerFlags flags,
        std::string& log
    )
    {
        ref<Shader> pShader = ref<Shader>(new Shader(type));
        pShader->mEntryPointName = entryPointName;
        return pShader->init(linkedSlangEntryPoint, entryPointName, flags, log) ? pShader : nullptr;
    }

    virtual ~Shader();

    /**
     * Get the shader Type
     */
    ShaderType getType() const { return mType; }

    /**
     * Get the name of the entry point.
     */
    const std::string& getEntryPoint() const { return mEntryPointName; }

    BlobData getBlobData() const;

protected:
    // API handle depends on the shader Type, so it stored be stored as part of the private data
    bool init(
        Slang::ComPtr<slang::IComponentType> linkedSlangEntryPoint,
        const std::string& entryPointName,
        CompilerFlags flags,
        std::string& log
    );
    Shader(ShaderType Type);
    ShaderType mType;
    std::string mEntryPointName;
    std::unique_ptr<ShaderData> mpPrivateData;
};
FALCOR_ENUM_CLASS_OPERATORS(Shader::CompilerFlags);
} // namespace Falcor

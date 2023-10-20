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
#include "MaterialTypeRegistry.h"
#include "MaterialData.slang"

// Include parameter layouts.
#include "Scene/Material/StandardMaterial.h"
#include "Scene/Material/StandardMaterialParamLayout.slang"
#include "Scene/Material/PBRT/PBRTDiffuseMaterial.h"
#include "Scene/Material/PBRT/PBRTDiffuseMaterialParamLayout.slang"
#include "Scene/Material/PBRT/PBRTConductorMaterial.h"
#include "Scene/Material/PBRT/PBRTConductorMaterialParamLayout.slang"

#include <mutex>
#include <map>

namespace Falcor
{
namespace
{

class MaterialTypeRegistry
{
public:
    static MaterialTypeRegistry& get()
    {
        static MaterialTypeRegistry sInstance; // TODO: REMOVEGLOBAL
        return sInstance;
    }

    MaterialType registerType(std::string typeName)
    {
        std::lock_guard<std::mutex> l(mMaterialTypeNameMutex);

        // Return existing material type if type name is already registered.
        if (auto it = mMaterialName2Type.find(typeName); it != mMaterialName2Type.end())
        {
            return it->second;
        }

        // Register new material type with the given type name.
        FALCOR_ASSERT_LT(mNextMaterialTypeID, 1u << MaterialHeader::kMaterialTypeBits);
        if (mNextMaterialTypeID >= (1u << MaterialHeader::kMaterialTypeBits))
        {
            FALCOR_THROW("Registered material {} would receive MaterialType {}, exceeding the maximum limit {} (given by MaterialHeader::kMaterialTypeBits)",
                typeName, mNextMaterialTypeID, (1u << MaterialHeader::kMaterialTypeBits));
        }

        MaterialType type = MaterialType(mNextMaterialTypeID++);
        mMaterialType2Name[type] = typeName;
        mMaterialName2Type[typeName] = type;

        return type;
    }

    std::string to_string(MaterialType type) const
    {
        std::lock_guard<std::mutex> l(mMaterialTypeNameMutex);
        auto it = mMaterialType2Name.find(type);
        if (it != mMaterialType2Name.end())
            return it->second;
        FALCOR_THROW("Invalid material type: {}", int(type));
    }

    size_t getTypeCount() const
    {
        std::lock_guard<std::mutex> l(mMaterialTypeNameMutex);
        return mNextMaterialTypeID;
    }

    MaterialParamLayout getMaterialParamLayout(MaterialType type) const
    {
        switch (type)
        {
        case MaterialType::Standard:
            return StandardMaterialParamLayout::layout();
        case MaterialType::PBRTDiffuse:
            return PBRTDiffuseMaterialParamLayout::layout();
        case MaterialType::PBRTConductor:
            return PBRTConductorMaterialParamLayout::layout();
        }
        return {};
    }

private:
    MaterialTypeRegistry()
    {
        // Register names for all built-in material types.
        mMaterialType2Name = {
            { MaterialType::Unknown, "Unknown" },
            { MaterialType::Standard, "Standard" },
            { MaterialType::Cloth, "Cloth" },
            { MaterialType::Hair, "Hair" },
            { MaterialType::MERL, "MERL" },
            { MaterialType::MERLMix, "MERLMix" },
            { MaterialType::PBRTDiffuse, "PBRTDiffuse" },
            { MaterialType::PBRTDiffuseTransmission, "PBRTDiffuseTransmission" },
            { MaterialType::PBRTConductor, "PBRTConductor" },
            { MaterialType::PBRTDielectric, "PBRTDielectric" },
            { MaterialType::PBRTCoatedConductor, "PBRTCoatedConductor" },
            { MaterialType::PBRTCoatedDiffuse, "PBRTCoatedDiffuse" },
            { MaterialType::RGL, "RGL" },
        };

        for (const auto& [type, name] : mMaterialType2Name)
        {
            mMaterialName2Type[name] = type;
        }
        FALCOR_CHECK(mMaterialType2Name.size() == mMaterialName2Type.size(), "Material type names must be unique.");
    }

    mutable std::mutex mMaterialTypeNameMutex; ///< Mutex to registering new material types
    uint32_t mNextMaterialTypeID { uint32_t(MaterialType::BuiltinCount) + 1 }; ///< Holds id of the next free material type, used when creating material types dynamically
    std::map<MaterialType, std::string> mMaterialType2Name; ///< Map for to_string names
    std::map<std::string, MaterialType> mMaterialName2Type; ///< Map from name to type for deduplication when creating material types.
};

}

MaterialType registerMaterialType(std::string typeName)
{
    return MaterialTypeRegistry::get().registerType(std::move(typeName));
}

std::string to_string(MaterialType type)
{
    return MaterialTypeRegistry::get().to_string(type);
}

size_t getMaterialTypeCount()
{
    return MaterialTypeRegistry::get().getTypeCount();
}

MaterialParamLayout getMaterialParamLayout(MaterialType type)
{
    return MaterialTypeRegistry::get().getMaterialParamLayout(type);
}

} // namespace Falcor

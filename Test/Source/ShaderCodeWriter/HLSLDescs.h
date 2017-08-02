#pragma once
#include "CommonShaderDescs.h"


namespace Falcor
{
    //
    namespace HLSLDescs
    {
        enum class HLSLBaseResourceType : uint32_t;
        enum class HLSLBufferType : uint32_t;
        struct HLSLResourcesDesc;

        //  HLSL Base Resource Type.
        enum class HLSLBaseResourceType : uint32_t
        {
            BasicType = 0u,
            TextureType = 1u,
            SamplerType = 2u,
            BufferType = 3u
        };

        //  Buffer Types.
        enum class HLSLBufferType : uint32_t
        {
            Unknown = 0u,
            Buffer = 1u,
            StructuredBuffer = 2u,
            RawBuffer = 3u,
            ConstantBuffer = 4u,
        };

        //  HLSL Semantic Resources.
        struct HLSLSemanticDesc
        {
            //  Whether or not we have a semantic value for the variable.s
            bool hasSemanticValue = false;

            //  Semantic Value.
            std::string semanticValue = "";
        };

        //  Shader Struct Descs.
        struct HLSLStructDesc
        {
            //  Whether or not to define a new struct.
            bool definesStructType = true;

            //  Use the Struct Type.
            bool usesStructType = true;

            //  The Name of the Struct Variable Type.
            std::string structVariableType = "";

            //  The Array of Shader Resources.
            std::vector<HLSLResourcesDesc> structVariables = {};
        };


        //
        struct HLSLResourcesDesc : public CommonShaderDescs::CommonResourceDesc
        {

            //
            HLSLResourcesDesc(const std::string &newResourceVariable) : CommonShaderDescs::CommonResourceDesc(newResourceVariable)
            {
            }

            //  Base Resource Type.
            HLSLBaseResourceType baseResourceType;

            //  Buffer Type.
            HLSLBufferType bufferType;

            //  Semantic Desc
            HLSLSemanticDesc semanticDesc;

            //  Struct Desc.
            HLSLStructDesc structDesc;
        };


        //  
        struct ConstantBufferData : public HLSLResourcesDesc
        {
            ConstantBufferData(const std::string &newResourceVariable) : HLSLResourcesDesc(newResourceVariable)
            {

            }

            //  Add a Variable to the Uniform Buffer.
            void addVariableToConstantBuffer(const HLSLDescs::HLSLResourcesDesc & rDesc);

            //  Set the Array Desc.
            void setArrayDesc(const std::vector<uint32_t> & dimensions, bool isArray = false, bool isUnboundedArray = false);

            //  Set the Register Index and Space.
            void setAttachmentDesc(const uint32_t newRegisterIndex, const uint32_t newRegisterSpace = 0, bool isExplicitRegisterIndex = false, bool isExplicitRegisterSpace = true);

            //  Get the Constant Buffer Name.
            std::string getCBVariable() const;

            //  Return whether or not this is a global Uniform/Constant Buffer Resource.
            bool getIsGlobalCBType() const;

            //  Return whether or not this is a Uniform/Constant Buffer Array Resource Desc.
            const CommonShaderDescs::ArrayDesc & viewArrayDesc() const;

            //  Return the associated Uniform/Constant Buffer Attachment Resource Desc.
            const CommonShaderDescs::ResourceAttachmentDesc & viewAttachmentDesc() const;

            //  Return whether or not this is a Uniform/Constant Buffer Struct Desc.
            const HLSLDescs::HLSLStructDesc & viewStructDesc() const;

            //  Whether or not we use the Global type for the Uniform/Constant Buffer.
            bool isGlobalUCBType = true;

        };
    };
};

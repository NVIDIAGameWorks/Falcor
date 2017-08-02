#pragma once
#include "CommonShaderDescs.h"

namespace Falcor
{
    namespace GLSLDescs
    {
        enum class GLSLBaseResourceType : uint32_t;
        enum class GLSLBufferType : uint32_t;
        struct GLSLResourcesDesc;

        //  GLSL Base Resource Type.
        enum class GLSLBaseResourceType : uint32_t
        {
            BasicType = 0u,
            TextureType = 1u,
            SamplerType = 2u,
            ImageType = 3u,
            BufferBackedType = 4u
        };

        //
        enum class GLSLBufferType : uint32_t
        {
            Uniform = 0u,
            SSBO = 1u
        };

        //  GLSL Layout Location.
        struct GLSLLocationDesc
        {
            uint32_t layoutLocation = 0;
        };

        //  Shader Struct Descs.
        struct GLSLStructDesc
        {
            //  Whether or not to define a new struct.
            bool definesStructType = true;

            //  Use the Struct Type.
            bool usesStructType = true;

            //  The Name of the Struct Variable Type.
            std::string structVariableType = "";

            //  The Array of Shader Resources.
            std::vector<GLSLResourcesDesc> structVariables = {};
        };


        //  
        struct GLSLResourcesDesc : public CommonShaderDescs::CommonResourceDesc
        {
            GLSLResourcesDesc(const std::string & newResourceVariable) : CommonShaderDescs::CommonResourceDesc(newResourceVariable)
            {

            }

            //  Base Resource Type.
            GLSLBaseResourceType baseResourceType;

            //  Buffer Type.
            GLSLBufferType bufferType;

            //  Location Desc.
            GLSLLocationDesc locationDesc;

            //  Struct Desc.
            GLSLStructDesc structDesc;

        };


        //  
        struct UniformBufferData : public GLSLDescs::GLSLResourcesDesc
        {
            UniformBufferData(const std::string &newResourceVariable) : GLSLResourcesDesc(newResourceVariable)
            {

            }

            //  Add a Variable to the Uniform Buffer.
            void addVariableToUniformBuffer(const GLSLDescs::GLSLResourcesDesc & rDesc);

            //  Set the Array Desc.
            void setArrayDesc(const std::vector<uint32_t> & dimensions, bool isArray = false, bool isUnboundedArray = false);

            //  Set the Register Index and Space.
            void setAttachmentDesc(const uint32_t newAttachmentSubpoint, const uint32_t newAttachmentPoint = 0, bool isExplicitRegisterIndex = false, bool isExplicitRegisterSpace = true);

            //  Get the Constant Buffer Name.
            std::string getUBVariable() const;

            //  Return whether or not this is a global Uniform/Constant Buffer Resource.
            bool getIsGlobalUBType() const;

            //  Return whether or not this is a Uniform/Constant Buffer Array Resource Desc.
            const CommonShaderDescs::ArrayDesc & viewArrayDesc() const;

            //  Return the associated Uniform/Constant Buffer Attachment Resource Desc.
            const CommonShaderDescs::ResourceAttachmentDesc & viewAttachmentDesc() const;

            //  Return whether or not this is a Uniform/Constant Buffer Struct Desc.
            const GLSLDescs::GLSLStructDesc & viewStructDesc() const;

            //  Whether or not we use the Global type for the Uniform/Constant Buffer.
            bool isGlobalUCBType = true;

        };
    };
};
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
#include "RenderGraph/RenderPassReflection.h"
#include "Core/API/Resource.h"

namespace Falcor
{
    class dlldecl ResourceCache : public std::enable_shared_from_this<ResourceCache>
    {
    public:
        using SharedPtr = std::shared_ptr<ResourceCache>;
        using ResourcesMap = std::unordered_map<std::string, Resource::SharedPtr>;

        /** Create a new object
        */
        static SharedPtr create();

        /** Properties to use during resource creation when its property has not been fully specified.
        */
        struct DefaultProperties
        {
            uint2 dims;                                         ///< Width, height of the swap chain
            ResourceFormat format = ResourceFormat::Unknown;    ///< Format to use for texture creation
        };

        /** Add/Remove reference to a graph input resource not owned by the cache
            \param[in] name The resource's name
            \param[in] pResource The resource to register. If this is null, will unregister the resource
        */
        void registerExternalResource(const std::string& name, const Resource::SharedPtr& pResource);

        /** Register a field that requires resources to be allocated.
            \param[in] name String in the format of PassName.FieldName
            \param[in] field Reflection data for the field
            \param[in] timePoint The point in time for when this field is used. Normally this is an index into the execution order.
            \param[in] alias Optional. Another field name described in the same way as 'name'.
                If specified, and the field exists in the cache, the resource will be aliased with 'name' and field properties will be merged.
        */
        void registerField(const std::string& name, const RenderPassReflection::Field& field, uint32_t timePoint, const std::string& alias = "");

        /** Get a resource by name. Includes external resources known by the cache.
        */
        const Resource::SharedPtr& getResource(const std::string& name) const;

        /** Get the field-reflection of a resource
        */
        const RenderPassReflection::Field& getResourceReflection(const std::string& name) const;

        /** Allocate all resources that need to be created/updated.
            This includes new resources, resources whose properties have been updated since last allocation call.
        */
        void allocateResources(const DefaultProperties& params);

        /** Clears all registered field/resource properties and allocated resources.
        */
        void reset();

    private:
        ResourceCache() = default;

        struct ResourceData
        {
            RenderPassReflection::Field field;      // Holds merged properties for aliased resources
            std::pair<uint32_t, uint32_t> lifetime; // Time range where this resource is being used
            Resource::SharedPtr pResource;          // The resource
            bool resolveBindFlags;                  // Whether or not we should resolve the field's bind-flags before creating the resource
            std::string name;                       // Full name of the resource, including the pass name
        };

        // Resources and properties for fields within (and therefore owned by) a render graph
        std::unordered_map<std::string, uint32_t> mNameToIndex;
        std::vector<ResourceData> mResourceData;

        // References to output resources not to be allocated by the render graph
        ResourcesMap mExternalResources;
    };

}

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
#include "Core/API/RtStateObject.h"
#include "D3D12NvApiExDesc.h"
#include "Core/API/Device.h"
#include "Core/API/D3D12/D3D12API.h"
#include "Utils/StringUtils.h"

#include <set>

namespace Falcor
{
    D3D12_RAYTRACING_PIPELINE_FLAGS getD3D12RtPipelineFlags(RtPipelineFlags flags)
    {
        D3D12_RAYTRACING_PIPELINE_FLAGS result = (D3D12_RAYTRACING_PIPELINE_FLAGS)0;
        if (is_set(flags, RtPipelineFlags::None)) result |= D3D12_RAYTRACING_PIPELINE_FLAG_NONE;
        if (is_set(flags, RtPipelineFlags::SkipTriangles)) result |= D3D12_RAYTRACING_PIPELINE_FLAG_SKIP_TRIANGLES;
        if (is_set(flags, RtPipelineFlags::SkipProceduralPrimitives)) result |= D3D12_RAYTRACING_PIPELINE_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
        return result;
    }

    class RtStateObjectHelper
    {
    public:
        ~RtStateObjectHelper()
        {
            clear();
        }

        void addPipelineConfig(uint32_t maxTraceRecursionDepth, RtPipelineFlags pipelineFlags)
        {
            addSubobject<PipelineConfig>(maxTraceRecursionDepth, pipelineFlags);
            mDirty = true;
        }

        void addProgramDesc(ID3DBlobPtr pBlob, const std::wstring& exportName)
        {
            addSubobject<ProgramDesc>(pBlob, exportName);
            mDirty = true;
        }

        void addHitGroupDesc(const std::wstring& ahsExportName, const std::wstring& chsExportName, const std::wstring& intersectionExportName, const std::wstring& name)
        {
            addSubobject<HitGroupDesc>(ahsExportName, chsExportName, intersectionExportName, name);
            mDirty = true;
        }

        void addGlobalRootSignature(ID3D12RootSignature* pRootSig)
        {
            addSubobject<GlobalRootSignature>(pRootSig);
            mDirty = true;
        }

        void addShaderConfig(const std::wstring exportNames[], uint32_t count, uint32_t maxPayloadSizeInBytes, uint32_t maxAttributeSizeInBytes)
        {
            std::unique_ptr<ShaderConfig> pConfig(new ShaderConfig(maxPayloadSizeInBytes, maxAttributeSizeInBytes));
            addSubobject<ExportAssociation>(exportNames, count, std::move(pConfig));
            mDirty = true;
        }

        D3D12_STATE_OBJECT_DESC getDesc()
        {
            finalize();
            D3D12_STATE_OBJECT_DESC desc;
            desc.NumSubobjects = (uint32_t)mSubobjects.size();
            desc.pSubobjects = mSubobjects.data();
            desc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
            return desc;
        }

        void clear()
        {
            mSubobjects.clear();
            mBaseSubObjects.clear();
            mDirty = true;
        }

    private:
        using SubobjectVector = std::vector<D3D12_STATE_SUBOBJECT>;

        struct RtStateSubobjectBase
        {
            virtual ~RtStateSubobjectBase() = default;
            D3D12_STATE_SUBOBJECT subobject = {};

            virtual void addToVector(SubobjectVector& vec)
            {
                vec.push_back(subobject);
            }
        };

        struct PipelineConfig : public RtStateSubobjectBase
        {
            PipelineConfig(uint32_t maxTraceRecursionDepth, RtPipelineFlags flags)
            {
                config.MaxTraceRecursionDepth = maxTraceRecursionDepth;
                config.Flags = getD3D12RtPipelineFlags(flags);

                subobject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG1;
                subobject.pDesc = &config;
            }
            virtual ~PipelineConfig() = default;
            D3D12_RAYTRACING_PIPELINE_CONFIG1 config = {};
        };

        struct ProgramDesc : public RtStateSubobjectBase
        {
            ProgramDesc(ID3DBlobPtr pBlob, const std::wstring& exportName_) : pShaderBlob(pBlob), exportName(exportName_)
            {
                subobject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
                subobject.pDesc = &dxilLibDesc;

                dxilLibDesc = {};
                if (pBlob)
                {
                    dxilLibDesc.DXILLibrary.pShaderBytecode = pBlob->GetBufferPointer();
                    dxilLibDesc.DXILLibrary.BytecodeLength = pBlob->GetBufferSize();

                    exportDesc.Name = exportName.c_str();
                    exportDesc.Flags = D3D12_EXPORT_FLAG_NONE;
                    exportDesc.ExportToRename = nullptr;

                    dxilLibDesc.NumExports = 1;
                    dxilLibDesc.pExports = &exportDesc;
                }
            };

            virtual ~ProgramDesc() = default;

            D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {};
            ID3DBlobPtr pShaderBlob;
            D3D12_EXPORT_DESC exportDesc;
            std::wstring exportName;
        };

        struct HitGroupDesc : public RtStateSubobjectBase
        {
            HitGroupDesc(
                const std::wstring& ahsExportName,
                const std::wstring& chsExportName,
                const std::wstring& intersectionExportName,
                const std::wstring& name)
                : exportName(name)
                , ahsName(ahsExportName)
                , chsName(chsExportName)
                , intersectionName(intersectionExportName)
            {
                desc.Type = intersectionName.empty() ? D3D12_HIT_GROUP_TYPE_TRIANGLES : D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE;
                desc.IntersectionShaderImport = intersectionName.empty() ? nullptr : intersectionName.c_str();
                desc.AnyHitShaderImport = ahsName.empty() ? nullptr : ahsName.c_str();
                desc.ClosestHitShaderImport = chsName.empty() ? nullptr : chsName.c_str();
                desc.HitGroupExport = exportName.c_str();

                subobject.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
                subobject.pDesc = &desc;
            }

            virtual ~HitGroupDesc() = default;
            std::wstring exportName;
            std::wstring ahsName;
            std::wstring chsName;
            std::wstring intersectionName;

            D3D12_HIT_GROUP_DESC desc = {};

            virtual void addToVector(SubobjectVector& vec) override
            {
                vec.push_back(subobject);
            }
        };

        struct ExportAssociation : public RtStateSubobjectBase
        {
            ExportAssociation(const std::wstring names[], uint32_t count, std::unique_ptr<RtStateSubobjectBase> pSubobjectToAssociate) : exportNames(count)
            {
                association.NumExports = count;
                pName.resize(exportNames.size());
                for (size_t i = 0; i < exportNames.size(); i++)
                {
                    exportNames[i] = names[i];
                    pName[i] = exportNames[i].c_str();
                }
                association.pExports = pName.data();

                subobject.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
                subobject.pDesc = &association;
                pAssociatedSubobject = std::move(pSubobjectToAssociate);
            }

            virtual ~ExportAssociation() = default;

            std::vector<std::wstring> exportNames;
            std::vector<const WCHAR*> pName;
            D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION association = {};
            std::unique_ptr<RtStateSubobjectBase> pAssociatedSubobject;

            virtual void addToVector(SubobjectVector& vec) override
            {
                // TODO: finalize() assumes that the subobject to associate comes right before the export associataion subobject. Need to figure out a way to remove this assumption
                FALCOR_ASSERT(pAssociatedSubobject);
                pAssociatedSubobject->addToVector(vec);
                vec.push_back(subobject);
            }
        };

        struct GlobalRootSignature : public RtStateSubobjectBase
        {
            GlobalRootSignature(ID3D12RootSignature* pRootSig)
            {
                pSignature = pRootSig;
                subobject.pDesc = &pSignature;
                subobject.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
            }
            ID3D12RootSignature* pSignature;
        };

        struct ShaderConfig : public RtStateSubobjectBase
        {
            ShaderConfig(uint32_t maxPayloadSizeInBytes, uint32_t maxAttributeSizeInBytes)
            {
                shaderConfig.MaxAttributeSizeInBytes = maxAttributeSizeInBytes;
                shaderConfig.MaxPayloadSizeInBytes = maxPayloadSizeInBytes;

                subobject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
                subobject.pDesc = &shaderConfig;
            }
            virtual ~ShaderConfig() = default;
            D3D12_RAYTRACING_SHADER_CONFIG shaderConfig = {};
        };

        bool mDirty = false;
        SubobjectVector mSubobjects;
        std::list<std::unique_ptr<RtStateSubobjectBase>> mBaseSubObjects;

        template<typename T, typename... Args>
        RtStateSubobjectBase* addSubobject(Args&&... args)
        {
            T* pSubobject = new T(std::forward<Args>(args)...);
            mBaseSubObjects.emplace_back(std::unique_ptr<RtStateSubobjectBase>(pSubobject));
            return pSubobject;
        }

        void finalize()
        {
            if (mDirty == false) return;
            mSubobjects.clear();
            for (const auto& l : mBaseSubObjects)
            {
                l->addToVector(mSubobjects);
            }

            // For every export association, we need to correct the address of the associated object
            for (size_t i = 0; i < mSubobjects.size(); i++)
            {
                if (mSubobjects[i].Type == D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION)
                {
                    // The associated object is the one before the association itself. See `ExportAssociation`
                    D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION* pAssociation = (D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION*)mSubobjects[i].pDesc;
                    pAssociation->pSubobjectToAssociate = &mSubobjects[i - 1];
                }
            }
            mDirty = false;
        }
    };


    void RtStateObject::apiInit()
    {
        RtStateObjectHelper rtsoHelper;
        std::set<std::wstring> configuredShaders;

        auto configureShader = [&](ID3DBlobPtr pBlob, const std::wstring& shaderName, RtEntryPointGroupKernels* pEntryPointGroup)
        {
            if (pBlob && !shaderName.empty() && !configuredShaders.count(shaderName))
            {
                rtsoHelper.addProgramDesc(pBlob, shaderName);
                rtsoHelper.addShaderConfig(&shaderName, 1, pEntryPointGroup->getMaxPayloadSize(), pEntryPointGroup->getMaxAttributesSize());
                configuredShaders.insert(shaderName);
            }
        };

        // Pipeline config
        rtsoHelper.addPipelineConfig(mDesc.mMaxTraceRecursionDepth, mDesc.mPipelineFlags);

        auto pKernels = getKernels();

#if FALCOR_HAS_NVAPI
        // Enable NVAPI extension if required
        auto nvapiRegisterIndex = findNvApiShaderRegister(pKernels);
        if (nvapiRegisterIndex)
        {
            if (NvAPI_Initialize() != NVAPI_OK) throw RuntimeError("Failed to initialize NvApi");
            if (NvAPI_D3D12_SetNvShaderExtnSlotSpace(gpDevice->getApiHandle(), *nvapiRegisterIndex, 0) != NVAPI_OK) throw RuntimeError("Failed to set NvApi extension");
        }
#endif

        // Loop over the programs
        for (const auto& pBaseEntryPointGroup : pKernels->getUniqueEntryPointGroups())
        {
            FALCOR_ASSERT(dynamic_cast<RtEntryPointGroupKernels*>(pBaseEntryPointGroup.get()));
            auto pEntryPointGroup = static_cast<RtEntryPointGroupKernels*>(pBaseEntryPointGroup.get());
            switch (pBaseEntryPointGroup->getType())
            {
            case EntryPointGroupKernels::Type::RtHitGroup:
                {
                    const Shader* pIntersection = pEntryPointGroup->getShader(ShaderType::Intersection);
                    const Shader* pAhs = pEntryPointGroup->getShader(ShaderType::AnyHit);
                    const Shader* pChs = pEntryPointGroup->getShader(ShaderType::ClosestHit);

                    ID3DBlobPtr pIntersectionBlob = pIntersection ? pIntersection->getD3DBlob() : nullptr;
                    ID3DBlobPtr pAhsBlob = pAhs ? pAhs->getD3DBlob() : nullptr;
                    ID3DBlobPtr pChsBlob = pChs ? pChs->getD3DBlob() : nullptr;

                    const std::wstring& exportName = string_2_wstring(pEntryPointGroup->getExportName());
                    const std::wstring& intersectionExport = pIntersection ? string_2_wstring(pIntersection->getEntryPoint()) : L"";
                    const std::wstring& ahsExport = pAhs ? string_2_wstring(pAhs->getEntryPoint()) : L"";
                    const std::wstring& chsExport = pChs ? string_2_wstring(pChs->getEntryPoint()) : L"";

                    configureShader(pIntersectionBlob, intersectionExport, pEntryPointGroup);
                    configureShader(pAhsBlob, ahsExport, pEntryPointGroup);
                    configureShader(pChsBlob, chsExport, pEntryPointGroup);

                    rtsoHelper.addHitGroupDesc(ahsExport, chsExport, intersectionExport, exportName);
                }
                break;

            default:
                {
                    const std::wstring& exportName = string_2_wstring(pEntryPointGroup->getExportName());

                    const Shader* pShader = pEntryPointGroup->getShaderByIndex(0);
                    rtsoHelper.addProgramDesc(pShader->getD3DBlob(), exportName);

                    // Payload size
                    rtsoHelper.addShaderConfig(&exportName, 1, pEntryPointGroup->getMaxPayloadSize(), pEntryPointGroup->getMaxAttributesSize());
                }
                break;
            }
        }

        // Add an empty global root-signature
        D3D12RootSignature* pRootSig = mDesc.mpKernels->getD3D12RootSignature() ? mDesc.mpKernels->getD3D12RootSignature().get() : D3D12RootSignature::getEmpty().get();
        rtsoHelper.addGlobalRootSignature(pRootSig->getApiHandle());

        // Create the state
        D3D12_STATE_OBJECT_DESC objectDesc = rtsoHelper.getDesc();
        FALCOR_GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
        FALCOR_D3D_CALL(pDevice5->CreateStateObject(&objectDesc, IID_PPV_ARGS(&mApiHandle)));

        FALCOR_MAKE_SMART_COM_PTR(ID3D12StateObjectProperties);
        ID3D12StateObjectPropertiesPtr pRtsoProps = getApiHandle();

        for (const auto& pBaseEntryPointGroup : pKernels->getUniqueEntryPointGroups())
        {
            FALCOR_ASSERT(dynamic_cast<RtEntryPointGroupKernels*>(pBaseEntryPointGroup.get()));
            auto pEntryPointGroup = static_cast<RtEntryPointGroupKernels*>(pBaseEntryPointGroup.get());
            const std::wstring& exportName = string_2_wstring(pEntryPointGroup->getExportName());

            void const* pShaderIdentifier = pRtsoProps->GetShaderIdentifier(exportName.c_str());
            mShaderIdentifiers.push_back(pShaderIdentifier);
        }

#if FALCOR_HAS_NVAPI
        if (nvapiRegisterIndex)
        {
            if (NvAPI_D3D12_SetNvShaderExtnSlotSpace(gpDevice->getApiHandle(), 0xFFFFFFFF, 0) != NVAPI_OK) throw RuntimeError("Failed to unset NvApi extension");
        }
#endif
    }

}

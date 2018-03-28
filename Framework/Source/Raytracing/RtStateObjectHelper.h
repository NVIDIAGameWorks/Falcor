/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
#pragma once
#include <list>

namespace Falcor
{
    class RtStateObjectHelper
    {
    public:
        ~RtStateObjectHelper()
        {
            clear();
        }

        void addPipelineConfig(uint32_t maxTraceRecursionDepth)
        {
            addSubobject<PipelineConfig>(maxTraceRecursionDepth);
            mDirty = true;
        }

        void addProgramDesc(ID3DBlobPtr pBlob, const std::wstring& exportName)
        {
            addSubobject<ProgramDesc>(pBlob, exportName);
            mDirty = true;
        }

        void addHitProgramDesc(ID3DBlobPtr pAhsBlob, const std::wstring& ahsExportName, ID3DBlobPtr pChsBlob, const std::wstring& chsExportName, ID3DBlobPtr pIntersectionBlob, const std::wstring& intersectionExportName, const std::wstring& name)
        {
            addSubobject<HitProgramDesc>(pAhsBlob, ahsExportName, pChsBlob, chsExportName, pIntersectionBlob, intersectionExportName, name);
            mDirty = true;
        }

        void addLocalRootSignature(const std::wstring exportNames[], uint32_t count, ID3D12RootSignature* pRootSig)
        {
            LocalRootSignature* pLocalRoot = new LocalRootSignature(pRootSig);
            addSubobject<ExportAssociation>(exportNames, count, pLocalRoot);
            mDirty = true;
        }

        void addGlobalRootSignature(ID3D12RootSignature* pRootSig)
        {
            addSubobject<GlobalRootSignature>(pRootSig);
            mDirty = true;
        }

        void addShaderConfig(const std::wstring exportNames[], uint32_t count, uint32_t maxPayloadSizeInBytes, uint32_t maxAttributeSizeInBytes)
        {
            ShaderConfig* pConfig = new ShaderConfig(maxPayloadSizeInBytes, maxAttributeSizeInBytes);
            addSubobject<ExportAssociation>(exportNames, count, pConfig);
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
            for (auto& p : mBaseSubObjects)
            {
                safe_delete(p);
            }
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
            PipelineConfig(uint32_t maxTraceRecursionDepth)
            {
                config.MaxTraceRecursionDepth = maxTraceRecursionDepth;

                subobject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
                subobject.pDesc = &config;
            }
            virtual ~PipelineConfig() = default;
            D3D12_RAYTRACING_PIPELINE_CONFIG config = {};            
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

        struct HitProgramDesc : public RtStateSubobjectBase
        {
            HitProgramDesc(
                ID3DBlobPtr pAhsBlob, const std::wstring& ahsExportName,
                ID3DBlobPtr pChsBlob, const std::wstring& chsExportName,
                ID3DBlobPtr pIntersectionBlob, const std::wstring& intersectionExportName,
                const std::wstring& name) :
                anyHitShader(pAhsBlob, ahsExportName),
                closestHitShader(pChsBlob, chsExportName),
                intersectionShader(pIntersectionBlob, intersectionExportName),
                exportName(name)
            {
                desc.IntersectionShaderImport = pIntersectionBlob ? intersectionShader.exportName.c_str() : nullptr;
                desc.AnyHitShaderImport = pAhsBlob ? anyHitShader.exportName.c_str() : nullptr;
                desc.ClosestHitShaderImport = pChsBlob ? closestHitShader.exportName.c_str() : nullptr;
                desc.HitGroupExport = exportName.c_str();

                subobject.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
                subobject.pDesc = &desc;
            }

            virtual ~HitProgramDesc() = default;
            std::wstring exportName;
            ProgramDesc anyHitShader;
            ProgramDesc closestHitShader;
            ProgramDesc intersectionShader;

            D3D12_HIT_GROUP_DESC desc = {};

            virtual void addToVector(SubobjectVector& vec) override
            {
                if (desc.AnyHitShaderImport)        anyHitShader.addToVector(vec);
                if (desc.ClosestHitShaderImport)    closestHitShader.addToVector(vec);
                if (desc.IntersectionShaderImport)   intersectionShader.addToVector(vec);
                vec.push_back(subobject);
            }
        };

        struct ExportAssociation : public RtStateSubobjectBase
        {
            ExportAssociation(const std::wstring names[], uint32_t count, RtStateSubobjectBase* pSubobjectToAssociate) : exportNames(count)
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
                pAssociatedSubobject = pSubobjectToAssociate;
            }

            virtual ~ExportAssociation()
            {
                safe_delete(pAssociatedSubobject);
            }

            std::vector<std::wstring> exportNames;
            std::vector<const WCHAR*> pName;
            D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION association = {};
            RtStateSubobjectBase* pAssociatedSubobject = nullptr;

            virtual void addToVector(SubobjectVector& vec) override
            {
                // TODO: finalize() assumes that the subobject to associate comes right before the export associataion subobject. Need to figure out a way to remove this assumption
                assert(pAssociatedSubobject);
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
                subobject.Type = D3D12_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
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

        struct LocalRootSignature : public RtStateSubobjectBase
        {
            LocalRootSignature(ID3D12RootSignature* pRootSig)
            {
                pSignature = pRootSig;
                subobject.pDesc = &pSignature;
                subobject.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
            }
            virtual ~LocalRootSignature() = default;
            ID3D12RootSignature* pSignature;
        };

        bool mDirty = false;
        SubobjectVector mSubobjects;
        std::list<RtStateSubobjectBase*> mBaseSubObjects;

        template<typename T, typename... Args>
        RtStateSubobjectBase* addSubobject(Args... args)
        {
            T* pSubobject = new T(args...);
            mBaseSubObjects.emplace_back(pSubobject);
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
}

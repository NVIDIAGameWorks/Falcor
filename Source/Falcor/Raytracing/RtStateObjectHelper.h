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
#include <list>

namespace Falcor
{
    class dlldecl RtStateObjectHelper
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

        void addHitGroupDesc(const std::wstring& ahsExportName, const std::wstring& chsExportName, const std::wstring& intersectionExportName, const std::wstring& name)
        {
            addSubobject<HitGroupDesc>(ahsExportName, chsExportName, intersectionExportName, name);
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

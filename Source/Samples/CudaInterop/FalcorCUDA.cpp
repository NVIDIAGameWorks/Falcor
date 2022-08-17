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
#include "FalcorCUDA.h"
#include <cuda.h>
#include <AccCtrl.h>
#include <aclapi.h>
#include "Core/API/Device.h"

#define CU_CHECK_SUCCESS(x)                                                             \
    do {                                                                                \
        CUresult result = x;                                                            \
        if (result != CUDA_SUCCESS)                                                     \
        {                                                                               \
            const char* msg;                                                            \
            cuGetErrorName(result, &msg);                                               \
            reportError("CUDA Error: " #x " failed with error " + std::string(msg));    \
            return 0;                                                                   \
        }                                                                               \
    } while(0)

#define CUDA_CHECK_SUCCESS(x)                                                                               \
    do {                                                                                                    \
        cudaError_t result = x;                                                                             \
        if (result != cudaSuccess)                                                                          \
        {                                                                                                   \
            reportError("CUDA Error: " #x " failed with error " + std::string(cudaGetErrorString(result))); \
            return 0;                                                                                       \
        }                                                                                                   \
    } while(0)

using namespace Falcor;

namespace
{
    class WindowsSecurityAttributes
    {
    protected:
        SECURITY_ATTRIBUTES mWinSecurityAttributes;
        PSECURITY_DESCRIPTOR mWinPSecurityDescriptor;

    public:
        WindowsSecurityAttributes()
        {
            mWinPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));
            FALCOR_ASSERT(mWinPSecurityDescriptor != (PSECURITY_DESCRIPTOR)NULL);

            PSID* ppSID = (PSID*)((PBYTE)mWinPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
            PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

            InitializeSecurityDescriptor(mWinPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);

            SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
            AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

            EXPLICIT_ACCESS explicitAccess;
            ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
            explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
            explicitAccess.grfAccessMode = SET_ACCESS;
            explicitAccess.grfInheritance = INHERIT_ONLY;
            explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
            explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
            explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

            SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

            SetSecurityDescriptorDacl(mWinPSecurityDescriptor, TRUE, *ppACL, FALSE);

            mWinSecurityAttributes.nLength = sizeof(mWinSecurityAttributes);
            mWinSecurityAttributes.lpSecurityDescriptor = mWinPSecurityDescriptor;
            mWinSecurityAttributes.bInheritHandle = TRUE;
        }

        ~WindowsSecurityAttributes()
        {
            PSID* ppSID = (PSID*)((PBYTE)mWinPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
            PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

            if (*ppSID) FreeSid(*ppSID);
            if (*ppACL) LocalFree(*ppACL);
            free(mWinPSecurityDescriptor);
        }
        SECURITY_ATTRIBUTES* operator&() { return &mWinSecurityAttributes; }
    };

    uint32_t gNodeMask;
    CUdevice  gCudaDevice;
    CUcontext gCudaContext;
    CUstream  gCudaStream;

}

namespace FalcorCUDA
{
    bool initCUDA()
    {
        CU_CHECK_SUCCESS(cuInit(0));
        int32_t firstGPUID = -1;
        cudaDeviceProp prop;
        int32_t count;
        cudaError_t err = cudaGetDeviceCount(&count);

        for (int32_t i = 0; i < count; ++i)
        {
            err = cudaGetDeviceProperties(&prop, i);
            if (prop.major >= 3)
            {
                firstGPUID = i;
                break;
            }
        }

        if (firstGPUID < 0)
        {
            reportError("No CUDA 10 compatible GPU found");
            return false;
        }
        gNodeMask = prop.luidDeviceNodeMask;
        CUDA_CHECK_SUCCESS(cudaSetDevice(firstGPUID));
        CU_CHECK_SUCCESS(cuDeviceGet(&gCudaDevice, firstGPUID));
        CU_CHECK_SUCCESS(cuCtxCreate(&gCudaContext, 0, gCudaDevice));
        CU_CHECK_SUCCESS(cuStreamCreate(&gCudaStream, CU_STREAM_DEFAULT));
        return true;
    }

    bool importTextureToMipmappedArray(Falcor::Texture::SharedPtr pTex, cudaMipmappedArray_t& mipmappedArray, uint32_t cudaUsageFlags)
    {
        HANDLE sharedHandle = pTex->getSharedApiHandle();
        if (sharedHandle == NULL)
        {
            reportError("FalcorCUDA::importTextureToMipmappedArray - texture shared handle creation failed");
            return false;
        }

        cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
        memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

        externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
        externalMemoryHandleDesc.size = pTex->getTextureSizeInBytes();
        externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

        cudaExternalMemory_t externalMemory;
        CUDA_CHECK_SUCCESS(cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));

        // Map mipmapped array onto external memory
        cudaExternalMemoryMipmappedArrayDesc mipDesc;
        memset(&mipDesc, 0, sizeof(mipDesc));
        auto format = pTex->getFormat();
        mipDesc.formatDesc.x = getNumChannelBits(format, 0);
        mipDesc.formatDesc.y = getNumChannelBits(format, 1);
        mipDesc.formatDesc.z = getNumChannelBits(format, 2);
        mipDesc.formatDesc.w = getNumChannelBits(format, 3);
        mipDesc.formatDesc.f = (getFormatType(format) == FormatType::Float) ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
        mipDesc.extent.depth = 1;
        mipDesc.extent.width = pTex->getWidth();
        mipDesc.extent.height = pTex->getHeight();
        mipDesc.flags = cudaUsageFlags;
        mipDesc.numLevels = 1;

        CUDA_CHECK_SUCCESS(cudaExternalMemoryGetMappedMipmappedArray(&mipmappedArray, externalMemory, &mipDesc));
        return true;
    }

    cudaSurfaceObject_t mapTextureToSurface(Texture::SharedPtr pTex, uint32_t cudaUsageFlags)
    {
        // Create a mipmapped array from the texture
        cudaMipmappedArray_t mipmap;
        if (!importTextureToMipmappedArray(pTex, mipmap, cudaUsageFlags))
        {
            reportError("Failed to import texture into a mipmapped array");
            return 0;
        }

        // Grab level 0
        cudaArray_t cudaArray;
        CUDA_CHECK_SUCCESS(cudaGetMipmappedArrayLevel(&cudaArray, mipmap, 0));

        // Create cudaSurfObject_t from CUDA array
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.res.array.array = cudaArray;
        resDesc.resType = cudaResourceTypeArray;

        cudaSurfaceObject_t surface;
        CUDA_CHECK_SUCCESS(cudaCreateSurfaceObject(&surface, &resDesc));
        return surface;
    }
}

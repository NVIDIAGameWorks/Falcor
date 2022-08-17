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
#if FALCOR_HAS_CUDA
#include "CUDAProgram.h"
#include "ProgramVersion.h"
#include "ProgramVars.h"

// This file implements the CUDA-specific logic that allows the `Program`,
// `ProgramVars`, etc. types to support execution via CUDA.
//
// The approach taken here relies on details of how Slang compiles compute
// shader entry points for execution on CUDA, and we will call out those
// details where we make use of them. If Slang makes breaking changes to
// its CUDA code generation approach, this file will need to be updated
// accordingly.

// We avoid including CUDA headers in other files because of conflicts
// they can apparently create with the vector types that Falcor uses.
//
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace Falcor
{
    // For those unfamiliar with the way that the Falcor `Program` hierarchy
    // is implemented behind the scenes:
    //
    // * A `Program` (subclasses include `ComputeProgram`, `GraphicsProgram`, etc.)
    //   represents a collection of Slang source files/strings containing zero
    //   or more entry points. It presents a mutable set of `#define`s and allows
    //   creation of `ProgramVersion`s based on the current `#define`s.
    //
    // * A `ProgramVersion` represents a program that has been parsed/checked by
    //   Slang, using a particular set of `#define`s. It allows reflection to be
    //   performed, which then allows `ProgramVars` to be allocated.
    //
    // * A `ProgramVars` represents a mapping of the parameters of a `ProgramVersion`
    //   to the value to use for each parameter. Some parameters might have
    //   interface types, or otherwise interact with Slang langauge features for
    //   specialization.
    //
    // * A `ProgramKernels` represents a compiled "variant" of a `ProgramVersion`,
    //   based on taking the program version and specializing it to zero or more
    //   types that were bound to its parameters in a particular `ProgramVars`.
    //
    // Our goal in integrating CUDA support is to leverage as much of the existing
    // work done on these types as possible. In practice, we only introduce two
    // CUDA-specific types: `CUDAProgram` and `CUDAProgramKernels`.
    //
    // The `CUDAProgram` type was declared in `CUDAProgram.h`, and is responsible
    // for overriding a few key `virtual` methods from `Program` so that it can
    // facilitate CUDA-specific behavior.
    //
    // The creation logic for `CUDAProgram` is straightforward, and nearly identical
    // to that for `ComputeProgram`.

    CUDAProgram::SharedPtr CUDAProgram::createFromFile(
        const std::filesystem::path& path,
        const std::string& csEntry,
        const DefineList& programDefines,
        Shader::CompilerFlags flags,
        const std::string& shaderModel)
    {
        Desc d(path);
        if (!shaderModel.empty()) d.setShaderModel(shaderModel);
        d.setCompilerFlags(flags);
        d.csEntry(csEntry);
        return create(d, programDefines);
    }

    CUDAProgram::SharedPtr CUDAProgram::create(
        const Desc& desc,
        const DefineList& programDefines)
    {
        auto pProg = SharedPtr(new CUDAProgram(desc, programDefines));
        registerProgramForReload(pProg);
        return pProg;
    }

    CUDAProgram::CUDAProgram(const Desc& desc, const DefineList& programDefines)
        : ComputeProgram(desc, programDefines)
    {
    }

    // The first place where an interesting difference arises is when
    // it is time to invoke the Slang compiler front-end to parse and
    // check the code of a program.
    //
    void CUDAProgram::setUpSlangCompilationTarget(
        slang::TargetDesc& ioTargetDesc,
        char const*& ioTargetMacroName) const
    {
        // When compiling Slang source code for execution via CUDA, we need to
        // customize the compilation environment in two ways that differ from
        // the existing D3D and VK paths.
        //
        // First, we need to make sure to generate PTX code instead of DXBC,
        // DXIL, SPIR-V, etc.
        //
        ioTargetDesc.format = SLANG_PTX;

        // Second, we set a global define of `FALCOR_CUDA` to allow source code
        // to customize its behavior based on knowledge of the target.
        //
        // Note: Shader code should try to avoid using this macro if at all possible,
        // so that as much code as possible can remain portable.
        //
        ioTargetMacroName = "FALCOR_CUDA";
    }

    // The next customization point comes when it is time to load compiled
    // code into a `ProgramKernels`, because the `CUDAProgram` needs to
    // ensure that a `CUDAProgramKernels` gets created instead.
    //
    // The `CUDAProgramKernels` type stores and interacts with CUDA-specific
    // types, so it is necessary to declare it here in a source file instead
    // of a header.
    //
    class FALCOR_API CUDAProgramKernels : public ProgramKernels
    {
    public:
        using SharedPtr = std::shared_ptr<CUDAProgramKernels>;
        using SharedConstPtr = std::shared_ptr<const CUDAProgramKernels>;

        /** Create a CUDA program kernels.
            \param[in] pVersion The program version the kernels represent.
            \param[in] pReflector The reflection information for the compiled/specialized kernels.
            \param[in] uniqueEntryPointGroups The (deduplicated) entry-point groups that the program includes
            \param[in,out] log Log of error messages for compilation/linking failures
            \param[in] name Name to use for debugging purposes.
            \return New object, or throws an exception if creation failed.
        */
        static SharedPtr create(
            const ProgramVersion* pVersion,
            slang::IComponentType* pSpecializedSlangGlobalScope,
            const std::vector<slang::IComponentType*>& pTypeConformanceSpecializedEntryPoints,
            const ProgramReflection::SharedPtr& pReflector,
            const UniqueEntryPointGroups& uniqueEntryPointGroups,
            std::string& log,
            const std::string& name = "")
        {
            return SharedPtr(new CUDAProgramKernels(pVersion, pReflector, uniqueEntryPointGroups, name));
        }

        virtual ~CUDAProgramKernels();

        /** Dispatch a CUDA grid using the compiled kernels.
            \param[in] pVars Parameter data to use (must have been created from the same base `Program`
            \param[in] gridShape The shape of the dispatch grid, in units of thread blocks.
        */
        void dispatchCompute(ComputeVars* pVars, uint3 const& gridShape) const;

    protected:
        CUDAProgramKernels(
            const ProgramVersion* pVersion,
            const ProgramReflection::SharedPtr& pReflector,
            const UniqueEntryPointGroups& uniqueEntryPointGroups,
            const std::string& name = "")
            : ProgramKernels(pVersion, pReflector, uniqueEntryPointGroups, name)
        {
            init();
        }

        void init();

        /// The CUDA module that contains the compiled code of the entry-point kernel(s).
        CUmodule mCudaModule = 0;

        // We currently only support CUDA for compute programs, which have
        // a single entry point kernel function. If/when we generalize to support
        // OptiX for ray-tracing programs, we would need to have a variant of
        // `CUDAProgramKernels` that instead stores a full OptiX PSO including
        // all the relevant kernels.

        /// The CUDA function for the entry point kernel.
        CUfunction mCudaEntryPoint = 0;

        /// The device address of the `SLANG_globalParams` variable (if any) for the CUDA module.
        CUdeviceptr mpGlobalParamsSymbol = 0;
    };

    // With the declaration of `CUDAProgramKernels` out of the way, the logic
    // required for a `CUDAProgram` to actually load one is unremarkable.
    //
    // Note that `createProgramKernels` is an internal routine used by `Program`
    // to allow its subclasses to customize the way that compiled kernels are
    // loaded and represented. It is not part of the user-facing API of `Program`.
    //
    ProgramKernels::SharedPtr CUDAProgram::createProgramKernels(
        const ProgramVersion* pVersion,
        slang::IComponentType* pSpecializedSlangGlobalScope,
        const std::vector<slang::IComponentType*>& pTypeConformanceSpecializedEntryPoints,
        const ProgramReflection::SharedPtr& pReflector,
        const ProgramKernels::UniqueEntryPointGroups& uniqueEntryPointGroups,
        std::string& log,
        const std::string& name) const
    {
        return CUDAProgramKernels::create(
            pVersion,
            pSpecializedSlangGlobalScope,
            pTypeConformanceSpecializedEntryPoints,
            pReflector,
            uniqueEntryPointGroups,
            log,
            name);
    }

    // The more interesting work occurs inside of `CUDAProgramKernels::init`,
    // but before we get to that we will define a few utilities for interacting
    // with the CUDA API.
    //
    // We want to detect errors when interacting with the CUDA and
    // turn them into exceptions to match the default error-handling
    // policy used in the rest of the code.
    //
    // CUDA functions can return either a `CUresult` or a `cudaError_t`,
    // so we add helper functions for each of these.
    //
    // TODO: OptiX uses yet another type for errors, so a third overload
    // will be needed if/when OptiX support is added.

    static void checkCUDAResult(CUresult result, char const* what)
    {
        if (result != 0)
        {
            throw RuntimeError(what);
        }
    }

    static void checkCUDAResult(cudaError_t result, char const* what)
    {
        if (result != 0)
        {
            throw RuntimeError(what);
        }
    }

    // To make it easier to use the error-checking facilities, we
    // add a helper macro that invokes a CUDA function and immediately
    // passes it to `checkCUDAResult` for translation to an exception
    // (if needed).
    //
#define FALCOR_CUDA_THROW_ON_FAIL(_expr) \
    checkCUDAResult(_expr, "CUDA failure in '" #_expr "'")

    // With the utilities in place, we can now work through the steps
    // of loading Slang-compiled PTX code via the CUDA API.

    void CUDAProgramKernels::init()
    {
        // First, we ensure that the CUDA API has been initialized.
        //
        FALCOR_CUDA_THROW_ON_FAIL(cuInit(0));

        // Next we get the device to use for CUDA operations.
        //
        // Note: It might eventually be important to allow a `CUDAProgram`
        // and/or its kernels to be loaded on a specific device under
        // application control. Such a feature would need to be part of
        // an overhauled approach to exposing CUDA as a "first-class"
        // API in Falcor.
        //
        CUdevice cudaDevice;
        FALCOR_CUDA_THROW_ON_FAIL(cuDeviceGet(&cudaDevice, 0));

        // We also create a CUDA context to manage CUDA-related
        // state on the current thread.
        //
        // TODO: This need to be something the application can
        // own/control.
        //
        unsigned int flags = 0;
        CUcontext cudaContext;
        FALCOR_CUDA_THROW_ON_FAIL(cuCtxCreate(&cudaContext, flags, cudaDevice));

        // For now expect that every `CUDAProgram` represents a compute
        // program with a single entry point.
        //
        auto pComputeShader = getShader(ShaderType::Compute);
        if(!pComputeShader)
        {
            throw RuntimeError("Expected CUDA program to have a single compute entry point");
        }

        // The binary "blob" stored on the compute entry point is the PTX
        // code for the kernel, and we load it into the CUDA API as a module.
        //
        FALCOR_CUDA_THROW_ON_FAIL(cuModuleLoadData(&mCudaModule, pComputeShader->getBlobData().data));

        // The CUDA module could in principle contain zero or more entry points
        // or other global symbols, so that we need to query for the global
        // function symbol that represents our desired entry point.
        //
        auto entryPointName = pComputeShader->getEntryPoint();
        FALCOR_CUDA_THROW_ON_FAIL(cuModuleGetFunction(&mCudaEntryPoint, mCudaModule, entryPointName.c_str()));

        // TODO: Eventually it would be better to support having a single CUDA module that
        // might contain multiple entry points (which could even be composed into
        // multiple distinct programs, in the OptiX case).
        //
        // The big catch here is that much of the Falcor code has been written with the assumption
        // that the binary code is always associated with the individual kernels/entry-points and
        // not with a program/module.
        //
        // The second catch is that the Falcor API does not make a distinction between a `Program`
        // as a unit of shader code loading vs. a `Program` as a unit of assembling shader code
        // to make an executable entity (like a PSO).

        // Shader parameters in the input Slang code could appear as either entry-point `uniform` parameters
        // or global parameters (where the latter is what most existing HLSL code defaults to).
        //
        // Shader parameters get translated from Slang to CUDA in a way that depends on how they
        // were declared:
        //
        // * Entry-point `uniform` parameters in Slang translate directly to entry-point parameters
        //   in the generated CUDA code (in order to match the programmer's intuition).
        //
        // * Global-scope parameters in Slang get aggregated into a `struct` type and then are
        //   used to declare a single global `__constant__` parameter, named `SLANG_globalParams`.
        //
        // In order to handle any global-scope parameters, we will query the module for a symbol
        // named `SLANG_globalParams` and save its address (if it is present).
        //
        // Note: We do *not* throw if this call to `cuModuleGetGlobal` returns an error, because
        // it is valid for the loaded module to not define a symbol of this name (e.g., if the
        // module had no global-scope parameters).
        //
        // Note: This is an important place where we rely on the details of how Slang generates
        // code for CUDA. If the name of the global symbols changes from `SLANG_globalParams` or
        // the code generation strategy changes in other ways (e.g., by declaring distinct global
        // symbols for each global parameter), then we will need to update this logic to match.
        //
        size_t globalParamsSymbolSize = 0;
        cuModuleGetGlobal(&mpGlobalParamsSymbol, &globalParamsSymbolSize, mCudaModule, "SLANG_globalParams");
    }

    CUDAProgramKernels::~CUDAProgramKernels()
    {
        // When destroying a CUDA program kernels object, we need to
        // unload the CUDA module it loaded.
        //
        if (mCudaModule)
        {
            cuModuleUnload(mCudaModule);
        }
    }

    // Because the allocation of `ParameterBlock`s in Falcor is driven by Slang-generated
    // reflection information, most of the existing reflection and parameter-binding
    // APIs continue to Just Work for a `CUDAProgram`. Note that we do *not* define
    // a custom `CUDAProgramVars` or `CUDAParameterBlock` type to represent the parameter
    // bindings for CUDA, and instead rely on the existing types.
    //
    // We will revisit the question of how shader parameter binding for CUDA needs to
    // be handled once we see the requirements that arise when we actually try to
    // execute a CUDA kernel.

    void CUDAProgram::dispatchCompute(
        ComputeContext* pContext,
        ComputeVars*    pVars,
        uint3 const&    gridShape)
    {
        // When a CUDA program is told to dispatch itself, it queries the
        // active version and the kernels for that active version,
        // and then delegates to the kernels for the details.
        //
        auto pProgramVersion = getActiveVersion();
        auto pKernels = std::dynamic_pointer_cast<const CUDAProgramKernels>(pProgramVersion->getKernels(pVars));
        pKernels->dispatchCompute(pVars, gridShape);
    }

    void CUDAProgramKernels::dispatchCompute(
        ComputeVars* pGlobalVars,
        uint3 const& gridShape) const
    {
        // Our primary goal here is to issue a single call to `cuLaunchKernel`,
        // and all the other work we do is toward the goal of determining its
        // parameters:
        //
        //      CUresult CUDAAPI cuLaunchKernel(
        //            CUfunction kernelFunc,
        //          unsigned int  gridShapeX, unsigned int  gridShapeY, unsigned int  gridShapeZ,
        //          unsigned int blockShapeX, unsigned int blockShapeY, unsigned int blockShapeZ,
        //          unsigned int dynamicSharedMemorySizeInBytes,
        //              CUstream stream,
        //                void** kernelParams,
        //                void** extra);
        //
        // We will work through the the paramete list in order to figure
        // out what we need to pass in.
        //
        // The kernel function is easy, since it is already stored on
        // the `CUDAProgramKernels`.
        //
        CUfunction kernelFunc = mCudaEntryPoint;

        // The grid shape is also easy, since it was passed in directly.

        // The block shape can be determined dynamically for CUDA, but we know
        // that the original Slang/HLSL shader had a `[numthreads(...)]` attribute
        // that should tell us what to do, so we can query that.
        //
        uint3 blockShape = pGlobalVars->getReflection()->getThreadGroupSize();

        // CUDA will automatically allocate the fixed/static shared memory
        // requirements for a kernel, and the dynamic size is only needed
        // when declaring a shared-memory arrays of statically unknown
        // size (which is not allowed in Slang/HLSL).
        //
        unsigned int dynamicSharedMemorySizeInBytes = 0;

        // For now we will always execute CUDA kernels on the default stream.
        //
        // TODO: This system could be extended to support multiple streams
        // by having a CUDA-specific subclass of `ComputeContext` that represents
        // a stream.
        //
        CUstream stream = 0;

        // Now we come to the tricky part of things: actually providing the
        // parameter data that the kernel requires.
        //
        // The typical way to drive `cuLaunchKernel` is with an array of
        // pointers to argument values for the parameters, but this requires
        // building up that array by knowing the number and type of each
        // parameter.
        //
        // We technically have access to information on the number and type
        // of the parameters (via reflection), but actually using reflection
        // data on the critical path here seems slow. We are going to use
        // a less well known part of the CUDA API instead of the ordinary
        // kernel parameters.
        //
        void** kernelParams = nullptr;

        // The `extra` parameter to `cuLaunchKernel` is structured as a kind
        // of key/value list, and it supports passing in the argument data
        // for *all* of the kernel parameters as a single buffer.
        //
        // We start by assuming that we can get the entry-point arguments
        // and encode them as a CUDA-friendly host-memory buffer.
        // (The details are implemented later in this file)
        //
        auto pEntryPointArgs = pGlobalVars->getEntryPointGroupVars(0);
        FALCOR_ASSERT(pEntryPointArgs);
        size_t entryPointArgsSize = 0;
        void* entryPointArgsData = pEntryPointArgs->getCUDAHostBuffer(entryPointArgsSize);

        // Now that we have a contiguous buffer that represents the entry-point
        // arguments, we can set up the `extra` argument so that it passes
        // in the pointer and size.
        //
        // When these extra options are specified, CUDA will use the provided
        // buffer instead of the explicit `kernelParams`.
        //
        void* extra[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, (void*)entryPointArgsData,
            CU_LAUNCH_PARAM_BUFFER_SIZE, &entryPointArgsSize,
            CU_LAUNCH_PARAM_END,
        };

        // At this point we've determined the values to pass for
        // all the arguments to `cuLaunchKernel`, and we are *almost*
        // ready to just go ahead and make the call.
        //
        // The last remaining wrinkle is that because of how typical
        // Slang/HLSL shaders are written, there might also be shader
        // parameters at global scope (not just entry-point parameters).
        //
        // As discussed earlier in this file, the Slang compiler bundles
        // all global-scope shader parameters into a single `__constant__`
        // variable named `SLANG_globalParams`. If there are any such
        // parameters, then the symbol will exist and we will have its
        // (device) address.
        //
        if (mpGlobalParamsSymbol)
        {
            // The argument values to use for global parameters were
            // already passed in as the `ProgramVars* pGlobalVars` argument
            // to this function.
            //
            // We start by querying for a device-memory buffer that can
            // represents those parameters in a CUDA-compatible layout.
            //
            size_t globalParamsDataSize = 0;
            void* pGlobalParamsDeviceData = pGlobalVars->getCUDADeviceBuffer(globalParamsDataSize);

            // Next we kick of an async copy from that device-memory buffer
            // over to the global `__constant__` symbol.
            //
            // Note: this is an important place where the design choice
            // in Slang to bundle all the global-scope parameters together
            // pays off; we can issue a single `cudaMemcpyAsync` to set
            // all the parameters, instead of having to use reflection
            // and emit one copy per parameter.
            //
            // Note: it might seem wasteful to first reify the argument
            // values into an allocated device-memory buffer and *then*
            // copy over to the address of the global `__constant__`. Why
            // can't we just have `pGlobalVars` write its state directly
            // to the global variable?
            //
            // The key reason why we cannot write directly to the global
            // `__constant__` in host code here is that there could still
            // be prior kernel launches in flight that are reading from
            // that data.
            //
            // In this case, we are relying on the fact that a `cudaMemcpyAsync`
            // in the `cudaMemcpyDeviceToDevice` mode is guaranteed not
            // to overlap with GPU kernel execution. Thus, our strategy
            // of first collecting the arguments in a device-memory buffer
            // and then copying it over avoids having to synchronize
            // execution between CPU and GPU.
            //
            cudaMemcpyAsync(
                (void*)mpGlobalParamsSymbol,
                (void*)pGlobalParamsDeviceData,
                globalParamsDataSize,
                cudaMemcpyDeviceToDevice,
                stream);
        }

        // Now that we've figured out all the argument values to use,
        // the actual call to `cuLaunchKernel` is straightforward.
        //
        FALCOR_CUDA_THROW_ON_FAIL(cuLaunchKernel(
            kernelFunc,
             gridShape.x,  gridShape.y,  gridShape.z,
            blockShape.x, blockShape.y, blockShape.z,
            dynamicSharedMemorySizeInBytes,
            stream,
            kernelParams,
            extra));

        // For bringup/debugging, we also immediately synchronize with
        // the CUDA context to ensure that our kernel has completed
        // execution. Any errors encountered during execution should
        // show up as an exception here.
        //
        // TODO: Once we are confident that the CUDA path is working
        // reasonably well, we can move this synchronization out and
        // give users another way to wait on CUDA.
        //
        FALCOR_CUDA_THROW_ON_FAIL(cuCtxSynchronize());
    }

#ifdef FALCOR_D3D12
    // The main missing detail that came up in dispatching CUDA
    // compute was the problem of getting host- or device-memory
    // buffers from a `ParameterBlock` (reminder: `ProgramVars`
    // is a subclass of `ParameterBlock`).
    //
    // We will bottleneck both the host-memory and device-memory
    // cases through a single routine because they share so much
    // of their logic.

    void* ParameterBlock::getCUDAHostBuffer(size_t& outSize)
    {
        return getCUDABuffer(CUDABufferKind::Host, outSize);
    }

    void* ParameterBlock::getCUDADeviceBuffer(size_t& outSize)
    {
        return getCUDABuffer(CUDABufferKind::Device, outSize);
    }

    void* ParameterBlock::getCUDABuffer(
        CUDABufferKind  bufferKind,
        size_t&         outSize)
    {
        // A parameter block might need to look at Slang specialization
        // information (e.g., the way that interface-type parameters
        // have been bound) before making decisions about how parameter
        // data should be laid out.
        //
        // We start by checking for any changes that might alter the
        // way the block/program is being specialized, and perform
        // subsequent steps using the refleciton information for
        // the specialized variant.
        //
        updateSpecialization();
        auto pReflector = mpSpecializedReflector.get();

        return getCUDABuffer(pReflector, bufferKind, outSize);
    }

    void* ParameterBlock::getCUDABuffer(
        const ParameterBlockReflection* pReflector,
        CUDABufferKind                  bufferKind,
        size_t&                         outSize)
    {
        // Because parameter blocks in Falcor are mutable rather than
        // write-once, it is possible that a change made to a byte in
        // a nested constant buffer or parameter block could have caused
        // it to be reallocated, getting a new device address, and thus
        // requiring a new version of *this* block to be generated.
        //
        // We start by checking for "indirect" changes that need to
        // be accounted for in the change epoch of this block.
        //
        checkForIndirectChanges(pReflector);
        auto epochOfLastChange = mEpochOfLastChange;

        // We cache a CUDA buffer on this parameter block, and will
        // try to use it if no parameter values have changed.
        //
        // If there have been changes made to this block (or the blocks
        // it transitively points to), then we need to reallocate
        // and fill in the buffer.
        //
        if (mUnderlyingCUDABuffer.epochOfLastObservedChange != epochOfLastChange)
        {
            updateCUDABuffer(pReflector, bufferKind);
            mUnderlyingCUDABuffer.epochOfLastObservedChange = epochOfLastChange;
        }

        // Note: The above logic did *not* check that any cached buffer
        // has the required `bufferKind`. The reason for that is because
        // there is expected to be a clear split: an `EntryPointGroupVars`
        // will always be queried for a host-memory buffer, and all other
        // cases will always be queried for a device-memory buffer.
        //
        // We defensively check and then error out if the assumption didn't hold.
        //
        if (mUnderlyingCUDABuffer.kind != bufferKind)
        {
            throw RuntimeError("Inconsistent CUDA buffer kind requested");
        }

        // Whether or not we were able to use the cached buffer,
        // we return the buffer pointer and size once we are done.
        //
        outSize = mUnderlyingCUDABuffer.size;
        return mUnderlyingCUDABuffer.pData;
    }

    void ParameterBlock::updateCUDABuffer(
        const ParameterBlockReflection* pReflector,
        CUDABufferKind                  bufferKind)
    {
        // If there is an existing buffer already allocated, we need to free it.
        //
        if (mUnderlyingCUDABuffer.pData)
        {
            auto pBufferData = mUnderlyingCUDABuffer.pData;
            if (mUnderlyingCUDABuffer.kind == CUDABufferKind::Host)
            {
                // In the case of a host-memory buffer, we can free
                // it without worrying about any in-flight GPU accesses.
                //
                free(pBufferData);
            }
            else
            {
                // In the case of a device-memory buffer, we rely on
                // the existing logic for freeing `Buffer`s, rather
                // than try to use `cudaFree()` and have to worry
                // about CPU/GPU synchornization.
                //
                mUnderlyingCUDABuffer.pBuffer = nullptr;
            }
        }

        // The refleciton information can tell us how big the type
        // being stored in the block is, and that tells us how big
        // of a buffer to allocate.
        //
        auto bufferSize = pReflector->getElementType()->getByteSize();

        // We need to allocate a new buffer either in device or
        // host memory, as determined by the requested `bufferKind`.
        //
        Buffer::SharedPtr pBuffer;
        void* pBufferData = nullptr;
        //
        // The kind of buffer to allocate also determined the kind
        // of `cudaMemcpy` operation(s) we need to perform when
        // filling it in.
        //
        cudaMemcpyKind memcpyKind;

        if (bufferKind == CUDABufferKind::Device)
        {
            // Note: the device-memory buffer we allocate will only
            // be used in CUDA API calls, so we could in principle
            // just allocate it with `cudaMalloc()`. The problem
            // in that case would be handling deallocation of the
            // buffer at the right time.
            //
            // Instead, we allocate an ordinary `Buffer` and then
            // share it over to CUDA, so that we can allow the
            // existing Falcor memory management to apply.
            //
            ResourceBindFlags flags = ResourceBindFlags::Constant | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared;
            pBuffer = Buffer::create(bufferSize, flags);
            pBufferData = pBuffer->getCUDADeviceAddress();
            memcpyKind = cudaMemcpyHostToDevice;
        }
        else
        {
            pBufferData = malloc(bufferSize);
            memcpyKind = cudaMemcpyHostToHost;
        }

        // Once the allocation is done, we can fill in the storage
        // on this parameter block that tracks the underlying CUDA
        // buffer.
        //
        mUnderlyingCUDABuffer.pBuffer = pBuffer;
        mUnderlyingCUDABuffer.pData = pBufferData;
        mUnderlyingCUDABuffer.size = bufferSize;
        mUnderlyingCUDABuffer.kind = bufferKind;

        // Now we need tostart in on the task of actually filling in
        // the buffer with a representation of the values bound
        // into this `ParameterBlock`.
        //
        // For fields of "ordinary" types (scalars, vectors, matrices,
        // and arrays/structures of those), the values are simply stored
        // in the `mData` array of the parameter block, and we can
        // simply copy that data over to the CUDA buffer.
        //
        auto dataSize = getSize();
        FALCOR_ASSERT(dataSize <= bufferSize);
        FALCOR_CUDA_THROW_ON_FAIL(cudaMemcpy(pBufferData, mData.data(), dataSize, memcpyKind));

        // For fields with "extraordinary" types (buffers, textures, etc.)
        // the values are stored in the `ParameterBlock` as arrays associated
        // with the different "resource ranges" in its type layout.
        //
        // For CUDA, however, even these buffer/texture/etc. types are in
        // effect "ordinary" data, and they will be represented as plain
        // bytes in the buffer we are filling in.
        //
        // Thus, we need to walk through all of the resource ranges indicated
        // by the reflection information for this block, and assign appropriate
        // data for each one over to the CUDA buffer.
        //
        auto resourceRangeCount = pReflector->getResourceRangeCount();
        for (uint32_t resourceRangeIndex = 0; resourceRangeIndex < resourceRangeCount; ++resourceRangeIndex)
        {
            // We need both the information on how the resource range was allocated
            // into the storage of the `ParameterBlock`, and also on how it is to
            // be bound into the API-specific storage.
            //
            auto resourceRange = pReflector->getResourceRange(resourceRangeIndex);
            auto resourceRangeBindingInfo = pReflector->getResourceRangeBindingInfo(resourceRangeIndex);

            // Each resource range represents one or more values with the
            // same descriptor type.
            //
            ShaderResourceType descriptorType = resourceRange.descriptorType;
            size_t descriptorCount = resourceRange.count;

            // We will go ahead and loop through all of the descriptors, and set each
            // one individually.
            //
            // TODO: A more efficient approach might seek to hoist the `switch` statement
            // that follows outside of the loop, and instead perform the loop inside of
            // each `case`. The current appraoch has been taken to favor simple verification
            // of correctness over maximum efficiency.
            //
            for (uint32_t descriptorIndex = 0; descriptorIndex < descriptorCount; descriptorIndex++)
            {
                // Each buffer/texture/whatever value in the resource range is bound
                // at some "flat" index in one of the arrays stored on the `ParameterBlock`.
                // We can compute that flat index based on the information stored on the
                // resource range and the index of the descriptor (in the case of a range
                // that represents an array).
                //
                size_t flatIndex = resourceRange.baseIndex + descriptorIndex;

                // For CUDA, every parameter in a block is stored at some byte offset within
                // the data of that block, and its "register" index as stored in the Falcor
                // reflection data is actually its byte offset.
                //
                // We can thus compute the destination bytes within the buffer by offsetting
                // from the start of the buffer by the reflected register index.
                //
                auto pDest = (char*)pBufferData + resourceRangeBindingInfo.regIndex;

                // TODO: The above logic is not taking `descriptorIndex` into account. We have
                // a problem here that the Falcor encoding of resource ranges does not include
                // any information about the "stride" of each range (the increment to add to
                // get from one array element to the next). Without that information, we can
                // only handle ranges with a single element for now.
                //
                if (descriptorIndex != 0)
                {
                    throw RuntimeError("Unsupported: resource/object arrays in CUDA parameter block");
                }

                // The remaining work to be done depends entirely on the kind of
                // buffers/sampler/texture/whatever binding we are dealing with.
                //
                switch (resourceRangeBindingInfo.flavor)
                {
                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ConstantBuffer:
                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ParameterBlock:
                    {
                        // A Slang `ParameterBlock<X>` or `ConstantBuffer<X>` will translate
                        // to a simple `X*` in the output CUDA code, and as such they are
                        // among the simplest cases to handle here.
                        //
                        // We start by asking the "sub-object" represent the block/buffer to
                        // produce a CUDA-compatible device memory buffer.
                        //
                        auto pSubObject = mParameterBlocks[flatIndex].pBlock;
                        auto pSubObjectReflector = resourceRangeBindingInfo.pSubObjectReflector.get();
                        size_t subObjectSize = 0;
                        CUdeviceptr pSubObjectDevicePtr = (CUdeviceptr) pSubObject->getCUDABuffer(pSubObjectReflector, CUDABufferKind::Device, subObjectSize);

                        // Once we have the device-memory pointer that represents the sub-object,
                        // we simply write its bytes (the bytes of the *pointer* and not those
                        // being pointed to) into the buffer we are building.
                        //
                        FALCOR_CUDA_THROW_ON_FAIL(cudaMemcpy(pDest, &pSubObjectDevicePtr, sizeof(pSubObjectDevicePtr), memcpyKind));
                    }
                    break;

                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Simple:
                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::RootDescriptor:
                default:
                    // The common case for resource ranges is that they represent single
                    // descriptor-bound resources/samplers, and in these cases we need
                    // to consider the descriptor type to know what should be bound.
                    //
                    switch (descriptorType)
                    {
                    default:
                        FALCOR_UNREACHABLE();
                        return;

                    case ShaderResourceType::RawBufferSrv:
                    case ShaderResourceType::StructuredBufferSrv:
                        {
                            // A Slang `StructuredBuffer<X>` translates to CUDA as a
                            // structure with two fields:
                            //
                            // 1. An `X*` device pointer for the data.
                            // 2. A `size_t` for the element count.
                            //
                            // A `ByteAddressBuffer` translates in a way that is
                            // equivalent to a `StructuredBuffer<uint8_t>`.

                            // We start by computing the view that should be used
                            // (filling in a default view if one is not bound).
                            //
                            auto pView = mSRVs[flatIndex].pView;
                            if (!pView) pView = ShaderResourceView::getNullView(resourceRangeBindingInfo.dimension);

                            // Next, we rely on a utility function to give us
                            // a CUDA device pointer that is equivalent to the
                            // given view.
                            //
                            CUdeviceptr pViewDevicePtr = (CUdeviceptr) pView->getCUDADeviceAddress();

                            // The view itself should know how many elements it covers.
                            //
                            // TODO: We need to confirm that byte-address buffer views
                            // have `elementCount` set to the number of bytes.
                            //
                            size_t viewElementCount = pView->getViewInfo().elementCount;

                            // Once we've computed the values for fields (1) and (2)
                            // described above, we can fill them in.
                            //
                            FALCOR_CUDA_THROW_ON_FAIL(cudaMemcpy(pDest, &pViewDevicePtr, sizeof(pViewDevicePtr), memcpyKind));
                            FALCOR_CUDA_THROW_ON_FAIL(cudaMemcpy(pDest + sizeof(pViewDevicePtr), &viewElementCount, sizeof(viewElementCount), memcpyKind));

                            // TODO: If Slang ever adds support for the implicit atomic
                            // counter on a structured buffer, then the layout may need
                            // to change to include a third pointer.
                        }
                        break;

                    case ShaderResourceType::RawBufferUav:
                    case ShaderResourceType::StructuredBufferUav:
                        {
                            // The logic for seting a `RW(StructuredBuffer|ByteAddressBuffer)`
                            // is identical to that above for read-only buffers, with the
                            // exception of using the `UnorderedAccessView` type instead.

                            auto pView = mUAVs[flatIndex].pView;
                            if (!pView) pView = UnorderedAccessView::getNullView(resourceRangeBindingInfo.dimension);

                            CUdeviceptr pViewDevicePtr = (CUdeviceptr) pView->getCUDADeviceAddress();
                            size_t viewElementCount = pView->getViewInfo().elementCount;

                            FALCOR_CUDA_THROW_ON_FAIL(cudaMemcpy(pDest, &pViewDevicePtr, sizeof(pViewDevicePtr), memcpyKind));
                            FALCOR_CUDA_THROW_ON_FAIL(cudaMemcpy(pDest + sizeof(pViewDevicePtr), &viewElementCount, sizeof(viewElementCount), memcpyKind));
                        }
                        break;

                    case ShaderResourceType::Sampler:
                        // CUDA does not support separate samplers, so there is no
                        // meaningful translation of a Slang `SamplerState` over to CUDA.
                        // We can simply skip over these ranges.
                        //
                        // TODO: There is a risk of error if the user's code relies on separate
                        // samplers, and the binding logic here silently ignoring samplers
                        // seems to contribute to the problem. As it stands, there isn't a much
                        // better option that we can implement, since issuing an error here
                        // would make a lot of existing Falcor shader code incompatible with CUDA.
                        break;

                    case ShaderResourceType::TextureSrv:
                    case ShaderResourceType::TextureUav:
                    case ShaderResourceType::TypedBufferSrv:
                    case ShaderResourceType::TypedBufferUav:
                        // Slang translates any read-only texture/buffer that involves format
                        // conversion/interpretation into a `CUtexObject`.
                        //
                        // Similarly, any read-write or write-only texture/buffer that involves
                        // format conversion/interpretation translates as a `CUsurfObject`.
                        //
                        // TODO: Support for these cases needs to be added. They require adding
                        // support to resources/views for querying the CUDA texture/surface object
                        // handle.
                        //
                        throw RuntimeError("Unexpected: interface-type field in CUDA parameter block");
                        break;
                    }
                    break;

                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Interface:
                    {
                        // A Slang parameter of interface type (`IThing`) translates to
                        // a structured representation with the following pieces:
                        //
                        // 1. A pointer to the run-time type information for the concrete
                        //    type `C` being stored.
                        //
                        // 2. A pointer to a "witness table," which can be understood as
                        //    an array of function pointers that show how `C` implements
                        //    the chosen interface (`IThing`).
                        //
                        // 3. The actual bytes of a `C` value, if it fits within the storage
                        //    constraints imposed by `IThing`, or a device pointer to a `C`,
                        //    in the cse where `C` is tood big.
                        //
                        // TODO: Actually handling all these details requires further integration
                        // of Falcor with the Slang reflection information to allow  the required
                        // information to be looked up.
                        //
                        // For now we simply fail if there are any interface-type fields in
                        // the parameter block.
                        //
                        throw RuntimeError("Unexpected: interface-type field in CUDA parameter block");
                    }
                    break;
                }
            }
        }

        // Once all of the "extraordinary" parameters represented as resource
        // ranges have been copied over to the buffer, its contents should reflect
        // the current state of this block.
    }

    // In order to bind buffers/resources into a CUDA parameter block, we
    // need to be able to produce a CUDA-compatible device memory address
    // for those buffers/resources. E.g., the above code calls `getCUDADeviceAddress()`
    // on SRVs/UAVs and expects it to Just Work.
    //
    // We now turn our attention to implementing the parts of the `Buffer`
    // API that allow sharing with CUDA to work.

    void* Buffer::getCUDADeviceAddress() const
    {
        // Our goal is to get the device memory address at which this
        // buffer resides, so that we can share it with CUDA code.
        //
        // Because a `Buffer` directly represents a GPU allocation
        // (no implicit versioning), we can cache and re-use the
        // device address each time is is queried.
        //
        void* deviceAddress = mCUDADeviceAddress;
        if (!deviceAddress)
        {
            // If the device address has not been created/cached,
            // then we need to go about setting it up using the
            // CUDA API.
            //
            // No matter what, we need to know the size of the
            // buffer/memory that we plan to share.
            //
            auto sizeInBytes = getSize();

            // CUDA manages sharing of buffers through the idea of an
            // "external memory" object, which represents the relationship
            // with another API's objects.
            //
            // Just as with the device address, we cache and re-use the
            // external memory relationship if one already exists.
            //
            cudaExternalMemory_t externalMemory = (cudaExternalMemory_t)mCUDAExternalMemory;
            if (!externalMemory)
            {
                // If no external memory relationship exists, we will
                // try to set one up, which requires working with
                // the shared handle for this resource (which will
                // only exist if the resource was created with
                // sharing enabled).
                //
                HANDLE sharedHandle = getSharedApiHandle();
                if (sharedHandle == NULL) return nullptr;

                // In order to create the external memory association
                // for CUDA, we need to fill in a descriptor struct.
                //
                // Note: This logic is D3D12-specific, so in order for
                // Falcor to support other graphics APIs this code would
                // need to be moved into a D3D-specific location.
                //
                cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
                memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));
                externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
                externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
                externalMemoryHandleDesc.size = sizeInBytes;
                externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

                // Once we have filled in the descriptor, we can request
                // that CUDA create the required association between the
                // D3D buffer and a its own memory.
                //
                FALCOR_CUDA_THROW_ON_FAIL(cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));
                mCUDAExternalMemory = externalMemory;
            }

            // The CUDA "external memory" handle is not itself a device
            // pointer, so we need to query for a suitable device address
            // for the buffer with another call.
            //
            // Just as for the external memory, we fill in a descriptor
            // structure (although in this case we only need to specify
            // the size).
            //
            cudaExternalMemoryBufferDesc bufferDesc;
            memset(&bufferDesc, 0, sizeof(bufferDesc));
            bufferDesc.size = sizeInBytes;

            // Finally, we can "map" the buffer to get a device address.
            //
            FALCOR_CUDA_THROW_ON_FAIL(cudaExternalMemoryGetMappedBuffer(&deviceAddress, externalMemory, &bufferDesc));
            mCUDADeviceAddress = deviceAddress;
        }
        return deviceAddress;
    }

    void* Buffer::getCUDADeviceAddress(ResourceViewInfo const& viewInfo) const
    {
        // Getting the CUDA device address for a view of a buffer starts
        // with determining the device address of the buffer itself.
        //
        auto bufferAddress = getCUDADeviceAddress();

        // Next, we need to determine the offset from the start of the buffer
        // that we intend to use, which is determined by the index of the
        // first element in the view, and the size of each element.
        //
        // Note: in the case of a "raw" buffer (unformatted, non-structured)
        // the `firstElement` is assumed to be in units of bytes, and the
        // `Buffer::getElementSize()` function returns 1.
        //
        size_t offset = viewInfo.firstElement * getElementSize();

        return (char*)bufferAddress + offset;
    }
#endif // FALCOR_D3D12
}
#endif

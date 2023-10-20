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
#include "TestPyTorchPass.h"

namespace
{
const char kShaderFilename[] = "RenderPasses/TestPasses/TestPyTorchPass.cs.slang";
}

void TestPyTorchPass::registerScriptBindings(pybind11::module& m)
{
    pybind11::class_<TestPyTorchPass, RenderPass, ref<TestPyTorchPass>> pass(m, "TestPyTorchPass");

    pass.def("generateData", &TestPyTorchPass::generateData);
    pass.def("verifyData", &TestPyTorchPass::verifyData);
}

TestPyTorchPass::TestPyTorchPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderFilename).csEntry("writeBuffer");
        mpWritePass = ComputePass::create(mpDevice, desc);
    }
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderFilename).csEntry("readBuffer");
        mpReadPass = ComputePass::create(mpDevice, desc);
    }

    mpCounterBuffer = mpDevice->createBuffer(sizeof(uint32_t));
    mpCounterStagingBuffer = mpDevice->createBuffer(sizeof(uint32_t), ResourceBindFlags::None, MemoryType::ReadBack, nullptr);

#if FALCOR_HAS_CUDA
    // Initialize CUDA.
    if (!mpDevice->initCudaDevice())
        FALCOR_THROW("Failed to initialize CUDA device.");
#endif
}

TestPyTorchPass::~TestPyTorchPass()
{
#if FALCOR_HAS_CUDA
    mSharedWriteBuffer.free();
    mSharedReadBuffer.free();
#endif
}

Properties TestPyTorchPass::getProperties() const
{
    return {};
}

RenderPassReflection TestPyTorchPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    return reflector;
}

TestPyTorchPass::PyTorchTensor TestPyTorchPass::generateData(const uint3 dim, const uint32_t offset)
{
#if FALCOR_HAS_CUDA
    // We create a tensor and return to PyTorch. Falcor retains ownership of the memory.
    // The Pytorch side is free to access the tensor up until the next call to this function.
    // The caller is responsible for synchronizing the access or copying the data into its own memory.

    RenderContext* pRenderContext = mpDevice->getRenderContext();

    const size_t elemCount = (size_t)dim.x * dim.y * dim.z;
    const size_t byteSize = elemCount * sizeof(float);
    FALCOR_CHECK(byteSize <= std::numeric_limits<uint32_t>::max(), "Buffer is too large.");

    if (mpBuffer == nullptr || mpBuffer->getElementCount() < elemCount)
    {
        // Create data buffer and CUDA shared buffer for async PyTorch access.
        // Pytorch can access the data in the shared buffer while we generate new data into the data buffer.
        // It is fine to recreate the buffers here without syncing as the caller is responsible for synchronization.
        logInfo("Reallocating buffers to size {} bytes", byteSize);
        mpBuffer = mpDevice->createStructuredBuffer(
            sizeof(float),
            elemCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mSharedWriteBuffer = createInteropBuffer(mpDevice, byteSize);
    }

    auto var = mpWritePass->getRootVar();
    var["bufferUav"] = mpBuffer;
    var["CB"]["dim"] = dim;
    var["CB"]["offset"] = offset;

    logInfo("Generating data on {}x{}x{} grid", dim.x, dim.y, dim.z);
    mpWritePass->execute(pRenderContext, dim);

    // Copy data to shared CUDA buffer.
    pRenderContext->copyResource(mSharedWriteBuffer.buffer.get(), mpBuffer.get());

    // Wait for copy to finish.
    pRenderContext->waitForFalcor();

    // Construct PyTorch tensor from CUDA buffer.
    const size_t shape[3] = {dim.x, dim.y, dim.z};
    const pybind11::dlpack::dtype dtype = pybind11::dtype<float>();
    int32_t deviceType = pybind11::device::cuda::value;
    int32_t deviceId = 0; // TODO: Consistent enumeration of GPU device IDs.

    TestPyTorchPass::PyTorchTensor tensor = TestPyTorchPass::PyTorchTensor(
        (void*)mSharedWriteBuffer.devicePtr, 3, shape, pybind11::handle() /* owner */, nullptr /* strides */, dtype, deviceType, deviceId
    );
    return tensor;
#else
    FALCOR_THROW("CUDA is not available.");
#endif
}

bool TestPyTorchPass::verifyData(const uint3 dim, const uint32_t offset, TestPyTorchPass::PyTorchTensor data)
{
#if FALCOR_HAS_CUDA
    // Pytorch owns the memory for the tensor that is passed in.
    // We copy it into a shared CUDA/DX buffer and run a compute pass to verify its contents.
    // The caller is responsible for synchronizing so that the tensor is not modified while accessed here.

    RenderContext* pRenderContext = mpDevice->getRenderContext();

    // Verify that the data is a valid Torch tensor.
    if (!data.is_valid() || data.dtype() != pybind11::dtype<float>() || data.device_type() != pybind11::device::cuda::value)
    {
        logWarning("Expected CUDA float tensor");
        return false;
    }

    if (data.ndim() != 3 || data.shape(0) != dim.x || data.shape(1) != dim.y || data.shape(2) != dim.z)
    {
        logWarning("Unexpected tensor dimensions (dim {}, expected dim {})", uint3(data.shape(0), data.shape(1), data.shape(2)), dim);
        return false;
    }

    // Note: For dim == 1, the stride is undefined as we always multiply it by
    // the index 0. Different versions of pytorch seem to define the stride
    // differently. Here, we replace any undefined stride with zeros to make
    // the check work for different pytorch versions.
    const uint3 stride = {
        dim[0] > 1 ? data.stride(0) : 0,
        dim[1] > 1 ? data.stride(1) : 0,
        dim[2] > 1 ? data.stride(2) : 0,
    };
    const uint3 expectedStride = {
        dim[0] > 1 ? dim[1] * dim[2] : 0,
        dim[1] > 1 ? dim[2] : 0,
        dim[2] > 1 ? 1 : 0,
    };
    if (any(stride != expectedStride))
    {
        logWarning("Unexpected tensor layout (stride {}, expected stride {})", stride, expectedStride);
        return false;
    }

    // Create shared CUDA/DX buffer for accessing the data.
    const size_t elemCount = (size_t)dim.x * dim.y * dim.z;
    const size_t byteSize = elemCount * sizeof(float);
    FALCOR_CHECK(byteSize <= std::numeric_limits<uint32_t>::max(), "Buffer is too large.");

    if (mSharedReadBuffer.buffer == nullptr || mSharedReadBuffer.buffer->getSize() < byteSize)
    {
        // Note it is ok to free the buffer here without syncing as the buffer is not in use between iterations.
        // We sync below after copying data into it and after the compute pass has finished, we don't access it anymore.
        logInfo("Reallocating shared CUDA/DX buffer to size {} bytes", byteSize);
        mSharedReadBuffer.free();
        mSharedReadBuffer = createInteropBuffer(mpDevice, byteSize);
    }

    // Copy to shared CUDA/DX buffer for access from compute pass.
    CUdeviceptr srcPtr = (CUdeviceptr)data.data();
    cuda_utils::memcpyDeviceToDevice((void*)mSharedReadBuffer.devicePtr, (const void*)srcPtr, byteSize);

    // Wait for CUDA to finish the copy.
    pRenderContext->waitForCuda();

    pRenderContext->clearUAV(mpCounterBuffer->getUAV().get(), uint4(0));

    // Run compute pass to count number of elements in the tensor that has the expected value.
    auto var = mpReadPass->getRootVar();
    var["bufferSrv"] = mSharedReadBuffer.buffer;
    var["counter"] = mpCounterBuffer;
    var["CB"]["dim"] = dim;
    var["CB"]["offset"] = offset;

    logInfo("Reading [{}, {}, {}] tensor", dim.x, dim.y, dim.z);
    mpReadPass->execute(pRenderContext, dim);

    // Copy counter to staging buffer for readback.
    pRenderContext->copyResource(mpCounterStagingBuffer.get(), mpCounterBuffer.get());

    // Wait for results to be available.
    pRenderContext->submit(true);

    const uint32_t counter = *reinterpret_cast<const uint32_t*>(mpCounterStagingBuffer->map(Buffer::MapType::Read));
    mpCounterStagingBuffer->unmap();
    FALCOR_ASSERT(counter <= elemCount);

    if (counter != elemCount)
    {
        logWarning("Unexpected tensor data ({} out of {} values correct)", counter, elemCount);
        return false;
    }

    return true;
#else
    FALCOR_THROW("CUDA is not available.");
#endif
}

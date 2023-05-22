import torch
import numpy as np
import falcor
from falcor import *
from time import time
import random

def createTensor(dim, offset, device):
    data = torch.zeros(dim, dtype=torch.float32, device=torch.device('cpu'))
    for k in range(dim[0]):
        for j in range(dim[1]):
            for i in range(dim[2]):
                idx = (k * dim[1] + j) * dim[2] + i
                data[k][j][i] = idx + offset
    return data.to(device)

def testTensorToFalcor(device, test_pass, iterations=10):
    print('Testing passing tensors to Falcor')
    for offset in range(iterations):
        dim = random.sample(range(1, 32), 3)
        data = createTensor(dim, offset, device)
        #torch.cuda.synchronize(device) # Does not seem to be necessary

        res = test_pass.verifyData(uint3(dim[0],dim[1],dim[2]), offset, data)
        if not res:
            raise RuntimeError(f'Test {offset} to pass tensor to Falcor failed')

def testTensorFromFalcor(test_pass, iterations=10):
    print('Testing passing tensors from Falcor')
    for offset in range(iterations):
        dim = random.sample(range(1, 32), 3)
        data = test_pass.generateData(uint3(dim[0],dim[1],dim[2]), offset)

        # Check the returned tensor
        if not isinstance(data, torch.Tensor):
            raise RuntimeError('Expected torch.Tensor object')
        if not data.is_cuda or data.dtype != torch.float32:
            raise RuntimeError('Expected CUDA float tensor')
        if list(data.size()) != dim:
            raise RuntimeError(f'Unexpected tensor dimensions (dim {list(data.size())}, expected dim {dim})')

        d = data.to("cpu")
        count = 0
        for k in range(dim[0]):
            for j in range(dim[1]):
                for i in range(dim[2]):
                    idx = (k * dim[1] + j) * dim[2] + i
                    if d[k][j][i] == float(idx + offset):
                        count += 1

        elemCount = dim[0] * dim[1] * dim[2]
        if count != elemCount:
            raise RuntimeError(f'Unexpected tensor data ({counter} out of {elemCount} values correct)", , elemCount)')

def main():
    random.seed(1)

    # Create torch CUDA device
    print('Creating CUDA device')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    device = torch.device("cuda:0")
    print(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Create testbed
    testbed = falcor.Testbed()
    render_graph = testbed.createRenderGraph("TestPybind")
    test_pass = render_graph.createPass("test_pybind_pass", "TestPyTorchPass", {})
    testbed.renderGraph = render_graph

    # Test passing tensors to Falcor
    testTensorToFalcor(device, test_pass, 100)

    # Test passing tensors from Falcor
    testTensorFromFalcor(test_pass, 100)

    print('SUCCESS!')

if __name__ == '__main__':
    main()

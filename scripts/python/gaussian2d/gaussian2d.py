"""
This is a simple example of using Falcor + DiffSlang + pytorch.
It learns to represent an image by a set of (additive) 2D gaussians.
A Falcor compute pass is used to render the gaussians to an image (forward).
DiffSlang is used to compute the gradients of the gaussians with respect to the
image (backward).
pytorch is used to optimize the gaussians to match a target image.
"""

import falcor
from pathlib import Path
import torch
import numpy as np
from PIL import Image

DIR = Path(__file__).parent

TARGET_IMAGE = DIR / "../../../media/test_images/monalisa.jpg"

BLOB_COUNT = 1024 * 4
RESOLUTION = 1024
ITERATIONS = 4000


class Splat2D:
    def __init__(self, device: falcor.Device):
        self.device = device

        self.blobs_buf = device.create_structured_buffer(
            struct_size=32,
            element_count=BLOB_COUNT,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared,
        )

        self.grad_blobs_buf = device.create_structured_buffer(
            struct_size=32,
            element_count=BLOB_COUNT,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared,
        )

        self.image_buf = device.create_structured_buffer(
            struct_size=12,
            element_count=RESOLUTION * RESOLUTION,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared,
        )

        self.grad_image_buf = device.create_structured_buffer(
            struct_size=12,
            element_count=RESOLUTION * RESOLUTION,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared,
        )

        self.forward_pass = falcor.ComputePass(
            device, file=DIR / "splat2d.cs.slang", cs_entry="forward_main"
        )

        self.backward_pass = falcor.ComputePass(
            device, file=DIR / "splat2d.cs.slang", cs_entry="backward_main"
        )

    def forward(self, blobs):
        self.blobs_buf.from_torch(blobs.detach())
        self.device.render_context.wait_for_cuda()
        vars = self.forward_pass.globals.forward
        vars.blobs = self.blobs_buf
        vars.blob_count = BLOB_COUNT
        vars.image = self.image_buf
        vars.resolution = falcor.uint2(RESOLUTION, RESOLUTION)
        self.forward_pass.execute(threads_x=RESOLUTION, threads_y=RESOLUTION)
        self.device.render_context.wait_for_falcor()
        return self.image_buf.to_torch([RESOLUTION, RESOLUTION, 3], falcor.float32)

    def backward(self, blobs, grad_intensities):
        self.grad_blobs_buf.from_torch(torch.zeros([BLOB_COUNT, 8]).cuda())
        self.blobs_buf.from_torch(blobs.detach())
        self.grad_image_buf.from_torch(grad_intensities.detach())
        self.device.render_context.wait_for_cuda()
        vars = self.backward_pass.globals.backward
        vars.blobs = self.blobs_buf
        vars.grad_blobs = self.grad_blobs_buf
        vars.blob_count = BLOB_COUNT
        vars.grad_image = self.grad_image_buf
        vars.resolution = falcor.uint2(RESOLUTION, RESOLUTION)
        self.backward_pass.execute(threads_x=RESOLUTION, threads_y=RESOLUTION)
        self.device.render_context.wait_for_falcor()
        return self.grad_blobs_buf.to_torch([BLOB_COUNT, 8], falcor.float32)


class Splat2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, blobs):
        image = splat2d.forward(blobs)
        ctx.save_for_backward(blobs)
        return image

    @staticmethod
    def backward(ctx, grad_intensities):
        blobs = ctx.saved_tensors[0]
        grad_blobs = splat2d.backward(blobs, grad_intensities)
        return grad_blobs


class Splat2DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, blobs):
        return Splat2DFunction.apply(blobs)


# create testbed instance with a window
testbed = falcor.Testbed(create_window=True, width=RESOLUTION, height=RESOLUTION)
testbed.show_ui = False
device = testbed.device

# create a texture for displaying the result in the window
testbed.render_texture = device.create_texture(
    format=falcor.ResourceFormat.RGB32Float,
    width=RESOLUTION,
    height=RESOLUTION,
    mip_levels=1,
    bind_flags=falcor.ResourceBindFlags.ShaderResource,
)

# setup 2D splatting function
splat2d = Splat2D(device)
Splat2DFunction.splat2d = splat2d

# setup gaussian parameters
blob_positions = torch.rand([BLOB_COUNT, 2]).cuda()
blob_scales = torch.log(torch.ones([BLOB_COUNT, 2]).cuda() * 0.005)
blob_rotations = torch.rand([BLOB_COUNT, 1]).cuda() * (2 * np.pi)
blob_colors = torch.rand([BLOB_COUNT, 3]).cuda() * 0.5 + 0.25

params = (blob_positions, blob_scales, blob_colors, blob_rotations)
for param in params:
    param.requires_grad = True

# load target image
image = Image.open(TARGET_IMAGE).resize([RESOLUTION, RESOLUTION]).convert("RGB")
target = np.asarray(image).astype(np.float32) / 255.0
target_cuda = torch.from_numpy(target).cuda()

# torch.autograd.gradcheck(Splat2DFunction.apply, (blobs), fast_mode=True)

model = Splat2DModule()


def optimize():
    optimizer = torch.optim.Adam(params, lr=0.01)
    sigmoid = torch.nn.Sigmoid()
    for iteration in range(ITERATIONS):
        optimizer.zero_grad()

        blobs = torch.concat(
            (
                blob_positions,
                torch.exp(blob_scales),
                sigmoid(blob_colors),
                blob_rotations,
            ),
            dim=1,
        )
        image = model.forward(blobs)
        loss = torch.nn.functional.l1_loss(image, target_cuda)
        # loss = torch.nn.functional.mse_loss(image, target_cuda)
        loss.backward()
        optimizer.step()

        if iteration % 5 == 0:
            print(f"iteration={iteration}, loss={loss.item()}")
            render_image = image.detach()
            # render_image = target_cuda.detach()
            # render_image = torch.lerp(image.detach(), target_cuda.detach(), 0.5)
            # render_image = torch.abs(image.detach() - target_cuda.detach())
            render_image = torch.pow(render_image, 2.2)
            testbed.render_texture.from_numpy(render_image.cpu().numpy())
            testbed.frame()
            if testbed.should_close:
                break


# run optimization
optimize()

# display image until window is closed
while not testbed.should_close:
    testbed.frame()

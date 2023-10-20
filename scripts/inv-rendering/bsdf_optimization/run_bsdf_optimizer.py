import torch
import falcor


class BSDFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, context):
        ctx.context = context
        device: falcor.Device = context["device"]
        scene: falcor.Scene = context["scene"]
        material_ids_buffer: falcor.Buffer = context["material_ids_buffer"]
        material_params_buffer: falcor.Buffer = context["material_params_buffer"]

        # Set material parameters
        material_params_buffer.from_torch(x.detach())
        device.render_context.wait_for_cuda()
        scene.set_material_params(material_ids_buffer, material_params_buffer)
        device.render_context.wait_for_falcor()

        # Compute loss
        params_ref = context["params_ref"]
        diff = x - params_ref
        loss = torch.abs(diff).sum()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        context = ctx.context
        device: falcor.Device = context["device"]
        bsdf_optimizer: falcor.BSDFOptimizer = context["bsdf_optimizer"]

        grad_buffer = bsdf_optimizer.compute_bsdf_grads()
        device.render_context.wait_for_falcor()
        grad = grad_buffer.to_torch([falcor.Material.PARAM_COUNT], falcor.float32)

        return (grad, None)


class BSDFEvalModule(torch.nn.Module):
    def __init__(
        self,
        device: falcor.Device,
        scene: falcor.Scene,
        bsdf_optimizer: falcor.BSDFOptimizer,
        material_ids_buffer: falcor.Buffer,
        material_params_buffer: falcor.Buffer,
        params_ref,
    ):
        super(BSDFEvalModule, self).__init__()
        self.context = dict(
            device=device,
            scene=scene,
            bsdf_optimizer=bsdf_optimizer,
            material_ids_buffer=material_ids_buffer,
            material_params_buffer=material_params_buffer,
            params_ref=params_ref,
        )

    def forward(self, x):
        return BSDFFunction.apply(x, self.context)


def main():
    # Create torch CUDA device
    print("Creating CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    torch_device = torch.device("cuda:0")
    print(torch_device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Create testbed
    testbed = falcor.Testbed()
    device = testbed.device
    render_graph = testbed.create_render_graph("BSDFOptimizer")
    # Set material IDs here
    init_material_id: falcor.ObjectID = 0
    ref_material_id: falcor.ObjectID = 1
    bsdf_optimizer: falcor.BSDFOptimizer = render_graph.create_pass(
        "bsdf_optimizer",
        "BSDFOptimizer",
        {"initMaterialID": init_material_id, "refMaterialID": ref_material_id},
    )
    testbed.render_graph = render_graph
    testbed.load_scene("test_scenes/bsdf_optimizer.pyscene")
    scene = testbed.scene

    material_ids_buffer = device.create_structured_buffer(
        struct_size=4,
        element_count=1,
        bind_flags=falcor.ResourceBindFlags.ShaderResource
        | falcor.ResourceBindFlags.UnorderedAccess
        | falcor.ResourceBindFlags.Shared,
    )
    material_params_buffer = device.create_structured_buffer(
        struct_size=4,
        element_count=falcor.Material.PARAM_COUNT,
        bind_flags=falcor.ResourceBindFlags.ShaderResource
        | falcor.ResourceBindFlags.UnorderedAccess
        | falcor.ResourceBindFlags.Shared,
    )

    init_material_id = bsdf_optimizer.init_material_id
    ref_material_id = bsdf_optimizer.ref_material_id

    def read_material_params(material_id : falcor.ObjectID):
        # Set up `material_ids_buffer`
        material_ids = torch.tensor([material_id], dtype=torch.int32)
        material_ids_buffer.from_torch(material_ids)
        device.render_context.wait_for_cuda()

        scene.get_material_params(material_ids_buffer, material_params_buffer)
        device.render_context.wait_for_falcor()

        # Fetch parameters from `material_params_buffer`
        params = torch.tensor([0.0] * falcor.Material.PARAM_COUNT)
        material_params_buffer.copy_to_torch(params)

        device.render_context.wait_for_cuda()

        return params

    # Get material parameters
    params_ref = read_material_params(ref_material_id)
    params_init = read_material_params(init_material_id)

    print("Material type:", scene.get_material(ref_material_id).type)
    print("Initial material params:", params_init)
    print("Reference material params:", params_ref)

    # Create optimizer
    params_init.requires_grad_()
    optimizer = torch.optim.Adam(
        [
            {"params": params_init, "lr": 1e-2},
        ],
        eps=1e-6,
    )

    # Workaround to fix "If capturable=False, state_steps should not be CUDA tensors"
    for param_group in optimizer.param_groups:
        param_group["capturable"] = True

    bsdf_optimizer.bsdf_slice_resolution = 256
    bsdf_eval = BSDFEvalModule(
        device=device,
        scene=scene,
        bsdf_optimizer=bsdf_optimizer,
        material_ids_buffer=material_ids_buffer,
        material_params_buffer=material_params_buffer,
        params_ref=params_ref,
    )

    # Optimization
    iter_count = 200
    for i in range(iter_count):
        optimizer.zero_grad()
        loss = bsdf_eval(params_init)
        loss.backward()

        optimizer.step()

        if i % 10 == 0 or i == iter_count - 1:
            print("Iter {:d}: loss = {:f}".format(i, loss.item()))

    print("Optimized material params:", params_init)


if __name__ == "__main__":
    main()

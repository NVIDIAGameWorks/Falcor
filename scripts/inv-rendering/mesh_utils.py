import torch
import falcor
import numpy as np

EPS = 1e-10


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, dim=-1, keepdim=True)


def length(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(dot(x, x))


def length_safe(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps*eps))  # Clamp to avoid NaN gradients because grad(sqrt(0)) = NaN.


def normalize_safe(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / length_safe(x, eps)


class Mesh:
    def __init__(
        self,
        tri_idx=None,
        v_pos=None,
        v_norm=None,
        v_tangent=None,
        v_texcrd=None,
    ):
        self.tri_idx = tri_idx
        self.v_pos = v_pos
        self.v_norm = v_norm
        self.v_tangent = v_tangent
        self.v_texcrd = v_texcrd

        self.buffers = {
            "triangleIndices": None,
            "positions": None,
            "normals": None,
            "tangents": None,
            "texcrds": None,
        }

    def init_falcor(
        self,
        device: falcor.Device,
        scene: falcor.Scene,
        vertex_count: int,
        triangle_count: int,
    ):
        for key in self.buffers:
            elem_count = triangle_count if key == "triangleIndices" else vertex_count
            if (
                self.buffers[key] is None
                or self.buffers[key].element_count < elem_count
            ):
                self.buffers[key] = device.create_structured_buffer(
                    struct_size=12,
                    element_count=elem_count,
                    bind_flags=falcor.ResourceBindFlags.ShaderResource
                    | falcor.ResourceBindFlags.UnorderedAccess
                    | falcor.ResourceBindFlags.Shared,
                )

    def load_from_falcor(self, testbed: falcor.Testbed, mesh_id: int):
        scene = testbed.scene
        device = testbed.device
        mesh = scene.get_mesh(mesh_id)

        self.init_falcor(
            device, scene, mesh.vertex_count, mesh.triangle_count
        )
        scene.get_mesh_vertices_and_indices(mesh_id, self.buffers)

        # Copy from Falcor to PyTorch.
        self.tri_idx = torch.zeros([mesh.triangle_count, 3], dtype=torch.int32)
        self.buffers["triangleIndices"].copy_to_torch(self.tri_idx)

        self.v_pos = torch.zeros([mesh.vertex_count, 3], dtype=torch.float32)
        self.buffers["positions"].copy_to_torch(self.v_pos)

        self.v_texcrd = torch.zeros([mesh.vertex_count, 3], dtype=torch.float32)
        self.buffers["texcrds"].copy_to_torch(self.v_texcrd)

        device.render_context.wait_for_cuda()

    def update_to_falcor(self, testbed: falcor.Testbed, mesh_id: int):
        scene = testbed.scene
        device = testbed.device

        # Copy from PyTorch to Falcor.
        self.buffers["positions"].from_torch(self.v_pos.detach())
        self.buffers["normals"].from_torch(self.v_norm.detach())
        self.buffers["tangents"].from_torch(self.v_tangent.detach())
        self.buffers["texcrds"].from_torch(self.v_texcrd.detach())
        device.render_context.wait_for_cuda()

        # Bind shader data.
        scene.set_mesh_vertices(mesh_id, self.buffers)

    def compute_shading_frame(self):
        self.compute_normals()
        self.compute_tangents()

    # From nvdiffrec.
    # Compute smooth vertex normals.
    def compute_normals(self):
        idx = [
            self.tri_idx[:, 0].type(torch.int64),
            self.tri_idx[:, 1].type(torch.int64),
            self.tri_idx[:, 2].type(torch.int64),
        ]
        pos = [self.v_pos[idx[0], :], self.v_pos[idx[1], :], self.v_pos[idx[2], :]]
        face_normals = torch.cross(pos[1] - pos[0], pos[2] - pos[0])

        # Splat face normals to vertices.
        v_normals = torch.zeros_like(self.v_pos)
        v_normals.scatter_add_(0, idx[0][:, None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, idx[1][:, None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, idx[2][:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value.
        v_normals = torch.where(
            length(v_normals) > EPS,
            v_normals,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda"),
        )
        v_normals = normalize_safe(v_normals)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_normals))

        self.v_norm = v_normals

    # From nvdiffrec.
    # Compute tangent space from texture map coordinates.
    # Follows http://www.mikktspace.com/ conventions.
    def compute_tangents(self):
        idx = [
            self.tri_idx[:, 0].type(torch.int64),
            self.tri_idx[:, 1].type(torch.int64),
            self.tri_idx[:, 2].type(torch.int64),
        ]
        pos = [self.v_pos[idx[0], :], self.v_pos[idx[1], :], self.v_pos[idx[2], :]]
        texcrd = [
            self.v_texcrd[idx[0], :],
            self.v_texcrd[idx[1], :],
            self.v_texcrd[idx[2], :],
        ]

        v_tangents = torch.zeros_like(self.v_norm)

        # Compute tangent space for each triangle.
        uve1 = texcrd[1] - texcrd[0]
        uve2 = texcrd[2] - texcrd[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[:, 1:2] - pe2 * uve1[:, 1:2]
        denom = uve1[:, 0:1] * uve2[:, 1:2] - uve1[:, 1:2] * uve2[:, 0:1]

        # Avoid division by zerofor degenerated texture coordinates.
        tang = nom / torch.where(
            denom > 0.0, torch.clamp(denom, min=EPS), torch.clamp(denom, max=-EPS)
        )

        # Update all 3 vertices.
        for i in range(3):
            t_idx = idx[i][:, None].repeat(1, 3)
            v_tangents.scatter_add_(0, t_idx, tang)

        # Normalize, replace zero (degenerated) tangents with some default value.
        default_tangents = torch.where(
            dot(
                self.v_norm,
                torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device="cuda"),
            )
            > 0.9999,
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda"),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device="cuda"),
        )
        v_tangents = torch.where(length(v_tangents) > EPS, v_tangents, default_tangents)
        v_tangents = normalize_safe(v_tangents)

        # Make sure tangent is orthogonal to normal.
        v_tangents = normalize_safe(
            v_tangents - self.v_norm * dot(self.v_norm, v_tangents)
        )

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_tangents))

        self.v_tangent = v_tangents

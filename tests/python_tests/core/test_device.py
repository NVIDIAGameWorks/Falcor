import sys
import os
import unittest
import falcor
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.relpath(__file__))))
from helpers import for_each_device_type


class TestDevice(unittest.TestCase):
    @for_each_device_type
    def test_device(self, device: falcor.Device):
        print(
            f"""
info.adapter_name={device.info.adapter_name}
info.api_name={device.info.api_name}
limits.max_compute_dispatch_thread_groups={device.limits.max_compute_dispatch_thread_groups}
limits.max_shader_visible_samplers={device.limits.max_shader_visible_samplers}
"""
        )

    @for_each_device_type
    def test_buffer(self, device: falcor.Device):
        a = device.create_buffer(256)
        b = device.create_buffer(256)

        a_host = np.linspace(0, 255, 256, dtype=np.uint8)
        b_host = np.zeros(256, dtype=np.uint8)
        a.from_numpy(a_host)
        b.from_numpy(b_host)
        a_device = a.to_numpy()
        b_device = b.to_numpy()
        self.assertEqual(a_device.shape, (256,))
        self.assertEqual(a_device.dtype, np.uint8)
        self.assertTrue(np.all(a_device == a_host))
        self.assertEqual(b_device.shape, (256,))
        self.assertEqual(b_device.dtype, np.uint8)
        self.assertTrue(np.all(b_device == b_host))

        device.render_context.copy_resource(b, a)
        b_device = b.to_numpy()
        self.assertTrue(np.all(b_device == a_host))

    @for_each_device_type
    def test_typed_buffer_float(self, device: falcor.Device):
        a = device.create_typed_buffer(
            format=falcor.ResourceFormat.R32Float, element_count=1024
        )
        b = device.create_typed_buffer(
            format=falcor.ResourceFormat.R32Float, element_count=1024
        )

        a_host = np.linspace(0, 1, 1024, dtype=np.float32)
        b_host = np.linspace(1, 0, 1024, dtype=np.float32)
        a.from_numpy(a_host)
        b.from_numpy(b_host)
        a_device = a.to_numpy()
        b_device = b.to_numpy()
        self.assertEqual(a_device.shape, (1024,))
        self.assertEqual(a_device.dtype, np.float32)
        self.assertTrue(np.all(a_device == a_host))
        self.assertEqual(b_device.shape, (1024,))
        self.assertEqual(b_device.dtype, np.float32)
        self.assertTrue(np.all(b_device == b_host))

        device.render_context.copy_resource(b, a)
        b_device = b.to_numpy()
        self.assertTrue(np.all(b_device == a_host))

    @for_each_device_type
    def test_typed_buffer_float16_2(self, device: falcor.Device):
        a = device.create_typed_buffer(
            format=falcor.ResourceFormat.RG16Float, element_count=1024
        )
        b = device.create_typed_buffer(
            format=falcor.ResourceFormat.RG16Float, element_count=1024
        )

        a_host = np.linspace(0, 1, 2048, dtype=np.float16)
        b_host = np.linspace(1, 0, 2048, dtype=np.float16)
        a.from_numpy(a_host)
        b.from_numpy(b_host)
        a_device = a.to_numpy()
        b_device = b.to_numpy()
        self.assertEqual(a_device.shape, (1024, 2))
        self.assertEqual(a_device.dtype, np.float16)
        self.assertTrue(np.all(a_device.flatten() == a_host))
        self.assertEqual(b_device.shape, (1024, 2))
        self.assertEqual(b_device.dtype, np.float16)
        self.assertTrue(np.all(b_device.flatten() == b_host))

        device.render_context.copy_resource(b, a)
        b_device = b.to_numpy()
        self.assertTrue(np.all(b_device.flatten() == a_host))

    @for_each_device_type
    def test_texture_1d(self, device: falcor.Device):
        a = device.create_texture(width=128, format=falcor.ResourceFormat.R8Uint)
        b = device.create_texture(width=128, format=falcor.ResourceFormat.R8Uint)
        a_host = np.linspace(0, 255, 128, dtype=np.uint8)
        b_host = np.linspace(255, 0, 128, dtype=np.uint8)
        a.from_numpy(a_host)
        b.from_numpy(b_host)
        a_device = a.to_numpy()
        b_device = b.to_numpy()
        self.assertEqual(a_device.shape, (128,))
        self.assertEqual(a_device.dtype, np.uint8)
        self.assertTrue(np.all(a_device == a_host))
        self.assertEqual(b_device.shape, (128,))
        self.assertEqual(b_device.dtype, np.uint8)
        self.assertTrue(np.all(b_device == b_host))

        device.render_context.copy_resource(b, a)
        b_device = b.to_numpy()
        self.assertTrue(np.all(b_device == a_host))

    @for_each_device_type
    def test_texture_2d(self, device: falcor.Device):
        a = device.create_texture(
            width=128, height=64, format=falcor.ResourceFormat.RGBA32Float
        )
        b = device.create_texture(
            width=128, height=64, format=falcor.ResourceFormat.RGBA32Float
        )
        a_host = np.reshape(
            np.linspace(0, 1, 128 * 64 * 4, dtype=np.float32), (64, 128, 4)
        )
        b_host = np.reshape(
            np.linspace(1, 0, 128 * 64 * 4, dtype=np.float32), (64, 128, 4)
        )
        a.from_numpy(a_host)
        b.from_numpy(b_host)
        a_device = a.to_numpy()
        b_device = b.to_numpy()
        self.assertEqual(a_device.shape, (64, 128, 4))
        self.assertEqual(a_device.dtype, np.float32)
        self.assertTrue(np.all(a_device == a_host))
        self.assertEqual(b_device.shape, (64, 128, 4))
        self.assertEqual(b_device.dtype, np.float32)
        self.assertTrue(np.all(b_device == b_host))

        device.render_context.copy_resource(b, a)
        b_device = b.to_numpy()
        self.assertTrue(np.all(b_device == a_host))

    @for_each_device_type
    def test_texture_3d(self, device: falcor.Device):
        a = device.create_texture(
            width=128, height=64, depth=32, format=falcor.ResourceFormat.RG16Float
        )
        b = device.create_texture(
            width=128, height=64, depth=32, format=falcor.ResourceFormat.RG16Float
        )
        a_host = np.reshape(
            np.linspace(0, 1, 128 * 64 * 32 * 2, dtype=np.float16), (32, 64, 128, 2)
        )
        b_host = np.reshape(
            np.linspace(1, 0, 128 * 64 * 32 * 2, dtype=np.float16), (32, 64, 128, 2)
        )
        a.from_numpy(a_host)
        b.from_numpy(b_host)
        a_device = a.to_numpy()
        b_device = b.to_numpy()
        self.assertEqual(a_device.shape, (32, 64, 128, 2))
        self.assertEqual(a_device.dtype, np.float16)
        self.assertTrue(np.all(a_device == a_host))
        self.assertEqual(b_device.shape, (32, 64, 128, 2))
        self.assertEqual(b_device.dtype, np.float16)
        self.assertTrue(np.all(b_device == b_host))

        device.render_context.copy_resource(b, a)
        b_device = b.to_numpy()
        self.assertTrue(np.all(b_device == a_host))


if __name__ == "__main__":
    unittest.main()

import os
import unittest
import falcor

if os.name == 'nt':
    DEVICE_TYPES=[falcor.DeviceType.D3D12, falcor.DeviceType.Vulkan]
else:
    DEVICE_TYPES=[falcor.DeviceType.Vulkan]

class TestDevice(unittest.TestCase):

    def test_create(self):
        for device_type in DEVICE_TYPES:
            with self.subTest():
                print(f"device_type={device_type}")

                device = falcor.Device(type=device_type)
                self.assertTrue(device is not None)

                print(f"info.adapter_name={device.info.adapter_name}")
                print(f"info.api_name={device.info.api_name}")

                print(f"limits.max_compute_dispatch_thread_groups={device.limits.max_compute_dispatch_thread_groups}")
                print(f"limits.max_shader_visible_samplers={device.limits.max_shader_visible_samplers}")

                del device

if __name__ == '__main__':
    unittest.main()

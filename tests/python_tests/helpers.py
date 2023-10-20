import os
import falcor

class DeviceCache:
    """
    A cache of device instances, one for each device type.
    """

    def __init__(self):
        self.devices = {}

    def get(self, device_type):
        if device_type not in self.devices:
            self.devices[device_type] = falcor.Device(type=device_type)
        return self.devices[device_type]


device_cache = DeviceCache()

if os.name == "nt":
    DEVICE_TYPES = [falcor.DeviceType.D3D12, falcor.DeviceType.Vulkan]
else:
    DEVICE_TYPES = [falcor.DeviceType.Vulkan]

def for_each_device_type(func):
    """
    A decorator that runs a test function for each device type.
    """

    def wrapper(*args, **kwargs):
        for device_type in DEVICE_TYPES:
            with args[0].subTest():
                func(*args, **kwargs, device=device_cache.get(device_type))

    return wrapper

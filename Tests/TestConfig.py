import os

scenes = [ "Arcade/Arcade.fscene", "SunTemple/SunTemple.fscene", "Bistro/Bistro_Interior.fscene", "Cerberus/Standard/Cerberus.fscene" ]
exitTime = 40
frames = [ 16, 32, 64, 128, 256, 512, 1024, 2048]
defaultConfig = 'ReleaseD3D12'
localTestDir = 'TestResults'
tolerance = 0.00001
tolerance_lower = 0.1
framerate = 60

# Relative to root directory
ignoreDirectories = {}
ignoreDirectories['ReleaseD3D12'] = []
ignoreDirectories['ReleaseVK'] = [ os.path.join('Framework', 'Internals') ]

# Supported image extensions
imageExtensions = ['.png', '.jpg', '.tga', '.bmp', '.pfm', '.exr']    

slnFile = "falcor.sln"
mogwaiExe = 'Mogwai'
ultExe = 'FalcorTest'
if os.name == 'nt':
    mogwaiExe += '.exe'
    ultExe += '.exe'

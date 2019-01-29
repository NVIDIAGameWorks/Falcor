import os

TestConfig = {}
TestConfig['Scenes'] = [ "Arcade/Arcade.fscene", "SunTemple/SunTemple.fscene", "Bistro/Bistro_Interior.fscene" ]
TestConfig['Images'] = [ "StockImage.jpg" ]
TestConfig['Duration'] = 360
TestConfig['Frames'] = [ 16, 32, 128, 256]
TestConfig['DefaultConfiguration'] = 'ReleaseD3D12'
TestConfig['LocalTestingDir'] = 'testing'
TestConfig['Tolerance'] = 200.0
TestConfig['Tolerance_Lower'] = 0.1
TestConfig['FixedTimeDelta'] = 0.01666

# Relative to root directory
IgnoreDirectories = {}
IgnoreDirectories['ReleaseD3D12'] = []
IgnoreDirectories['ReleaseVK'] = [ os.path.join('Framework', 'Internals') ]

# Supported image extensions
ImageExtensions = ['.png', '.jpg', '.tga', '.bmp', '.pfm', '.exr']

# get 'static' part of the arguments
def get_config_arguments():
    current_args = '-testFrames '
    
    for testFrame in TestConfig['Frames']:
        current_args = current_args + str(testFrame) + ' '
        
    if 'Duration' in TestConfig:
        current_args = current_args + '-shutdownframe ' + str(TestConfig['Duration']) + ' ';
        
    if 'FixedTimeDelta' in TestConfig:
        current_args = current_args + '-fixedtimedelta ' + str(TestConfig['FixedTimeDelta']) + ' ';
    
    return current_args
    

test_arguments = get_config_arguments()
if os.name == 'nt':
    viewer_executable = 'RenderGraphViewer.exe'
else:
    viewer_executable = 'RenderGraphViewer'
    
if os.name == 'nt':
    unit_tests_executable = 'FalcorTest.exe'
else:
    unit_tests_executable = 'FalcorTest'
num_scenes = len(TestConfig["Scenes"])
num_images = len(TestConfig["Images"])

# get the arguments that change for each test
def get_next_scene_args(scene_index):
    if scene_index >= num_scenes : 
       scene_index =  num_scenes - 1
    
    args = '-scene ' + TestConfig["Scenes"][scene_index] + ' '
    return args

def get_next_image_args(image_index):
    if image_index >= num_images : 
       image_index =  num_images - 1

    args = '-image ' + TestConfig['Images'][image_index] + ' '
    return args
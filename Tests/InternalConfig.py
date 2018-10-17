import os

TestConfig = {}
TestConfig['Scenes'] = [ "Arcade/Arcade.fscene", "SunTemple/SunTemple.fscene", "Bistro/Bistro_Interior.fscene" ]
TestConfig['Images'] = [ "StockImage.jpg" ]
TestConfig['Duration'] = 5
TestConfig['Times'] = [ 1, 2, 3, 4]
TestConfig['DefaultConfiguration'] = 'ReleaseD3D12'
TestConfig['LocalTestingDir'] = 'testing'
TestConfig['Tolerance'] = 0.1

##if 'FixedTimeDelta' in TestConfig:
##    args = '-fixedtimedelta ' +  TestConfig['FixedTimeDelta'] + ' ';

# get 'static' part of the arguments
def get_config_arguments():
    current_args =  "-test "
    
    current_args = current_args + '-sstimes '
    
    for ssTime in TestConfig['Times']:
        current_args = current_args + str(ssTime) + ' '
        
    if 'Duration' in TestConfig:
        current_args = current_args + '-shutdowntime ' + str(TestConfig['Duration']);
        
    return current_args
    

test_arguments = get_config_arguments()
if os.name == 'nt':
    viewer_executable = 'RenderGraphViewer.exe'
else:
    viewer_executable = 'RenderGraphViewer'
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
TestConfig = {}
TestConfig['Scenes'] = [ "Arcade/arcade.fscene", "SunTemple/sunTemple.fscene" ]
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
    
# get the arguments that change for each test
def get_next_scene_args(scene_index):
    args = '-defaultScene ' + TestConfig["Scenes"][scene_index] + ' '
    return args

def get_next_image_args(image_index):
    args = '-defaultImage ' + TestConfig['Images'][image_index] + ' '
    return args
    

test_arguments = get_config_arguments()
viewer_executable = 'RenderGraphViewer.exe'
num_scenes = len(TestConfig["Scenes"])
num_images = len(TestConfig["Images"])

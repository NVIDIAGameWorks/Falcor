TestConfig = {}
TestConfig['Scenes'] = [ "Arcade/arcade.fscene", "SunTemple/sunTemple.fscene" ]
TestConfig['Images'] = [ "TestImage.jpg" ]
TestConfig['Duration'] = 5
TestConfig['Times'] = [ 1, 2, 3, 4]
TestConfig['DefaultConfiguration'] = 'ReleaseD3D12'
TestConfig['LocalTestingDir'] = 'testing'
TestConfig['Tolerance'] = 0.0

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
def get_next_arguments(run_image_tests, test_index):
    args = ""
    num_scenes = len(TestConfig["Scenes"])
    num_images = len(TestConfig["Images"])
    
    if (not run_image_tests and (test_index >= num_scenes)) or (test_index >= (num_scenes + num_images)):
            return ''
    
    if 'FixedTimeDelta' in TestConfig:
        args = '-fixedtimedelta ' +  TestConfig['FixedTimeDelta'] + ' ';
    
    if (num_scenes > test_index):
        args = args + '-defaultScene '+ TestConfig["Scenes"][test_index] + ' '
    else:
        image_test_index = test_index - num_scenes
        if (num_images > image_test_index):
            args = args + '-defaultImage ' + TestConfig['Images'][image_test_index] + ' '
            
    return args

    
test_arguments = get_config_arguments()
viewer_executable = 'RenderGraphViewer.exe'
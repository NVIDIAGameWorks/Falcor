import os
import json

#only uses this if fails to find file
default_json_data = {
    'Windows':
    {   
        'Destination Target' : 'C:\\Falcor\\GitHub\\',
        'Reference Target ' : '\\\\netapp-wa02\\public\\Falcor\\Github\\References\\',
        'Email List' : '\\\\netapp-wa02\\public\\Falcor\\email.txt',
        'Results Summary Target' : '\\\\netapp-wa02\\public\\Falcor\\Github\\Results\\'
    },
    'Linux':
    {   
        'Destination Target' : '/home/nvrgfxtest/Desktop/FalcorGitHub/',
        'Reference Target' : 'media/netapp/Falcor/Github/References/',
        'Email List' : '/media/netapp/Falcor/email.txt',
        'Results Summary Target' : '/media/netapp/Falcor/Github/Results/'
    }
}

machine_process_default_kill_time = 1200.0
machine_relative_checkin_local_results_directory = os.path.join('TestsResults', 'local-results-directory')

try:
    json_filename = 'MachinePaths.json'
    json_file = open(json_filename)
    json_data = json.load(json_file)
except: 
    print('Failed to open ' + json_filename + ', attempting to use defaults specified in MachineConfigs.py')
    json_data = default_json_data

if os.name == 'nt':
    machine_name = os.environ['COMPUTERNAME']
    platform_data = json_data['Windows']
else:
    import socket
    machine_name = socket.gethostname()
    platform_data = json_data['Linux']

machine_name = machine_name.lower()
machine_reference_directory = platform_data['Reference Target']
destination_target = platform_data['Destination Target']
machine_email_recipients = platform_data['Email List']
machine_results_summary_target = platform_data['Results Summary Target']

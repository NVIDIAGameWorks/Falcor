import os
import json

json_data = {
    'Windows':
    {   
        'Destination Target' : 'C:\\Falcor\\GitHub\\',
        'Remote References' : '\\\\netapp-wa02\\public\\Falcor\\References\\',
        'Email List' : '\\\\netapp-wa02\\public\\Falcor\\email.txt',
        'Results Summary Target' : '\\\\netapp-wa02\\public\\Falcor\\GitHub\\Results\\',
        'Default Main Directory' : '..\\',
        'Results Cache Directory' : 'C:\\FalcorResults\\',
        'Packman Repo Path' : 'C:\\packman-repo\\',
        'Local References' : 'C:\\FalcorRefs\\'
    },
    'Linux':
    {   
        'Destination Target' : '/home/nvrgfxtest/Desktop/FalcorGitHub/',
        'Remote References' : '/media/netapp/Falcor/References/',
        'Email List' : '/media/netapp/Falcor/email.txt',
        'Results Summary Target' : '/media/netapp/Falcor/GitHub/Results/',
        'Default Main Directory' : '../',
        'Results Cache Directory' : '/home/FalcorResults/',
        'Packman Repo Path' : '/packman-repo/',
        'Local References' : '/home/nvrgfxtest/FalcorRefs'
    }
}

machine_process_default_kill_time = 1200.0

if os.name == 'nt':
    machine_name = os.environ['COMPUTERNAME']
    platform_data = json_data['Windows']
else:
    import socket
    machine_name = socket.gethostname()
    platform_data = json_data['Linux']

packman_repo = platform_data['Packman Repo Path'];
results_cache_directory = platform_data['Results Cache Directory'];
default_reference_url = 'https://github.com/NVIDIAGameWorks/Falcor'
machine_name = machine_name.lower()
remote_reference_directory = platform_data['Remote References']
local_reference_directory = platform_data['Local References']
destination_target = platform_data['Destination Target']
machine_email_recipients = platform_data['Email List']
machine_results_summary_target = platform_data['Results Summary Target']
#for running test sets, not collections. Like check in tests
default_reference_machine_name = 'default'
default_reference_branch_name = 'develop'
default_main_dir = platform_data['Default Main Directory']
import os
import json

json_data = {
    'Windows':
    {   
        'Destination Target' : 'C:\\Falcor\\GitHub\\',
        'Reference Target' : '\\\\netapp-wa02\\public\\Falcor\\GitHub\\References\\',
        'Email List' : '\\\\netapp-wa02\\public\\Falcor\\email.txt',
        'Results Summary Target' : '\\\\netapp-wa02\\public\\Falcor\\GitHub\\Results\\',
        'Default Main Directory' : '..\\'
    },
    'Linux':
    {   
        'Destination Target' : '/home/nvrgfxtest/Desktop/FalcorGitHub/',
        'Reference Target' : '/media/netapp/Falcor/GitHub/References/',
        'Email List' : '/media/netapp/Falcor/email.txt',
        'Results Summary Target' : '/media/netapp/Falcor/GitHub/Results/',
        'Default Main Directory' : '../'
    }
}

machine_process_default_kill_time = 1200.0
machine_relative_checkin_local_results_directory = os.path.join('TestsResults', 'local-results-directory')

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
#for running test sets, not collections. Like check in tests
default_reference_machine_name = 'default'
default_reference_branch_name = 'master'
default_main_dir = platform_data['Default Main Directory']
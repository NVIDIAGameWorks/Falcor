import os

machine_process_default_kill_time = 1200.0
machine_relative_checkin_local_results_directory = os.path.join('TestsResults', 'local-results-directory')

#Get Machine Name
if os.name == 'nt':
    machine_name = os.environ['COMPUTERNAME']
    netapp_path = '\\\\netapp-wa02\\public\\'
else:
    import socket
    machine_name = socket.gethostname()
    netapp_path = '/media/netapp/'
machine_name = machine_name.lower()

#Try to find reference path
try:
    machine_reference_directory = os.environ['FALCOR_REFERENCE_PATH']
except KeyError:
    #couldn't find it, use the default. 
    machine_reference_directory = os.path.join(netapp_path, 'Falcor', 'Github', 'References')
    print('Failed find enviornment variable FALCOR_REFERENCE_PATH, using default path: ' + machine_reference_directory)

#Get path to email recipients list
machine_email_recipients = os.path.join(netapp_path, 'Falcor', 'email.txt')



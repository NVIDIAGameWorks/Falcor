import os

if os.name == 'nt':
    machine_name = os.environ['COMPUTERNAME']
    netappPath = '\\\\netapp-wa02\\public\\'
else:
    import socket
    machine_name = socket.gethostname()
    netappPath = '/media/netapp/'

machine_name = machine_name.lower()

machine_build_script = "BuildSolution.bat"
machine_process_default_kill_time = 1200.0

machine_relative_checkin_local_results_directory = os.path.join('TestsResults', 'local-results-directory')
machine_email_recipients = os.path.join(netappPath, 'Falcor', 'email.txt')

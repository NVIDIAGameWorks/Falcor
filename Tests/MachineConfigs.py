import os

machine_name = os.environ['COMPUTERNAME']
machine_name = machine_name.lower()

machine_build_script = "BuildSolution.bat"
machine_process_default_kill_time = 1200.0

machine_relative_checkin_local_results_directory = "TestsResults\\local-results-directory\\"
machine_default_checkin_reference_directory = '\\\\netapp-wa02\\public\\Falcor\\GitHub\\References\\nvrgfx-dt\\master\\GitHub_DailyTests\\'
machine_email_recipients = "\\\\netapp-wa02\\public\\Falcor\\email.txt"
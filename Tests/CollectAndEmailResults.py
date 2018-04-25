import MachineConfigs as machine_configs
import Helpers as helpers
import os
from datetime import date

target_base = machine_configs.machine_results_summary_target
date_str = date.today().strftime("%m-%d-%y")
today_dir = os.path.join(target_base, date_str)    

files = []
for file in os.listdir(today_dir):
    if file.endswith('.html'):
        files.append(os.path.join(today_dir, file))
    
num_files = len(files)
num_success = 0
for file in files:
    if file.find('[SUCCESS]') != -1:
        num_success += 1

subject = ''
if num_files == num_success:
    subject += '[SUCCESS (' 
else:
    subject += '[FAILURE (' 

subject += str(num_success) + ' / ' + str(num_files) + ')]'
subject += '_' + date_str
helpers.dispatch_email(subject, files)

    
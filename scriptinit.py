import subprocess
import time
script1_path = 'recorder.py'
script2_path = 'predictor.py'
subprocess.Popen(['python',script1_path])
time.sleep(15)
subprocess.Popen(['python',script2_path])
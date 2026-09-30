import subprocess
import sys
import time

NUM_CLIENTS = 8
processes = []

# avvia client
for i in range(NUM_CLIENTS):
    p = subprocess.Popen(
        [sys.executable, "client_app.py", str(i)]
    )
    processes.append(p)
    time.sleep(0.3)

# aspetta fine training
for p in processes:
    p.wait()
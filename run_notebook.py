"""Exécute un notebook en place (outputs conservés) : python run_notebook.py 01_got_ocr2.ipynb [timeout_s]"""
import sys
import time

import nbformat
from nbclient import NotebookClient

path = sys.argv[1]
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
nb = nbformat.read(path, as_version=4)
t0 = time.time()
try:
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
    status = "OK"
except Exception as e:
    status = f"ERREUR : {type(e).__name__}: {str(e)[-1500:]}"
nbformat.write(nb, path)
print(f"{path} : {status} ({time.time() - t0:.0f}s)")

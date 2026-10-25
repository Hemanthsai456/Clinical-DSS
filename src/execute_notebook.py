"""
Execute notebooks/01_data_audit.ipynb end-to-end and save outputs in-place.
"""
import sys
import asyncio
import nbformat
from nbclient import NotebookClient

# Set WindowsSelectorEventLoopPolicy on Windows for clean zmq subprocess handling
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

notebook_path = "notebooks/01_data_audit.ipynb"
print(f"Reading {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print("Executing notebook cells with NotebookClient (kernel: python3)...")
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

print(f"Saving executed notebook to {notebook_path}...")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("[SUCCESS] Notebook executed end-to-end without errors!")

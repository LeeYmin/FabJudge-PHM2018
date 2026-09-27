"""Execute 06c one cell at a time and retain output if a stop condition fires."""
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/06c_lstm_jev_semantic_gate.ipynb"
os.environ["PATH"] = str(ROOT / ".venv/Scripts") + os.pathsep + os.environ["PATH"]
os.environ["PYTHONIOENCODING"] = "utf-8"
notebook = nbformat.read(PATH, as_version=4)
client = NotebookClient(notebook, timeout=3600, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}})
with client.setup_kernel():
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        print(f"Running cell {index}", flush=True)
        try:
            client.execute_cell(cell, index)
        except Exception:
            nbformat.write(notebook, PATH)
            raise
        nbformat.write(notebook, PATH)
        for output in cell.get("outputs", []):
            if output.output_type == "stream" and output.name == "stdout":
                print(output.text[-2000:], flush=True)
print("06c notebook execution complete", flush=True)

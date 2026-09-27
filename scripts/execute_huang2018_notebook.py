"""Execute 04b with the bundled Python and save each completed cell."""

from pathlib import Path
import json
import os
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / ".venv" / "Lib" / "site-packages"
sys.path.insert(0, str(SITE))

spec_dir = ROOT / ".jupyter" / "kernels" / "fabjudge-runtime"
spec_dir.mkdir(parents=True, exist_ok=True)
(spec_dir / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "FabJudge runtime",
    "language": "python",
    "env": {
        "PYTHONPATH": os.pathsep.join((str(SITE), str(ROOT / "src"))),
        "MPLBACKEND": "Agg",
        "IPYTHONDIR": str(ROOT / ".ipython"),
    },
}), encoding="utf-8")
os.environ["JUPYTER_PATH"] = str(ROOT / ".jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")

import nbformat
from nbclient import NotebookClient

path = ROOT / "notebooks" / "04b_huang2018_reproduction.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=None, kernel_name="fabjudge-runtime", resources={"metadata": {"path": str(ROOT)}})
started = time.perf_counter()
try:
    with client.setup_kernel():
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                label = cell.source.splitlines()[0][:100]
                print(f"Cell {index+1}/{len(notebook.cells)}: {label}", flush=True)
                client.execute_cell(cell, index)
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "stream":
                        lines = output.get("text", "").splitlines()
                        if lines:
                            print("  " + " | ".join(lines[-3:])[:500], flush=True)
                nbformat.write(notebook, path)
finally:
    nbformat.write(notebook, path)
print(f"Completed in {time.perf_counter()-started:.1f} seconds: {path}", flush=True)

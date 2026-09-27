"""Execute 06e in the repository virtual environment and save cell outputs."""

import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/06e_fix_and_weakness_diagnosis.ipynb"
with tempfile.TemporaryDirectory() as temporary:
    kernel_dir = Path(temporary) / "kernels" / "fabjudge-06e"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "FabJudge 06e", "language": "python"}), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = temporary
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name="fabjudge-06e",
                            resources={"metadata": {"path": str(root)}})
    client.execute()
    nbformat.write(notebook, path)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            errors = [item for item in cell.outputs if item.output_type == "error"]
            if errors:
                raise RuntimeError(f"Notebook cell {index} failed: {errors[0].evalue}")
    print(f"Executed {path.name}: {sum(c.cell_type == 'code' for c in notebook.cells)} code cells")

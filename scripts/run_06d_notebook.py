"""Execute 06d, retaining cell outputs even if a stop condition fires."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/06d_lstm_jev_oof_edge.ipynb"
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                        resources={"metadata": {"path": str(root)}})
try:
    client.execute()
finally:
    nbformat.write(nb, path)

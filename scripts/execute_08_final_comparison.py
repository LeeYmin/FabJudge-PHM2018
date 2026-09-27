"""Execute only notebook 08, saving partial output even if an integrity check fails."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'notebooks/08_final_comparison.ipynb'
nb = nbformat.read(path, as_version=4)
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(nb, km=km, timeout=600, resources={'metadata': {'path': str(ROOT)}},
                        allow_errors=False)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
errors = [o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
assert not errors
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
print('PASS: notebook executed through final cell; no errors.')

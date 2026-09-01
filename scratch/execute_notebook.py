import json
import sys
import io
import contextlib
from pathlib import Path

notebook_path = Path("work/notebooks/w05_model.ipynb")
nb = json.loads(notebook_path.read_text(encoding="utf-8"))

global_scope = {}
execution_count = 1

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source_code = "".join(cell["source"])
        print(f"Executing Code Cell {execution_count}...")
        
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                exec(source_code, global_scope)
            except Exception as e:
                print(f"Error in cell {execution_count}: {e}", file=sys.stderr)
                raise e
        
        stdout_str = stdout_buf.getvalue()
        stderr_str = stderr_buf.getvalue()
        
        outputs = []
        if stdout_str:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [s + "\n" for s in stdout_str.split("\n")[:-1]] + ([stdout_str.split("\n")[-1]] if stdout_str.split("\n")[-1] else [])
            })
        if stderr_str:
            outputs.append({
                "name": "stderr",
                "output_type": "stream",
                "text": [s + "\n" for s in stderr_str.split("\n")[:-1]] + ([stderr_str.split("\n")[-1]] if stderr_str.split("\n")[-1] else [])
            })
            
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
        execution_count += 1

notebook_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"\nSuccessfully executed all {execution_count - 1} code cells and saved {notebook_path}")

"""
scripts/execute_laptop_notebook.py
==================================
Executes the code cells of notebooks/Model_Comparison_Laptop.ipynb,
capturing real stdout and populating cell outputs directly in the notebook JSON,
ensuring mentor can see the actual rendered execution results.
"""

import os
import sys
import json
import io
import contextlib
import traceback

def execute_notebook(nb_path: str):
    print(f"[*] Executing {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    global_env = {"__name__": "__main__"}
    exec_count = 0

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        exec_count += 1
        cell["execution_count"] = exec_count
        code = "".join(cell["source"])

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        print(f"  -> Running cell {idx+1}/{len(nb['cells'])} (exec_count: {exec_count})...")
        cell_start = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, global_env)
        except Exception as e:
            traceback.print_exc(file=stderr_buf)
            print(f"     [!] Cell {idx+1} raised: {e}")

        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()

        outputs = []
        if stdout_val:
            lines = [l + "\n" for l in stdout_val.split("\n")]
            if lines and lines[-1] == "\n":
                lines = lines[:-1]
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": lines
            })
        if stderr_val:
            lines = [l + "\n" for l in stderr_val.split("\n")]
            if lines and lines[-1] == "\n":
                lines = lines[:-1]
            outputs.append({
                "name": "stderr",
                "output_type": "stream",
                "text": lines
            })

        cell["outputs"] = outputs

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"[+] Successfully executed and saved {nb_path} with {exec_count} executed code cells.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_nb = os.path.join(base_dir, "notebooks", "Model_Comparison_Laptop.ipynb")
    execute_notebook(target_nb)


import json
import ast
import os

def validate_all_notebooks():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notebooks = [
        "01_Laptop_One_Epoch_12_Model_Benchmark.ipynb",
        "02_NVIDIA_Lab_Extensive_Training.ipynb",
        "Model_Comparison_Laptop.ipynb",
        "Model_Comparison_NVIDIA.ipynb"
    ]
    for nb_name in notebooks:
        path = os.path.join(base_dir, "notebooks", nb_name)
        assert os.path.exists(path), f"Missing {path}"
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        cell_count = len(nb["cells"])
        print(f"[*] Validating {nb_name} (Total cells: {cell_count})")
        code_cells = 0
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                code_cells += 1
                code = "".join(cell["source"])
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    print(f"[!] Syntax error in {nb_name} cell {idx}: {e}")
                    raise
        print(f"[+] {nb_name}: PASS ({code_cells} code cells verified)")

if __name__ == "__main__":
    validate_all_notebooks()

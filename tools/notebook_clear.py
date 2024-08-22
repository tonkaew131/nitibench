import os
import nbformat as nbf
import argparse

def clear_notebook_outputs(notebook_path):
    """Clears the outputs of a Jupyter Notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=nbf.NO_CONVERT)

    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"Cleared outputs for: {notebook_path}")

def main():

    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".ipynb"):
                clear_notebook_outputs(os.path.join(root, file))

if __name__ == "__main__":
    main()
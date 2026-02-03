# Bioinformatics Project

A toolkit for processing and analyzing chemical compound data, including molecular descriptor calculation using **PaDEL‑Descriptor**.

## Overview

This project provides utilities for bioinformatics and cheminformatics workflows, focusing on the generation of molecular descriptors and fingerprints for small molecules. These descriptors are essential for **Quantitative Structure‑Activity Relationship (QSAR)** modeling, virtual screening, and other computational analyses.

## Jupyter Notebooks

- **`notebooks/analysis.ipynb`** – Demonstrates how to load the generated descriptor CSV, perform exploratory data analysis, and visualize key features.
- **`notebooks/modeling.ipynb`** – Shows a simple predictive‑modeling pipeline (e.g., regression or classification) using the descriptors as input features.
- **`notebooks/tutorial.ipynb`** – Step‑by‑step tutorial for newcomers, covering file formats, command‑line usage, and interpreting results.

> The notebooks are **self‑contained**; you can open them in any Jupyter environment (JupyterLab, VS Code, Google Colab, etc.) without installing additional Python packages beyond what is listed in `requirements.txt`.

## Data Files

| Type | Description | Typical Extension |
|------|-------------|-------------------|
| **Input** | Chemical structures provided by the user | `.smi` (SMILES) or `.sdf` (Structure‑Data File) |
| **Output** | CSV file containing all calculated descriptors and fingerprints | `descriptors_output.csv` |
| **Configuration** | XML file specifying fingerprint settings | `PaDEL-Descriptor/PubchemFingerprinter.xml` |

- **Input files** should be placed in the project root directory.
- **`descriptors_output.csv`** has one row per compound and one column per descriptor; values are numeric (or binary for fingerprint bits).

## How to Run the Analysis

1. **Prerequisites**  
   - **Java Runtime Environment (JRE) 8** or newer.  
   - The bundled **PaDEL‑Descriptor** binary (`PaDEL-Descriptor.jar`) is included in the repository.

2. **Prepare your data**  
   - Create a folder (e.g., `data/`) in the project root and place your `.smi` or `.sdf` files there.

3. **Generate descriptors**  
   ```bash
   ./padel.sh
   ```
   This script invokes PaDEL‑Descriptor with the following options:  
   - `-removesalt` – removes salt atoms  
   - `-standardizenitro` – standardizes nitro groups  
   - `-fingerprints` – computes PubChem fingerprints  
   - `-descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml` – uses the predefined descriptor set  
   - `-dir ./` – writes output to the current directory  
   - `-file descriptors_output.csv` – names the output file.

4. **Explore the results**  
   - Open `descriptors_output.csv` in a spreadsheet or load it into Python/R.  
   - Run the provided Jupyter notebooks to visualize the data and build predictive models.

5. **Optional – customizing descriptors**  
   - Edit `PaDEL-Descriptor/PubchemFingerprinter.xml` to add/remove descriptor types or adjust fingerprint parameters.  
   - Re‑run `./padel.sh` to generate a new descriptor set.

## Installation & Dependencies

- No Python packages are required to **run** the descriptor calculation; only JRE is needed.
- For **notebook usage**, install the Python environment:
  ```bash
  pip install -r requirements.txt
  ```
  (The `requirements.txt` lists packages such as `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `jupyter`.)

## License

- **Project code** – released under the **MIT License**.
- **PaDEL‑Descriptor binaries** – distributed under the **GNU Lesser General Public License v2.1** and the **GNU Affero General Public License v3**.  
  See the `PaDEL-Descriptor/license/` directory for the full license texts.

---

*For any questions or contributions, please open an issue or submit a pull request.*

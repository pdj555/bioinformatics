# Bioinformatics Project

A toolkit for processing and analyzing chemical compound data, including molecular descriptor calculation using PaDEL-Descriptor.

## Description

This project provides utilities for bioinformatics analysis, particularly focused on calculating molecular descriptors and fingerprints for chemical compounds. These descriptors are essential for quantitative structure-activity relationship (QSAR) modeling and other cheminformatics applications.

## Features

- **Molecular Descriptor Calculation**: Uses PaDEL-Descriptor to generate Pubchem fingerprints
- **Data Processing**: Handles SMILES/SDF file input and CSV output
- **Jupyter Notebooks**: Includes examples for data analysis and visualization (coming soon)

## Prerequisites

- Java Runtime Environment (JRE) 8 or higher
- PaDEL-Descriptor software (included)

## Usage

1. Place your chemical compound files (.smi or .sdf) in the project root
2. Run the descriptor calculator:
```bash
./padel.sh
```
3. Output will be generated as `descriptors_output.csv`

## Data Files

- Input: Chemical structures in SMILES or SDF format
- Output: CSV file containing molecular descriptors/fingerprints
  - Columns represent molecular features
  - Rows represent individual compounds

## License

PaDEL-Descriptor is distributed under:
- GNU Lesser General Public License 2.1
- GNU Affero General Public License 3

See full licenses in the PaDEL-Descriptor/license directory.

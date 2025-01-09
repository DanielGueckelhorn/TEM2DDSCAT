# TEM2DDSCAT

This repository contains a Python code designed to convert Transmission Electron Microscopy (TEM) images into DDSCAT-compatible geometries. Additionally, the code calculates basic statistics, such as particle size distribution, to facilitate quantitative analysis of particle geometries.

---

## Features

- **TEM to DDSCAT Conversion**: Automatically generate DDSCAT-compatible geometry files from TEM images.
- **Statistical Analysis**: Compute particle size distribution and other basic statistics.
- **Ease of Use**: Simple input-output workflow with well-documented functionality.

---

## Project Environment Setup

This project requires a specific Conda environment for dependencies. Follow the steps below to set up and activate the environment.

### Prerequisites

Make sure you have Conda installed. You can install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Setup Instructions

1. **Create the Conda environment:**

   Run the following command to create the environment from the `environment.yml` file:

   ```bash
   conda env create -f environment.yml

2. **Activate the Conda environment:**

   After the environment is created, activate it by running:

   ```bash
   conda activate TEM2DDSCAT

3. **Deactivating the Environment:**

   After the environment is created, activate it by running:

   ```bash
   conda deactivate
   
## Citation

If you use this code, please ensure you cite the [original work](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-22-37994&id=561240) that underpins this tool:

Daniel Gueckelhorn, Aaron Dove, Andreas Dörfler, and Andreas Ruediger, "Kernel-inspired algorithm to transform transmission electron microscopy images into discrete dipole approximation geometries," Opt. Express 32, 37994-38003 (2024)

---

## Development Status

This project is not under active development, and pull requests may not be reviewed promptly. However, updates may occasionally be released to fix existing bugs.

If you encounter issues, feel free to open an issue on the repository, but please note that response times may vary.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

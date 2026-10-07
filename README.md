# RecoilLab

![Tests](https://github.com/DMRecoilLab/actions/workflows/tests.yml/badge.svg

A modular Python toolkit for dark matter direct-detection calculations.

RecoilLab provides a transparent and extensible framework for computing
nuclear recoil spectra from Weakly Interacting Massive Particles (WIMPs),
with an emphasis on physical clarity, reproducibility, and education.

The project is designed both for exploratory research and for teaching
graduate-level dark matter phenomenology.

---

## Features

Current capabilities include:

- Standard Halo Model (SHM)
- Elastic and inelastic WIMP scattering
- Spin-independent (SI) interactions
- Helm nuclear form factors
- Differential and integrated recoil rates
- Detector exposure modelling
- Benchmark nuclei (Xe, Ar, Ge, Si)
- Plotting utilities
- Extensive unit-test suite with physics-motivated validation tests

The code is organized into independent modules describing:

- Halo models
- Scattering kinematics
- Nuclear properties
- Nuclear response functions
- Interaction models
- Detector models
- Event-rate calculations

This separation mirrors the physical structure of a direct-detection
calculation and makes it straightforward to extend individual components.

---

## Example

Compute the recoil spectrum for a xenon detector:

```python
import numpy as np
from src.nuclei.isotopes import Xe131
from src.halo import StandardHaloModel
from src.kinematics import ElasticKinematics
from src.interactions.si import SIInteraction
from src.interactions.particles import WIMP
from src.nuclei.form_factors import HelmFormFactor
from src.detectors.detectors import IdealDetector
from src.rates import RateCalculator

shm = StandardHaloModel(v0=220, vesc=544).boost(vboost=230)
wimp = WIMP(mass=100e9, spin=0)
target = Xe131

helm = HelmFormFactor(nucleus=target)
kinematics = ElasticKinematics(mx=wimp.mass, mT=target.mass)
si = SIInteraction(wimp=wimp, nucleus=target,
                   kinematics=kinematics,
                   form_factor=helm,
                   sigma_p=1e-48)

detector = IdealDetector(nucleus=target, mass_kg=1e3,
                         exposure_days=365)

rate_calc = RateCalculator(wimp=wimp, halo=shm,
                           interaction=si, detector=detector)

ER_eV = np.linspace(0, 1e5, 100)

dRdE = rate_calc.dRdE(ER_eV)
```

See the notebooks directory for complete examples.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/stefanodavini/DMRecoilLab.git
cd DMRecoilLab
```

Create the environment:

```bash
micromamba env create -f environment.yml
```

Activate it:

```bash
micromamba activate dmrecoillab
```

For developers: install pre-commit:
```bash
pre-commit install
```

Optional: install a Jupyter kernel

```bash
python -m ipykernel install --user --name dmrecoillab
```

---

## Running the notebooks

Launch JupyterLab:

```bash
jupyter lab
```

and select the `recoillab` kernel.

The notebooks reproduce benchmark calculations and provide examples of
typical workflows.

---

## Running the tests

Run the complete test suite:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_halo.py
```

Run a specific test:

```bash
pytest tests/test_halo.py -k test_shm_normalization
```

---

## Repository Structure

```text
src/
├── utils/
├── nuclei/
├── form_factors/
├── interactions/
├── detectors/
├── plotting/
├── halo.py
├── kinematics.py
└── rates.py

tests/

notebooks/
```

---

## Validation

The implementation is validated through:

- Unit tests of individual components
- Internal consistency checks
- Published benchmark comparisons
- Known analytical limits

Benchmark plots reproducing representative results from the direct-detection
literature are included in the repository.

---

## Intended Use

RecoilLab is primarily intended for:

- Teaching and learning dark matter phenomenology
- Prototyping new interaction models
- Reproducible recoil-spectrum calculations
- Exploratory studies of direct-detection experiments

It is not currently intended as a replacement for large-scale
collaboration software frameworks.

---

## Future Development

Planned extensions include:

- Additional halo models
- Additional nuclear response models
- Spin-dependent interactions
- EFT models
- Detector response functions
- Improved sensitivity and exclusion-curve calculations

---

## Citation

If you use this project in teaching material, publications,
or scientific work, please cite the repository.

Citation metadata is provided through `CITATION.cff`.

---

## License

This project is distributed under the BSD 3-Clause License.
See the LICENSE file for details.


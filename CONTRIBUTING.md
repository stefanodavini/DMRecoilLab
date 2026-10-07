# Contributing to RecoilLab

Thank you for your interest in contributing to RecoilLab.

RecoilLab is a modular Python toolkit for dark matter direct-detection
calculations, developed with an emphasis on physical transparency,
reproducibility, software quality, and education.

Contributions of all sizes are welcome, including:

- Bug reports
- Documentation improvements
- Feature requests
- New tests
- Physics validation studies
- Performance improvements
- New interactions, detectors, or halo models
- Updated parameters for benchmarking

---

## Reporting Issues

If you encounter a bug or unexpected behaviour, please open an Issue and include:

- A clear description of the problem
- A minimal reproducible example
- The expected behaviour
- The observed behaviour
- Relevant software versions (Python, NumPy, SciPy, etc.)

Where possible, include the full traceback.

---

## Suggesting New Features

Feature requests are welcome.

Before implementing substantial changes, please open an Issue describing:

- The proposed functionality
- The physics motivation
- The expected API

Discussion before implementation often helps keep the codebase
consistent and avoids duplicated effort.

---

## Submitting Pull Requests

Direct commits to the main branch are discouraged.
Contributions should be developed on dedicated branches and
submitted through Pull Requests.

1. Fork the repository.
2. Create a dedicated branch.
3. Implement your changes.
4. Add or update tests where appropriate.
5. Ensure the test suite passes.
6. Submit a Pull Request with a clear description of the changes.

Pull Requests should focus on a single coherent change whenever possible.

---

## Coding Guidelines

RecoilLab aims for code that is:

- Physically transparent
- Readable
- Well documented
- Extensively tested

Please:

- Follow the existing code style.
- Use descriptive names.
- Add docstrings for public APIs.
- Include units in documentation where relevant.
- Prefer clarity over premature optimization.

---

## Testing

Before submitting a Pull Request, run:

```bash
pytest
```

Contributions introducing new functionality should generally include corresponding unit tests.
Physics-motivated validation tests are particularly encouraged.

For physics-related contributions, reference calculations,
published benchmarks, or analytical limits should be provided
when possible.

## Documentation

User-facing features should be documented through:
- Docstrings
- Examples
- Notebook updates (when appropriate)
Good documentation is considered part of the contribution.

## Scientific Scope

RecoilLab currently focuses on dark matter direct-detection calculations.

Examples of contributions within scope include:
- Halo models
- Scattering kinematics
- Nuclear response functions
- Interaction models
- Detector models
- Rate calculations
- Sensitivity calculations
- Validation benchmarks

## Review Process

All contributions are reviewed by the project maintainers before being merged.
Constructive discussion and scientific scrutiny are encouraged.

## Code of Conduct

Please be respectful and professional when interacting through Issues, Pull Requests, and discussions.
Scientific disagreements are welcome; personal attacks are not.
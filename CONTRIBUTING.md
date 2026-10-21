# Contributing to DMRecoilLab

Thank you for your interest in contributing to DMRecoilLab.

DMRecoilLab is a scientific framework for dark matter direct-detection calculations, developed with a strong emphasis on:

- Physical transparency
- Scientific reproducibility
- Validation
- Software quality
- Long-term maintainability

Contributions of all sizes are welcome, including:

- Bug reports
- Documentation improvements
- New tests
- Validation studies
- Performance improvements
- New interaction models
- New detector implementations
- Benchmark calculations
- Developer tooling and infrastructure improvements

Before contributing, please familiarize yourself with the project's governance and validation policies.

---

# Guiding Principles

All contributions should respect the core principles of the project:

1. Physics before performance.
2. Transparency before convenience.
3. Validation before integration.
4. Reproducibility before optimization.
5. Modularity before duplication.
6. No undocumented physics.
7. Scientific trust above development speed.

These principles guide both development and review decisions.

---

# Types of Contributions

## Scientific Contributions

Examples include:

- New interaction models
- Nuclear response functions and form factors
- Improved detector models
- New halo velocity distributions

Scientific contributions must include appropriate documentation and validation evidence.

## Software Contributions

Examples include:

- Refactoring
- Infrastructure improvements
- Build system updates
- Performance improvements
- Continuous integration enhancements

Software contributions should preserve maintainability and API consistency.

## Documentation Contributions

Examples include:

- Tutorials
- Examples
- API documentation
- Validation notes
- Physics references

Documentation improvements are strongly encouraged and may be submitted independently of code changes.

---

# Reporting Issues

If you encounter a bug or unexpected behaviour, please open an Issue and include:

- A clear description of the problem
- A minimal reproducible example
- The expected behaviour
- The observed behaviour
- Relevant software versions
- Full traceback when available

Issues affecting scientific correctness should be identified explicitly.

---

# Proposing New Features

For substantial changes, contributors are encouraged to open an Issue or Discussion before implementation.

The proposal should briefly describe:

- Motivation
- Physics rationale
- Proposed implementation
- Affected modules
- Validation strategy

Early discussion often prevents duplicated effort and simplifies review.

---

# Protected Areas

Certain components of DMRecoilLab directly affect scientific outputs and therefore require enhanced scrutiny.

Protected areas include:
- Every abstract base class (ABCs) defining framework interfaces and extension points
- Event-rate calculations
- Kinematics calculations
- Interaction models implementations
- Velocity-distribution implementations
- Nuclear response implementations
- Public APIs
- Validation infrastructure

Contributors are encouraged to discuss major modifications to protected areas before implementation.

---

# Development Workflow

### 1. Create an Issue

Every contribution must be associated with an Issue.

Issues provide traceability, facilitate discussion, and document the rationale for changes.

The Issue should describe:

- The problem being addressed
- The proposed solution
- Any relevant scientific or technical context

Development should not begin before an Issue has been created.

## 2. Create a Development Branch

Do not develop directly on the main branch.
Create a branch for the issue and link it to a Pull Request.

Every contribution must maintain a traceable chain:
Issue → Branch → Pull Request → Merge

Pull Requests must reference the corresponding Issue.
Branches should be named using the associated Issue identifier whenever possible.
For example:
```
42-eft-operator-o8-implementation
57-fix-sd-normalization
```

Each Pull Request should address a single coherent topic.

## 3. Implement the Changes

When implementing new functionality:

- Follow existing project conventions.
- Keep modules focused and cohesive.
- Prefer explicitness over cleverness.
- Avoid hidden assumptions.
- Preserve backward compatibility whenever reasonable.

## 4. Submit a Pull Request

Before opening a Pull Request:

- Ensure all tests pass.
- Update documentation where required.
- Include validation material when applicable.
- Exclude unrelated changes.
- Verify that the contribution is focused and self-contained.
- Ensure the output of every notebook is cleared

The description of the Pull Request must report any deviations from the proposed solution in the linked issue.

---

# Coding Guidelines

DMRecoilLab values code that is:

- Physically transparent
- Readable
- Modular
- Well documented
- Scientifically traceable

Please:

- Use descriptive names.
- Add docstrings for public APIs.
- Document assumptions and approximations.
- Reference relevant literature where appropriate.
- Include units where relevant.
- Prefer clarity over premature optimization.

Code should be understandable by a future contributor without requiring detailed knowledge of the original implementation.

---

# Physics Traceability

One of the primary goals of DMRecoilLab is physical transparency.

Whenever possible, physics implementations should document:

- Original references
- Relevant equations
- Assumptions
- Numerical approximations
- Known limitations

Future contributors should be able to understand the scientific rationale without reverse-engineering the implementation.

---

# Testing Requirements

All contributions must preserve the integrity of the test suite.

Before submitting a Pull Request, run:

```bash
pytest
```

Contributions introducing new functionality should generally include:

- Unit tests
- Integration tests where appropriate
- Regression tests where appropriate

A contribution that cannot be tested is unlikely to be merged.

---

# Validation Requirements

Changes affecting physical outputs require validation.

Acceptable validation may include:

- Comparison with analytical limits
- Comparison with published results
- Comparison with benchmark calculations
- Comparison with previous validated releases

Validation material should accompany the Pull Request or be linked in supporting documentation.

For physics-related contributions, validation is considered part of the implementation.

---

# Documentation Requirements

User-facing features must be documented.

Documentation may include:

- Docstrings
- Tutorials
- Examples
- Validation notes
- Notebook updates

Code without documentation should be considered incomplete.

---

# Pull Request Review

All contributions undergo review before merging.

Review may evaluate:

- Scientific correctness
- Validation quality
- Architectural consistency
- Code quality
- Documentation completeness
- Test coverage

Review comments should be viewed as part of the collaborative scientific process rather than as criticism.

---

# Maintainers and Approval

Not all contributors possess merge authority.

Maintainers are responsible for:

- Reviewing contributions
- Protecting project standards
- Ensuring consistency across the framework

Changes affecting protected areas may additionally require approval from the Architecture Owner as defined in the Governance document.

---

# Scientific Scope

DMRecoilLab focuses on dark matter direct-detection calculations and supporting infrastructure.

Examples of in-scope contributions include:

- Halo models
- Interaction models
- Nuclear response functions
- Detector response models
- Recoil spectra
- Event-rate calculations
- Statistical inputs
- Validation benchmarks

Contributions outside the project's scientific scope may be declined even when technically correct.

---

# Code of Conduct

Please be respectful and professional in all project interactions.

Scientific disagreement is expected and encouraged.

Personal attacks, dismissive behaviour, or unprofessional conduct are not acceptable.

The goal of every discussion should be the improvement of both the science and the software.

---

# Final Note

DMRecoilLab aims to be a trusted scientific framework used in research and collaboration analyses.

Every contribution, regardless of size, should strengthen:

- Scientific correctness
- Transparency
- Reproducibility
- Maintainability

When uncertainty exists, contributors should favor solutions that improve long-term trust in the framework and its results.

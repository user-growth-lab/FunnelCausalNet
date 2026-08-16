# Environment

FunnelCausalNet v1.0.0 declares Python 3.9--3.11 support with bounded dependency ranges in `requirements.txt`. Those ranges are intended for portability and are not an exact environment lock.

Release validation was performed on macOS with Python 3.9.6 and the following installed versions:

| Package | Version |
|---|---:|
| NumPy | 2.0.2 |
| pandas | 2.3.3 |
| SciPy | 1.13.1 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.9.4 |
| PyTorch | 2.8.0 |

The validation covered Python syntax and imports, the no-data allocator smoke test, and the Criteo-backed model and conformal smoke tests described in `README.md`. Timing results are machine-dependent.

For archival runs, record the operating system, Python version, hardware, dataset checksum, source commit, and the output of `python -m pip freeze` alongside the generated results. The versions above document release validation; they should not be interpreted as evidence of the original manuscript environment.

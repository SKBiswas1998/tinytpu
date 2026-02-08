# Contributing to TinyTPU

Thanks for your interest in contributing to TinyTPU! This project aims to make edge AI accessible for robotics on resource-constrained devices.

## Getting Started

```bash
git clone https://github.com/SKBiswas1998/tinytpu.git
cd tinytpu
pip install -e ".[dev]"
pytest tests/ -v
```

## Development Workflow

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Run `pytest tests/ -v` — all 61+ tests must pass
5. Run `ruff check src/` — fix any lint issues
6. Submit a pull request

## Code Style

- Follow PEP 8, enforced by `ruff`
- Line length: 100 characters
- Type hints where practical
- Docstrings for public APIs

## Architecture

```
src/tinytpu/
├── hal/         # Hardware backends — add new accelerators here
├── inference/   # ONNX engine, model zoo
├── perception/  # Detection, tracking
├── control/     # Safety, pipeline, robot interface
├── monitoring/  # Thermal, memory, recording
├── numerical/   # Quantization methods
├── core/        # Systolic array simulation
└── cli/         # Command-line tools
```

## Adding a New Backend

1. Create a class inheriting from `InferenceBackend` in `hal/backends.py`
2. Implement `available()`, `load()`, `run()`, `supports()`
3. Set appropriate `priority` (higher = preferred)
4. Add to `ALL_BACKENDS` list
5. Add tests in `tests/test_hal_pipeline.py`

## Testing

```bash
pytest tests/ -v                    # All tests
pytest tests/test_package.py -v     # Package tests only
pytest tests/ -k "safety" -v       # Filter by name
pytest tests/ --cov=tinytpu        # With coverage
```

## Areas Where Help is Needed

- **Hardware testing**: Raspberry Pi, Hailo AI HAT+, Coral USB
- **Model conversion**: ONNX → HEF (Hailo), TFLite (Coral)
- **ROS2 integration**: Bridge to ROS2 nav stack
- **Dashboard**: FastAPI web UI for live monitoring
- **Documentation**: Tutorials, API docs, deployment guides

## Reporting Issues

- Include Python version, OS, and hardware info (`tinytpu version`, `tinytpu hardware`)
- For inference issues, include the model name and input shape
- For hardware issues, include accelerator model and driver version

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

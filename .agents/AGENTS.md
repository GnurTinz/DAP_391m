# Project Rules for Antigravity Agent

1. **Logging**: All configuration files (like Hydra `config.yaml`) or project scripts that generate run-level logs should direct those logs to the `tasks/` directory rather than the `logs/` directory. The `logs/` directory is primarily reserved for model checkpoints and `version_X` artifacts.

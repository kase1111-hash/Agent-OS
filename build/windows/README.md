# Windows Build Guide

This directory contains the build configuration for creating Windows distributable packages of Agent OS.

## Prerequisites

1. **Python 3.10+** - Required for building
2. **PyInstaller** - Creates standalone executables
3. **Project dependencies** - All runtime dependencies must be installed

### Install Build Dependencies

```bash
# Install PyInstaller and other build tools
pip install -r build/requirements-build.txt

# Install project dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Build Windows executable
python build/windows/build.py

# Build with single-file output
python build/windows/build.py --onefile

# Build portable ZIP package
python build/windows/build.py --mode portable

# Build everything
python build/windows/build.py --mode all --clean
```

## Build Modes

| Mode | Description | Output |
|------|-------------|--------|
| `exe` | Standalone executable (directory) | `dist/agent-os/agent-os.exe` |
| `portable` | ZIP package with all files | `dist/agent-os-X.X.X-windows-portable.zip` |
| `msi` | Windows installer (requires WiX) | `dist/agent-os-X.X.X.msi` |
| `all` | All of the above | Multiple outputs |

## Build Options

| Option | Description |
|--------|-------------|
| `--mode MODE` | Build mode: exe, portable, msi, all |
| `--onefile` | Create single-file executable (slower startup) |
| `--clean` | Clean previous build artifacts |
| `--no-upx` | Disable UPX compression |
| `--version X.X.X` | Override version string |
| `--verbose` | Enable verbose output |

## Output Structure

### Directory Mode (default)
```
dist/
└── agent-os/
    ├── agent-os.exe          # Main executable
    ├── python3XX.dll         # Python runtime
    ├── src/                  # Application modules
    └── config/               # Configuration files
```

### Single-File Mode (`--onefile`)
```
dist/
└── agent-os.exe              # Self-contained executable
```

## Customization

### Adding an Icon

Place an `.ico` file at `build/windows/agent-os.ico` before building:

```bash
# The build will automatically use it
python build/windows/build.py
```

### Modifying the Spec File

Edit `build/windows/agent-os.spec` to:
- Add/remove hidden imports
- Include additional data files
- Change executable options
- Enable single-file mode (uncomment section at bottom)

## Troubleshooting

### Missing Modules

If you get "ModuleNotFoundError" at runtime, add the module to `hiddenimports` in `agent-os.spec`:

```python
hiddenimports=[
    # ... existing imports ...
    'your.missing.module',
],
```

### Large Executable Size

To reduce size:
1. Enable UPX compression (default)
2. Use `--onefile` mode
3. Add unused packages to `excludes` in the spec file

### Code Signing

For production releases, sign the executable:

```powershell
# Using signtool (Windows SDK)
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com dist\agent-os\agent-os.exe
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Build Windows

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r build/requirements-build.txt

      - name: Build
        run: python build/windows/build.py --mode all --clean

      - uses: actions/upload-artifact@v4
        with:
          name: windows-build
          path: dist/
```

## Files in This Directory

| File | Purpose |
|------|---------|
| `build.py` | Main build script |
| `agent-os.spec` | PyInstaller specification file |
| `version_info.txt` | Windows version metadata (auto-generated) |
| `agent-os.ico` | Application icon (optional) |
| `README.md` | This documentation |

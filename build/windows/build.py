#!/usr/bin/env python3
"""
Windows Build Script for Agent OS

This script automates the creation of Windows distributable packages:
- Standalone executable (PyInstaller)
- MSI installer (optional, requires WiX Toolset)
- Portable ZIP package

Usage:
    python build/windows/build.py [options]

Options:
    --mode        Build mode: exe, msi, portable, all (default: exe)
    --onefile     Create single-file executable (slower startup)
    --clean       Clean build artifacts before building
    --no-upx      Disable UPX compression
    --sign        Sign the executable (requires certificate)
    --version     Override version string
    --output      Output directory (default: dist/)

Examples:
    python build/windows/build.py
    python build/windows/build.py --mode portable --clean
    python build/windows/build.py --mode all --onefile
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BUILD_DIR = PROJECT_ROOT / 'build'
DIST_DIR = PROJECT_ROOT / 'dist'
SPEC_FILE = SCRIPT_DIR / 'agent-os.spec'


class BuildError(Exception):
    """Build process error."""
    pass


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=check
        )
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.debug(f"  {line}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.stderr}")
        raise BuildError(f"Command failed: {' '.join(cmd)}")


def check_requirements() -> bool:
    """Check that all build requirements are met."""
    logger.info("Checking build requirements...")

    requirements_met = True

    # Check Python version
    if sys.version_info < (3, 10):
        logger.error("Python 3.10+ is required")
        requirements_met = False
    else:
        logger.info(f"  Python {sys.version_info.major}.{sys.version_info.minor} OK")

    # Check PyInstaller
    try:
        import PyInstaller
        logger.info(f"  PyInstaller {PyInstaller.__version__} OK")
    except ImportError:
        logger.error("PyInstaller not installed. Run: pip install pyinstaller")
        requirements_met = False

    # Check project dependencies
    try:
        import pydantic
        import fastapi
        import uvicorn
        logger.info("  Project dependencies OK")
    except ImportError as e:
        logger.error(f"Missing dependency: {e.name}")
        requirements_met = False

    return requirements_met


def clean_build() -> None:
    """Clean previous build artifacts."""
    logger.info("Cleaning build artifacts...")

    dirs_to_clean = [
        PROJECT_ROOT / 'dist',
        PROJECT_ROOT / 'build' / 'pyinstaller',
        PROJECT_ROOT / '__pycache__',
    ]

    for dir_path in dirs_to_clean:
        if dir_path.exists():
            logger.info(f"  Removing {dir_path}")
            shutil.rmtree(dir_path)

    # Clean .pyc files
    for pyc in PROJECT_ROOT.rglob('*.pyc'):
        pyc.unlink()

    # Clean __pycache__ directories
    for pycache in PROJECT_ROOT.rglob('__pycache__'):
        if pycache.is_dir():
            shutil.rmtree(pycache)


def get_version() -> str:
    """Get the current version from the package."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src import __version__
        return __version__
    except ImportError:
        return "0.1.0"


def create_version_info(version: str) -> Path:
    """Create Windows version info file for the executable."""
    version_parts = version.split('.')
    while len(version_parts) < 4:
        version_parts.append('0')

    version_tuple = ', '.join(version_parts[:4])
    version_string = version

    content = f'''# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({version_tuple}),
    prodvers=({version_tuple}),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [
            StringStruct(u'CompanyName', u'Agent OS Community'),
            StringStruct(u'FileDescription', u'Agent OS - Constitutional AI Framework'),
            StringStruct(u'FileVersion', u'{version_string}'),
            StringStruct(u'InternalName', u'agent-os'),
            StringStruct(u'LegalCopyright', u'Copyright (c) {datetime.now().year} Agent OS Community'),
            StringStruct(u'OriginalFilename', u'agent-os.exe'),
            StringStruct(u'ProductName', u'Agent OS'),
            StringStruct(u'ProductVersion', u'{version_string}'),
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''

    version_file = SCRIPT_DIR / 'version_info.txt'
    version_file.write_text(content)
    logger.info(f"  Created version info: {version_file}")
    return version_file


def build_exe(onefile: bool = False, upx: bool = True) -> Path:
    """Build the Windows executable using PyInstaller."""
    logger.info("Building Windows executable...")

    # Ensure dist directory exists
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--clean',
        '--noconfirm',
        '--workpath', str(BUILD_DIR / 'pyinstaller'),
        '--distpath', str(DIST_DIR),
    ]

    if onefile:
        cmd.extend(['--onefile'])
        logger.info("  Building single-file executable...")

    if not upx:
        cmd.extend(['--noupx'])

    cmd.append(str(SPEC_FILE))

    run_command(cmd)

    # Determine output path
    if onefile:
        exe_path = DIST_DIR / 'agent-os.exe'
    else:
        exe_path = DIST_DIR / 'agent-os' / 'agent-os.exe'

    if exe_path.exists():
        logger.info(f"  Executable created: {exe_path}")
        return exe_path
    else:
        raise BuildError("Executable not found after build")


def build_portable(exe_path: Path, version: str) -> Path:
    """Create a portable ZIP package."""
    logger.info("Creating portable ZIP package...")

    # Determine the directory to zip
    if exe_path.name == 'agent-os.exe' and exe_path.parent.name == 'agent-os':
        source_dir = exe_path.parent
    else:
        source_dir = exe_path.parent

    zip_name = f'agent-os-{version}-windows-portable.zip'
    zip_path = DIST_DIR / zip_name

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir.parent)
                zf.write(file_path, arcname)
                logger.debug(f"  Added: {arcname}")

    logger.info(f"  Portable package created: {zip_path}")
    return zip_path


def build_msi(exe_path: Path, version: str) -> Optional[Path]:
    """Create MSI installer using WiX Toolset (if available)."""
    logger.info("Creating MSI installer...")

    # Check for WiX Toolset
    try:
        run_command(['candle', '-?'], check=False)
    except (FileNotFoundError, BuildError):
        logger.warning("  WiX Toolset not found - skipping MSI creation")
        logger.warning("  Install WiX Toolset from: https://wixtoolset.org/")
        return None

    # TODO: Implement WiX-based MSI creation
    logger.warning("  MSI creation not yet implemented")
    return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Build Agent OS for Windows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--mode',
        choices=['exe', 'portable', 'msi', 'all'],
        default='exe',
        help='Build mode (default: exe)'
    )

    parser.add_argument(
        '--onefile',
        action='store_true',
        help='Create single-file executable'
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean build artifacts before building'
    )

    parser.add_argument(
        '--no-upx',
        action='store_true',
        help='Disable UPX compression'
    )

    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Override version string'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║              Agent OS Windows Build System                   ║
╚══════════════════════════════════════════════════════════════╝
""")

    try:
        # Check requirements
        if not check_requirements():
            logger.error("Build requirements not met")
            return 1

        # Get version
        version = args.version or get_version()
        logger.info(f"Building version: {version}")

        # Create version info
        create_version_info(version)

        # Clean if requested
        if args.clean:
            clean_build()

        # Build executable
        if args.mode in ('exe', 'portable', 'all'):
            exe_path = build_exe(
                onefile=args.onefile,
                upx=not args.no_upx
            )

        # Create portable package
        if args.mode in ('portable', 'all'):
            build_portable(exe_path, version)

        # Create MSI installer
        if args.mode in ('msi', 'all'):
            build_msi(exe_path, version)

        # Success
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    Build Successful!                         ║
╚══════════════════════════════════════════════════════════════╝
""")
        logger.info(f"Output directory: {DIST_DIR}")

        # List outputs
        logger.info("Build outputs:")
        for item in DIST_DIR.iterdir():
            size = item.stat().st_size if item.is_file() else sum(
                f.stat().st_size for f in item.rglob('*') if f.is_file()
            )
            size_mb = size / (1024 * 1024)
            logger.info(f"  {item.name}: {size_mb:.1f} MB")

        return 0

    except BuildError as e:
        logger.error(f"Build failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Build cancelled by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

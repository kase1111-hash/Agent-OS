# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Agent OS Windows build.

This creates a standalone Windows executable that includes:
- All Python dependencies
- Web UI templates and static files
- Default constitution files
- Configuration templates

Usage:
    pyinstaller build/windows/agent-os.spec

Output:
    dist/agent-os/agent-os.exe (directory mode)
    dist/agent-os.exe (onefile mode - see below)
"""

import os
import sys
from pathlib import Path

# Get the project root directory
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
PROJECT_ROOT = os.path.abspath(os.path.join(SPEC_DIR, '..', '..'))

# Add project root to path for imports
sys.path.insert(0, PROJECT_ROOT)

block_cipher = None


def collect_data_files():
    """Collect data files, only including directories that exist."""
    datas = []

    # Web static files
    static_dir = os.path.join(PROJECT_ROOT, 'src', 'web', 'static')
    if os.path.exists(static_dir):
        datas.append((static_dir, 'src/web/static'))

    # Web templates (if they exist)
    templates_dir = os.path.join(PROJECT_ROOT, 'src', 'web', 'templates')
    if os.path.exists(templates_dir):
        datas.append((templates_dir, 'src/web/templates'))

    # Config directory (if it exists)
    config_dir = os.path.join(PROJECT_ROOT, 'config')
    if os.path.exists(config_dir):
        datas.append((config_dir, 'config'))

    return datas


# Collect all source files
a = Analysis(
    [os.path.join(PROJECT_ROOT, 'src', 'web', 'app.py')],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=collect_data_files(),
    hiddenimports=[
        # Core modules
        'src',
        'src.core',
        'src.core.constitution',
        'src.core.parser',
        'src.core.validator',
        'src.core.models',
        # Agent modules
        'src.agents',
        'src.agents.interface',
        'src.agents.config',
        'src.agents.loader',
        'src.agents.whisper',
        'src.agents.smith',
        'src.agents.seshat',
        'src.agents.sage',
        'src.agents.quill',
        'src.agents.muse',
        # Kernel modules
        'src.kernel',
        'src.kernel.engine',
        'src.kernel.policy',
        # Memory modules
        'src.memory',
        'src.memory.vault',
        'src.memory.storage',
        # Messaging
        'src.messaging',
        'src.messaging.bus',
        # Web modules
        'src.web',
        'src.web.app',
        'src.web.routes',
        'src.web.routes.chat',
        'src.web.routes.agents',
        'src.web.routes.constitution',
        'src.web.routes.memory',
        'src.web.routes.system',
        # Tools
        'src.tools',
        'src.tools.registry',
        'src.tools.executor',
        # Boundary
        'src.boundary',
        'src.boundary.boundary_daemon',
        # Contracts
        'src.contracts',
        # Ceremony
        'src.ceremony',
        # Installer
        'src.installer',
        # SDK
        'src.sdk',
        # Dependencies
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'starlette',
        'pydantic',
        'pydantic_core',
        'yaml',
        'jinja2',
        'httpx',
        'websockets',
        'watchdog',
        'watchdog.observers',
        'watchdog.events',
        # Windows-specific
        'winreg',
        'ctypes',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude dev/test dependencies
        'pytest',
        'pytest_cov',
        'mypy',
        'black',
        'isort',
        'ipython',
        # Exclude unused backends
        'tkinter',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Directory mode (faster startup, easier debugging)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='agent-os',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False for GUI-only mode
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(PROJECT_ROOT, 'build', 'windows', 'agent-os.ico') if os.path.exists(os.path.join(PROJECT_ROOT, 'build', 'windows', 'agent-os.ico')) else None,
    version=os.path.join(PROJECT_ROOT, 'build', 'windows', 'version_info.txt') if os.path.exists(os.path.join(PROJECT_ROOT, 'build', 'windows', 'version_info.txt')) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='agent-os',
)


# =============================================================================
# ONEFILE BUILD (Alternative - uncomment to use)
# =============================================================================
# Creates a single executable file. Slower startup but easier to distribute.
# To use: Comment out the EXE and COLLECT above, uncomment below.
# =============================================================================

# exe_onefile = EXE(
#     pyz,
#     a.scripts,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     [],
#     name='agent-os',
#     debug=False,
#     bootloader_ignore_signals=False,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     runtime_tmpdir=None,
#     console=True,
#     disable_windowed_traceback=False,
#     argv_emulation=False,
#     target_arch=None,
#     codesign_identity=None,
#     entitlements_file=None,
#     icon=os.path.join(PROJECT_ROOT, 'build', 'windows', 'agent-os.ico') if os.path.exists(os.path.join(PROJECT_ROOT, 'build', 'windows', 'agent-os.ico')) else None,
# )

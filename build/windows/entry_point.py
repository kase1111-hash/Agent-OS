#!/usr/bin/env python3
"""
Entry point for PyInstaller-built Agent OS executable.

This script properly sets up the Python path and imports to work
correctly when bundled as a standalone executable.
"""

import os
import sys


def setup_path():
    """Set up the Python path for the bundled application."""
    # When running as a PyInstaller bundle, _MEIPASS contains the temp directory
    # where PyInstaller extracts the bundled files
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        bundle_dir = sys._MEIPASS
    else:
        # Running as script (development)
        bundle_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to project root
        bundle_dir = os.path.dirname(os.path.dirname(bundle_dir))

    # Add the bundle directory to the path so imports work
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

    return bundle_dir


def main():
    """Main entry point for the Agent OS executable."""
    bundle_dir = setup_path()

    # Now we can import using absolute imports
    from src.web.config import get_config
    from src.web.app import create_app, run_server

    import argparse

    parser = argparse.ArgumentParser(
        description='Agent OS - Constitutional AI Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--host',
        default=None,
        help='Host to bind to (default: 127.0.0.1)',
    )

    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Port to bind to (default: 8080)',
    )

    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version and exit',
    )

    args = parser.parse_args()

    if args.version:
        try:
            from src import __version__
            print(f"Agent OS v{__version__}")
        except ImportError:
            print("Agent OS v0.1.0")
        return 0

    # Print startup banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║                        Agent OS                              ║
║              Constitutional AI Framework                     ║
╚══════════════════════════════════════════════════════════════╝
""")

    config = get_config()
    host = args.host or config.host
    port = args.port or config.port

    print(f"Starting web server at http://{host}:{port}")
    print("Press Ctrl+C to stop\n")

    try:
        import uvicorn

        # Create app directly instead of using factory string
        # (factory string doesn't work well with PyInstaller)
        app = create_app()

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

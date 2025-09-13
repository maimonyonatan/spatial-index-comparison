#!/usr/bin/env python3
"""
Main entry point for the ZM R-Tree Research system.

This script provides a simple interface to run the research prototype
for comparing R-Tree vs Learned ZM Index performance.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from zm_rtree_research.cli import main as cli_main
from zm_rtree_research.gui.app import main as gui_main


def main():
    """Main entry point with mode selection."""
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        # Remove --gui from sys.argv so it doesn't interfere with Streamlit
        sys.argv.pop(1)
        print("🚀 Starting Streamlit GUI...")
        gui_main()
    else:
        # Run CLI interface
        cli_main()


if __name__ == "__main__":
    main()

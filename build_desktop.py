"""
Build script for packaging TuneOS desktop application.
Run this script to compile the frontend and package the desktop app with PyInstaller.
"""
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


def print_step(msg: str):
    print(f"\n[+] {msg}")


def run_cmd(cmd: list[str], env: dict = None):
    print(f"    Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def main():
    root_dir = Path(__file__).parent.resolve()
    os.chdir(root_dir)

    print("========================================")
    print("  Building TuneOS Desktop App")
    print("========================================")

    # 1. Clean previous builds
    print_step("Cleaning previous builds...")
    for d in ['build', 'dist']:
        if os.path.exists(d):
            shutil.rmtree(d)

    # 2. Export Reflex frontend
    print_step("Exporting Reflex frontend...")
    # reflex export creates the static bundle we need
    # Make sure we don't start the server
    env = os.environ.copy()
    run_cmd([sys.executable, "-m", "reflex", "export", "--no-zip"], env=env)

    # Make sure the export was successful
    # Usually reflex export creates a .web directory
    if not (root_dir / ".web").exists():
        print("    Warning: .web directory not found, PyInstaller might fail.")

    # 3. Run PyInstaller
    print_step("Running PyInstaller...")
    run_cmd(["pyinstaller", "--noconfirm", "tuneos.spec"])

    # 4. Final instructions
    print_step("Build complete!")
    
    system = platform.system()
    if system == "Darwin":
        print("    App is ready at: dist/TuneOS.app")
        print("    You can launch it with: open dist/TuneOS.app")
    elif system == "Windows":
        print("    App is ready at: dist\\TuneOS\\TuneOS.exe")
    else:
        print("    App is ready at: dist/TuneOS/TuneOS")


if __name__ == "__main__":
    main()

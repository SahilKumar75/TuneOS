# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# We only package the desktop UI shell, NOT the ML backend dependencies.
# The ML backend (torch, transformers, etc) runs inside Docker.
hiddenimports = [
    'desktop',
    'app',
]

datas = [
    ('.web', '.web'),
    ('rxconfig.py', '.'),
]

# Create the Analysis block
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'transformers', 'accelerate', 'datasets', 'trl', 'peft', 'bitsandbytes'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TuneOS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # True for debugging, False for release
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TuneOS',
)

# Build Mac .app bundle if on macOS
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='TuneOS.app',
        icon=None,
        bundle_identifier='com.tuneos.desktop',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSBackgroundOnly': 'False',
        }
    )

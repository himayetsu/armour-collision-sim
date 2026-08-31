# -*- mode: python ; coding: utf-8 -*-
# Build: python -m PyInstaller --noconfirm --clean ArmorSim.spec

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hidden = [], [], []
for pkg in ('pyqtgraph', 'OpenGL'):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hidden += pkg_hidden

hidden += collect_submodules('pyqtgraph.opengl')
hidden += [
    'PyQt5.sip',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.QtOpenGL',
    'OpenGL.platform.win32',
    'OpenGL.arrays.numpymodule',
    'scipy.spatial',
    'scipy.spatial._ckdtree',
    'scipy.spatial.ckdtree',
]

a = Analysis(
    ['armorsim_qt.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pygame', 'tkinter', 'matplotlib', 'IPython', 'jupyter',
        'torch', 'tensorflow', 'tensorboard', 'keras',
        'pandas', 'sklearn', 'cv2', 'PIL',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ArmorSim',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# decision_intelligence.spec
# PyInstaller build specification for AI Decision Intelligence Analyzer
# Run: pyinstaller decision_intelligence.spec

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# ── Project root ─────────────────────────────────────────────────────
ROOT = Path(SPEC).parent   # SPEC is set automatically by PyInstaller

# ── Collect hidden imports (scikit-learn has many C-extension sub-modules) ──
hidden = []
hidden += collect_submodules("sklearn")
hidden += collect_submodules("pandas")
hidden += collect_submodules("matplotlib")
hidden += collect_submodules("seaborn")
hidden += collect_submodules("flask")
hidden += collect_submodules("flask_cors")
hidden += collect_submodules("numpy")
hidden += [
    "pkg_resources",
    "packaging",
    "PIL",
    "chardet",
    "charset_normalizer",
]

# ── Data files to bundle ──────────────────────────────────────────────
extra_datas = [
    # (source_path,       dest_folder_inside_bundle)
    (str(ROOT / "frontend"),  "frontend"),
    (str(ROOT / "data"),      "data"),
]

# Also collect matplotlib/mpl-data (fonts, styles, etc.)
extra_datas += collect_data_files("matplotlib")
extra_datas += collect_data_files("seaborn")

# ── Analysis ──────────────────────────────────────────────────────────
a = Analysis(
    [str(ROOT / "exe_entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=extra_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
              "wx", "gi", "cv2", "tornado"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DecisionIntelligenceAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,           # keep console so user can see loading progress
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DecisionIntelligenceAnalyzer",
)

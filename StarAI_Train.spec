from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


project_root = Path(SPECPATH)
asset_extensions = {".json", ".mp3", ".ogg", ".png", ".pth", ".wav"}
datas = [
    (str(path), str(path.parent.relative_to(project_root)))
    for path in (project_root / "src").rglob("*")
    if path.is_file() and path.suffix.lower() in asset_extensions
]

# Ship and ability implementations are selected from JSON and imported with
# importlib at runtime, so static analysis cannot discover them reliably.
hiddenimports = collect_submodules("src.Objects.Ships") + [
    "src.training.process_worker",
]

a = Analysis(
    [str(project_root / "src" / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Pygame can optionally use NumPy, but StarAI does not use those APIs. Do
    # not let unrelated packages in the development environment inflate builds.
    excludes=["numpy"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StarAI_Train",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(project_root / "StarAI.ico")],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="StarAI_Train",
)

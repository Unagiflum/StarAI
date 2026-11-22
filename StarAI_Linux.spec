from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


project_root = Path(SPECPATH)
# .ttf and .txt carry the bundled monospaced font and its licence.
asset_extensions = {".json", ".mp3", ".ogg", ".png", ".pth", ".ttf", ".txt", ".wav"}
datas = [
    (str(path), str(path.parent.relative_to(project_root)))
    for path in (project_root / "src").rglob("*")
    if path.is_file() and path.suffix.lower() in asset_extensions
]

# Ship and ability implementations are selected from JSON and imported with
# importlib at runtime, so static analysis cannot discover them reliably.
hiddenimports = collect_submodules("src.Objects.Ships") + [
    "src.training.process_session",
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
    # Linux builds bundle the CPU-only PyTorch wheel from .venv-CPU-Linux, but
    # do not need the companion media packages.
    excludes=["torchvision", "torchaudio"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# UPX is disabled: compressing PyTorch's shared libraries on Linux saves little
# and can corrupt them. Linux executables do not embed an icon.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StarAI_Linux",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="StarAI_Linux",
)

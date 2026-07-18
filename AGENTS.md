# Repository workflow

## Verification

- Do not run PyInstaller, full application builds, archive creation, or packaged
  smoke tests during routine implementation and test iteration.
- Do not add tests that invoke `build.cmd`, `buildcpu.cmd`, `buildgpu.cmd`,
  `build.ps1`, or PyInstaller. The normal suite must remain source-level and
  avoid packaging read/write churn.
- Full builds and packaged smoke tests run on a separate packaging schedule.
  Run them only when the user explicitly requests packaging verification.
- Static tests that inspect build configuration without producing build
  artifacts are permitted.

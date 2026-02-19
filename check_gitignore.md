# Locate Un-ignored Dataset

The `.gitignore` file currently ignores:
- `data/datasets/`
- `data/models/`

I have confirmed that the project's data root is `d:\Github\ORAM-defect-detection\data\datasets`.

If there is still a dataset not being ignored, it might be in a different location.

Please run this command in your terminal to see which files are being tracked that should be ignored:

```powershell
git status
```

If you see a large folder or dataset files listed, please let me know the path, and I will add it to `.gitignore`.

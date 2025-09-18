# Conda Development Workflow

This guide explains how to develop sirf-viewer using conda packaging.

## Files Created

1. `conda-recipe/meta.yaml` - Conda package definition
2. `1_initial_conda_build.sh` - First-time setup and build
3. `2_conda_rebuild.sh` - Quick rebuild for development iterations

## Setup (First Time)

```bash
# Make scripts executable
chmod +x 1_initial_conda_build.sh 2_conda_rebuild.sh

# Run initial build
./1_initial_conda_build.sh
```

This will:

- Install conda-build (and boa/mamba if available)
- Build your package
- Install it locally
- Set everything up for development

## Development Workflow

After making code changes:

```bash
# Quick rebuild and reinstall
./2_conda_rebuild.sh
```

This will:

- Remove old version
- Build new version (with `--no-test` for speed)
- Install updated version

## Testing Your Changes

```bash
# Test import
python -c "import sirf_viewer; print('Success!')"

# Test GUI
sirf-viewer

# Test in Python
python -c "from sirf_viewer import SIRFViewer; print('Import works!')"
```

## Mamba vs Conda

The scripts automatically detect and prefer mamba if available:

- **mamba** = Much faster builds and installs
- **conda** = Fallback if mamba not available

To install mamba:

```bash
conda install mamba -c conda-forge
```

## Version Updates

To change version, edit `conda-recipe/meta.yaml`:

```yaml
package:
  name: sirf-viewer
  version: "0.2.0"  # Change this
```

Then rebuild.

## Troubleshooting

**YAML errors:** Check indentation in `meta.yaml`
**Build fails:** Try `conda build purge-all` then rebuild
**Import fails:** Check if package is installed with `conda list sirf-viewer`

## Clean Everything

```bash
# Remove package
conda remove sirf-viewer -y

# Clean build cache
conda build purge-all

# Start fresh
./1_initial_conda_build.sh
```

#!/bin/bash
set -e

echo "=== SIRF Viewer Initial Conda Build ==="

# Check if mamba is available, otherwise use conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Using mamba for faster builds..."
else
    CONDA_CMD="conda"
    echo "Using conda..."
fi

# Check if conda-build is installed
echo "Checking for conda-build..."
if ! $CONDA_CMD list conda-build &> /dev/null; then
    echo "Installing conda-build..."
    $CONDA_CMD install -y conda-build
fi

# Check if boa is available for faster builds (works with mamba)
if [[ "$CONDA_CMD" == "mamba" ]] && ! $CONDA_CMD list boa &> /dev/null; then
    echo "Installing boa for faster builds with mamba..."
    $CONDA_CMD install -y boa
fi

# Clean any previous builds
echo "Cleaning previous builds..."
$CONDA_CMD build purge-all -q || true

# Build the package
echo "Building sirf-viewer package..."
if [[ "$CONDA_CMD" == "mamba" ]] && command -v conda-mambabuild &> /dev/null; then
    conda-mambabuild conda-recipe
else
    $CONDA_CMD build conda-recipe
fi

# Get the path to the built package
if [[ "$CONDA_CMD" == "mamba" ]] && command -v conda-mambabuild &> /dev/null; then
    PACKAGE_PATH=$(conda-mambabuild conda-recipe --output)
else
    PACKAGE_PATH=$($CONDA_CMD build conda-recipe --output)
fi

echo "Package built at: $PACKAGE_PATH"

# Install the package locally
echo "Installing sirf-viewer locally..."
$CONDA_CMD install --use-local sirf-viewer -y

echo ""
echo "✅ Initial build complete!"
echo "✅ Package installed locally"
echo ""
echo "You can now:"
echo "  - Import with: python -c 'import sirf_viewer'"
echo "  - Run GUI with: sirf-viewer"
echo "  - For rebuilds, use: ./2_conda_rebuild.sh"
echo ""
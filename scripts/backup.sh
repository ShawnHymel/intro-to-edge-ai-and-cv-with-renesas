#!/bin/bash

# Usage: backup_project.sh <source_project_dir> <destination_dir>
# Example: backup_project.sh blinky/ ~/my-repo/projects/01-blinky/

set -e  # Exit on error

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_project_dir> <destination_dir>"
    echo "Example: $0 blinky/ ~/my-repo/projects/01-blinky/"
    exit 1
fi

SOURCE="$1"
DEST="$2"

# Remove trailing slashes for consistency
SOURCE="${SOURCE%/}"
DEST="${DEST%/}"

# Check if source exists
if [ ! -d "$SOURCE" ]; then
    echo "Error: Source directory '$SOURCE' does not exist"
    exit 1
fi

# Create destination directory
echo "Creating destination directory: $DEST"
mkdir -p "$DEST"

# Essential files and directories to copy
ESSENTIAL_FILES=(
    "configuration.xml"
    ".project"
    ".cproject"
)

ESSENTIAL_DIRS=(
    "src"
    "ra_cfg"
    "ra_gen"
    "script"
)

# Copy essential files
echo "Copying essential files..."
for file in "${ESSENTIAL_FILES[@]}"; do
    if [ -f "$SOURCE/$file" ]; then
        echo "  ✓ $file"
        cp "$SOURCE/$file" "$DEST/"
    else
        echo "  ⚠ Warning: $file not found (might be okay)"
    fi
done

# Copy essential directories
echo "Copying essential directories..."
for dir in "${ESSENTIAL_DIRS[@]}"; do
    if [ -d "$SOURCE/$dir" ]; then
        echo "  ✓ $dir/"
        cp -r "$SOURCE/$dir" "$DEST/"
    else
        echo "  ⚠ Warning: $dir/ not found"
    fi
done

# Optional: Copy README if it exists
if [ -f "$SOURCE/README.md" ]; then
    echo "  ✓ README.md"
    cp "$SOURCE/README.md" "$DEST/"
fi

echo ""
echo "✅ Project backup complete!"
echo "Source: $SOURCE"
echo "Destination: $DEST"
echo ""
echo "Files NOT copied (will be regenerated or are build artifacts):"
echo "  - ra/ (FSP library - regenerated from configuration.xml)"
echo "  - Debug/ (build artifacts)"
echo "  - Release/ (build artifacts)"
echo "  - .settings/ (IDE settings)"

#!/bin/bash
set -euo pipefail

: "${protein_name:?protein_name not set}"
: "${ligand_name:?ligand_name not set}"

SOURCE_DIR="./result/${protein_name}_${ligand_name}_structure_raw"
DEST_DIR="./result/${protein_name}_${ligand_name}_structure_processed"

mkdir -p "$DEST_DIR"

find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 \
| while IFS= read -r -d '' d; do
  base="$(basename "$d")"
  candidate="$d/${base}_model.cif"
  if [[ -f "$candidate" ]]; then
    cp -v -- "$candidate" "$DEST_DIR/${base}_model.cif"
  else
    echo "[SKIP] not found: $candidate"
  fi
done

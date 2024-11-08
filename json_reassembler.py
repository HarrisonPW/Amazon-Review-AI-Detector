import json
import time
from pathlib import Path
import sys


def reassemble_json_file(chunks_dir: str, output_file: str = None):
    chunks_path = Path(chunks_dir)
    manifest_path = chunks_path / 'manifest.json'

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found in {chunks_dir}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    if output_file is None:
        output_file = Path(manifest['original_file']).name
        output_file = f"reassembled_{output_file}"
    all_data = []
    for chunk_file in manifest['chunk_files']:
        chunk_path = chunks_path / chunk_file
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file {chunk_file} not found")

        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
            all_data.extend(chunk_data)

    if len(all_data) != manifest['total_items']:
        print(f"Warning: Reassembled data has {len(all_data)} items, "
              f"but original had {manifest['total_items']} items",
              file=sys.stderr)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully reassembled file: {output_file}")
    print(f"Total items: {len(all_data)}")
    return output_file


# Reassemble the file
print("\nReassembling JSON file...")
chunks_dir = "json_chunks"
start_time = time.time()
output_file = reassemble_json_file(
    chunks_dir=chunks_dir,
    output_file="Cell_Phones_and_Accessories.json"
)
reassemble_time = time.time() - start_time
print(f"Reassembly completed in {reassemble_time:.2f} seconds")

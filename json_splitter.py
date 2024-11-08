import json
from pathlib import Path
import math
import time


def split_json_file(input_file: str, chunk_size: int = 1024 * 1024, output_dir: str = 'chunks'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    total_size = len(json.dumps(data).encode('utf-8'))
    num_chunks = math.ceil(total_size / chunk_size)
    items_per_chunk = math.ceil(len(data) / num_chunks)
    for i in range(0, len(data), items_per_chunk):
        chunk = data[i:i + items_per_chunk]
        chunk_file = Path(output_dir) / f"chunk_{i // items_per_chunk:03d}.json"

        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

        chunk_size = Path(chunk_file).stat().st_size
        print(f"Created {chunk_file.name} - Size: {chunk_size / 1024:.2f} KB")

    manifest = {
        'original_file': input_file,
        'total_chunks': num_chunks,
        'total_items': len(data),
        'items_per_chunk': items_per_chunk,
        'chunk_files': [f"chunk_{i:03d}.json" for i in range(num_chunks)]
    }

    with open(Path(output_dir) / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    input_file = "Cell_Phones_and_Accessories.json"
    chunks_dir = "json_chunks"

    print("Splitting JSON file...")
    start_time = time.time()
    manifest = split_json_file(
        input_file=input_file,
        chunk_size=50 * 1024 * 1024,  # 50MB chunks
        output_dir=chunks_dir
    )
    split_time = time.time() - start_time
    print(f"Split completed in {split_time:.2f} seconds")


if __name__ == "__main__":
    main()

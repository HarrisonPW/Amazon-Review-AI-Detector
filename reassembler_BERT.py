import os

def reassemble_file(output_path, part_files):
    try:
        with open(output_path, 'wb') as output_file:
            for part_file in sorted(part_files):  # Sort to ensure correct order (part0, part1)
                print(f"Processing: {part_file}")
                with open(part_file, 'rb') as pf:
                    output_file.write(pf.read())
        print(f"Successfully created: {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

part_files1 = [f for f in os.listdir()
              if f.startswith('BERT.safetensors.part')]


part_files2 = [f for f in os.listdir()
              if f.startswith('BERTLSTM.safetensors.part')]



if part_files1:
    print(f"Found {len(part_files1)} parts: {part_files1}")
    reassemble_file('BERT.safetensors', part_files1)
else:
    print("No part files found!")


if part_files2:
    print(f"Found {len(part_files2)} parts: {part_files2}")
    reassemble_file('BERTLSTM.safetensors', part_files2)
else:
    print("No part files found!")

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

part_files = [f for f in os.listdir()
              if f.startswith('gpt2_spam_detector.pth.part')]

if part_files:
    print(f"Found {len(part_files)} parts: {part_files}")
    reassemble_file('gpt2_spam_detector.pth', part_files)
else:
    print("No part files found!")

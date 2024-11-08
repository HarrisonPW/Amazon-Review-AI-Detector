import os

def split_file(file_path, chunk_size=100*1024*1024):
    file_number = 0
    with open(file_path, 'rb') as f:
        chunk = f.read(chunk_size)
        while chunk:
            with open(f"{file_path}.part{file_number}", 'wb') as chunk_file:
                chunk_file.write(chunk)
            file_number += 1
            chunk = f.read(chunk_size)

split_file('gpt2_spam_detector.pth')

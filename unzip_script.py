
import zipfile
import os

zip_path = "signLanguage/004/2.Validation/combined_keypoints.zip"
extract_path = "signLanguage/004/2.Validation/extracted_keypoints"

if not os.path.exists(zip_path):
    print(f"Error: {zip_path} not found.")
    exit(1)

print(f"Extracting {zip_path} into {extract_path}...")
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(" extraction complete!")
except zipfile.BadZipFile:
    print("Error: Bad Zip File. The combination might have failed.")
except Exception as e:
    print(f"Error: {e}")

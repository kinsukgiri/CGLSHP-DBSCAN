import fitz
import os

pdf_path = "Jain_ECR_Visualization.pdf"

output_folder = "Extracted_Images"
os.makedirs(output_folder, exist_ok=True)

pdf = fitz.open(pdf_path)

for i, page in enumerate(pdf):
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    pix.save(f"{output_folder}/ECR_Image_{i+1}.png")

pdf.close()

print("All PDF pages extracted successfully.")

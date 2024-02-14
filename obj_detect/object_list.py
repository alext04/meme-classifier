

import re

# The path to your file
file_path = 'objdetect_without_captions.txt'

# Regular expression to match the desired part of each line
pattern = re.compile(r":\s*\d+x\d+\s*(.*),")

# List to store the extracted data
extracted_data_list = []

# Open the file and process each line
with open(file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            extracted_data = match.group(1).strip()
            extracted_data_list.append(extracted_data)

# Print or use the extracted data as needed
for data in extracted_data_list:
    print(data)


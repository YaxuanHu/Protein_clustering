import os
import csv
from Bio import SeqIO

# Function to read the first 10 sequences from a FASTA file and write them to a CSV file
def read_fasta_and_convert_first_10_to_csv(fasta_file, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['ID', 'Sequence'])
        
         # Read the FASTA file and extract ID and sequence
        with open(fasta_file, 'r') as file:
            for record in SeqIO.parse(file, 'fasta'):
                # Extract the ID, splitting by '|' and taking the second part
                fasta_id = record.id.split('|')[1]
                sequence = str(record.seq)
                # Write to CSV
                csvwriter.writerow([fasta_id, sequence])


# Function to process all FASTA files in one directory and save CSVs in another directory
def process_fasta_files_in_directory(fasta_directory, csv_directory):
    for filename in os.listdir(fasta_directory):
        if filename.endswith(".fasta"):  # Check if the file is a FASTA file
            fasta_file = os.path.join(fasta_directory, filename)
            # Generate CSV file name and save in the other directory
            csv_file = os.path.join(csv_directory, filename.replace('.fasta', '.csv'))
            # Convert the FASTA file to CSV
            read_fasta_and_convert_first_10_to_csv(fasta_file, csv_file)
            print(f"Processed {fasta_file} and saved as {csv_file}")

# Example usage
fasta_directory = './data/uniprotkb/'  # Replace with the path to your FASTA directory
csv_directory = './data/csv/'      # Replace with the path to your CSV output directory

# Ensure the CSV directory exists
if not os.path.exists(csv_directory):
    os.makedirs(csv_directory)

process_fasta_files_in_directory(fasta_directory, csv_directory)

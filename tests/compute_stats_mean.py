import pandas as pd
import glob

# Define the path where your CSV files are located
path = 'stats/flocking/'  # Update this path if needed
file_pattern = '*.csv'
files = glob.glob(path + file_pattern)

if not files:
    raise FileNotFoundError("No CSV files found in the specified directory.")

# Initialize a list to store DataFrames
dfs = []

# Read and store each CSV file into the list
for file in files:
    try:
        data = pd.read_csv(file)
        print(f"Loaded {file} with {len(data)} rows.")
        dfs.append(data)
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not dfs:
    raise ValueError("No valid CSV files could be read.")

# Merge all DataFrames on the 'Episode' column
all_data = dfs[0]
for df in dfs[1:]:
    all_data = pd.merge(all_data, df, on='Episode', suffixes=('', '_dup'))

# Drop duplicate 'Episode' columns if they exist
all_data = all_data.loc[:, ~all_data.columns.duplicated()]

# Ensure the DataFrame is not empty and has the correct structure
if all_data.empty or 'Episode' not in all_data.columns:
    raise ValueError("Merged data is empty or does not contain 'Episode' column.")

# Calculate the mean for each episode (excluding 'Episode' column itself)
all_data['Mean'] = all_data.drop(columns='Episode').mean(axis=1)

# Select only the 'Episode' and 'Mean' columns for the output
output_data = all_data[['Episode', 'Mean']]

# Save the result to a new CSV file
output_file = 'output.csv'
output_data.to_csv(output_file, index=False)

print(f"Results have been written to {output_file}")

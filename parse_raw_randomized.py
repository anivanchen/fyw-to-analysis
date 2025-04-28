import pandas as pd
import glob
import os
import random

data_directory = './data/'
output_filename = 'filtered_college_data_randomized.csv'

column_list = [
    'INSTNM', 'ADMCON7', 'PCTPELL', 'UGDS', 'UGDS_BLACK', 'UGDS_HISP',
    'ADM_RATE', 'ADM_RATE_ALL', 'SAT_AVG', 'SAT_AVG_ALL', 'RET_FT4',
    'WDRAW_ORIG_YR2_RT', 'LO_INC_WDRAW_ORIG_YR2_RT',
    'MD_INC_WDRAW_ORIG_YR2_RT', 'HI_INC_WDRAW_ORIG_YR2_RT',
    'PELL_WDRAW_ORIG_YR2_RT', 'NOPELL_WDRAW_ORIG_YR2_RT',
    'WDRAW_ORIG_YR4_RT', 'LO_INC_WDRAW_ORIG_YR4_RT',
    'MD_INC_WDRAW_ORIG_YR4_RT', 'HI_INC_WDRAW_ORIG_YR4_RT',
    'PELL_WDRAW_ORIG_YR4_RT', 'NOPELL_WDRAW_ORIG_YR4_RT',
    'C100_4', 'C150_4', 'C150_4_PELL',
    'C150_4_NOLOANNOPELL', 'C150_4_LOANNOPELL',
    'COSTT4_A', 'PCTFLOAN', 'DEBT_MDN', 'LOCALE', 'FEMALE',
    'UGDS_WHITE', 'UGDS_ASIAN', 'UGDS_AIAN', 'UGDS_NHPI', 'UGDS_2MOR',
    'UNITID', # Include UNITID to identify institutions uniquely across years
    'OPEID', # Include OPEID for identification
    'ICLEVEL', 'CONTROL', # Add some key institutional controls
    'YEAR'
]

csv_files = sorted(glob.glob(os.path.join(data_directory, '*.csv')), reverse=True)

if not csv_files:
    print(f"No CSV files found in the directory: {data_directory}")
    college_list = []
else:
    first_csv = csv_files[0]
    print(f"Selecting colleges from file: {first_csv}")
    
    try:
        df = pd.read_csv(first_csv, low_memory=False)
        
        # Calculate data completeness for each institution
        column_subset = [col for col in column_list if col in df.columns and col != 'YEAR']
        
        df['data_completeness'] = df[column_subset].notnull().mean(axis=1)
        
        complete_colleges = df[df['data_completeness'] > 0.75]['INSTNM'].unique()
        
        sample_size = min(1500, len(complete_colleges))
        college_list = random.sample(list(complete_colleges), sample_size)
        
        print(f"Selected {len(college_list)} colleges with >75% data completeness")
        
    except Exception as e:
        print(f"Error processing first CSV: {e}")
        college_list = []

all_data = []

if not csv_files:
    print(f"No CSV files found in the directory: {data_directory}")
    print("Please check the 'data_directory' variable in the script.")
else:
    print(f"Found {len(csv_files)} CSV files to process.")
    for file_path in csv_files:
        print(f"Processing file: {file_path}")
        try:
            df = pd.read_csv(file_path, low_memory=False)

            # Extract the year from the filename (format like 'MERGEDYYYY_YY_PP.csv')
            filename = os.path.basename(file_path)
            year_match = filename.split('_')[0].replace('MERGED', '')
            df['YEAR'] = year_match if year_match.isdigit() else filename

            filtered_df = df[df['INSTNM'].isin(college_list)].copy()

            present_columns = [col for col in column_list if col in filtered_df.columns]
            
            missing_columns = [col for col in column_list if col not in filtered_df.columns]
            if missing_columns:
                print(f"  Missing columns in this file: {missing_columns}")
            else:
                print("  All requested columns are present in this file.")
            
            selected_df = filtered_df[present_columns]

            all_data.append(selected_df)

            print(f"  Extracted {len(present_columns)} cols for specified colleges.")
            print(f"  Extracted {len(selected_df)} rows for specified colleges.\n")

        except FileNotFoundError:
            print(f"  Error: File not found at {file_path}")
        except pd.errors.EmptyDataError:
            print(f"  Warning: File {file_path} is empty. Skipping.")
        except Exception as e:
            print(f"  An error occurred while processing {file_path}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_filename, index=False)

        print(f"\nSuccessfully extracted data for {len(combined_df)} rows across all files.")
        print(f"Filtered data saved to: {output_filename}")
    else:
        print("\nNo data was extracted. Check the data directory and college names.")


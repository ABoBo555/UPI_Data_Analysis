import pandas as pd
import numpy as np
import sys

# --- Configuration ---
# !!! IMPORTANT: Replace with the correct path to YOUR Excel file !!!
excel_file_path = 'Paytm_data.xlsx'
sheet_name = 'Passbook Payment History' # Use None to read the first sheet, or specify 'Sheet1', 'Transactions', etc.
comment_to_remove = "This is not included in total paid and received calculations." # Exact string to filter out

# --- Step 0: Read Data ---
print(f"--- Reading data from: {excel_file_path} ---")
try:
    # Read all columns as strings initially to preserve original values before filtering/parsing
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name, dtype=str)
    print(f"--- Successfully read. Original DataFrame shape: {df.shape} ---")
except FileNotFoundError:
    print(f"Error: File not found at '{excel_file_path}'. Please check the path.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading Excel file: {e}")
    sys.exit(1)

# --- Step 1: Initial Cleanup (Before Filtering) ---
print("\n--- Applying Initial Cleanup (Before Filtering) ---")

# 1a. Trim Whitespace (Crucial for accurate comparisons)
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
print("Trimmed whitespace from all cells.")

# 1b. Rename Columns
rename_map = {
    'Transaction Details': 'Recipent/Sender Info',
    'Your Account': 'Payment Method'
}
columns_renamed = []
for old_name, new_name in rename_map.items():
    if old_name in df.columns:
        df.rename(columns={old_name: new_name}, inplace=True)
        columns_renamed.append(f"'{old_name}' to '{new_name}'")
if columns_renamed:
    print(f"Renamed columns: {', '.join(columns_renamed)}.")
else:
    print("Did not find 'Transaction Details' or 'Your Account' columns to rename.")

# 1c. Drop Non-Essential Columns (Keep 'Comment' for filtering!)
cols_to_drop_initial = ['UPI Ref No.', 'Order ID', 'Remarks', 'Tags']
existing_cols_to_drop = [col for col in cols_to_drop_initial if col in df.columns]
if existing_cols_to_drop:
    df.drop(columns=existing_cols_to_drop, inplace=True)
    print(f"Dropped initial non-essential columns: {existing_cols_to_drop}.")
else:
    print("Did not find 'UPI Ref No.', 'Order ID', 'Remarks', 'Tags' columns to drop.")

print(f"DataFrame shape after initial cleanup: {df.shape}")
# print("Columns after initial cleanup:", df.columns.tolist()) # Debugging

# --- Step 2: Filter Rows based on 'Comment' ---
print("\n--- Filtering Rows Based on 'Comment' ---")
if 'Comment' in df.columns:
    initial_rows = len(df)
    # Ensure comment column is string and handle potential None/NaN before comparison
    df['Comment'] = df['Comment'].astype(str).fillna('')

    # Create mask for rows to KEEP (comment DOES NOT match the string to remove)
    rows_to_keep_mask = (df['Comment'] != comment_to_remove)
    num_rows_to_remove = (~rows_to_keep_mask).sum()

    print(f"   Found {num_rows_to_remove} rows matching the comment to remove.")

    # Apply the filter
    df = df[rows_to_keep_mask].copy() # Use .copy() to ensure it's a separate DataFrame
    rows_removed = initial_rows - len(df)

    print(f"Step 5 logic: Removed {rows_removed} rows.")
    print(f"   DataFrame shape after filtering: {df.shape}")

    # --- Step 3: Drop 'Comment' Column (AFTER Filtering) ---
    if 'Comment' in df.columns: # Check it still exists (it should)
        df.drop(columns=['Comment'], inplace=True)
        print("Step 6 logic: Dropped 'Comment' column successfully.")
    else:
        # This case should ideally not happen if filtering was based on it
        print("   Warning: 'Comment' column was not found after filtering, couldn't drop it.")

else:
    print("Warning: Column 'Comment' not found in the original data. Skipping filtering and comment column drop.")
    # The DataFrame 'df' continues unchanged in this case

# Check if DataFrame is empty after filtering
if df.empty:
    print("Error: DataFrame is empty after filtering. No data left to process.")
    # Optionally exit or handle as needed
    # sys.exit("Exiting script as DataFrame is empty.")


# --- Step 4: Apply Transformations to the Filtered Data ---
print("\n--- Applying Transformations to Remaining Data ---")

# 4a. Add 'Title' Column
df['Title'] = 'Paytm'
print("Added 'Title' column.")

# 4b. Process 'Amount' and create 'Status'
if 'Amount' in df.columns:
    print("Processing 'Amount' and creating 'Status'...")
    original_amount_str = df['Amount'].astype(str).copy() # Keep original for fallback
    # Clean and convert to numeric
    df['Numeric_Amount'] = df['Amount'].str.replace(r'[+,"]', '', regex=True).str.strip()
    df['Numeric_Amount'] = pd.to_numeric(df['Numeric_Amount'], errors='coerce')
    failed_amount_mask = df['Numeric_Amount'].isna()
    if failed_amount_mask.any(): print(f"   Warning: {failed_amount_mask.sum()} 'Amount' values could not be converted to numeric.")

    # Create 'Status'
    df['Status'] = 'Unknown' # Default status
    df.loc[df['Numeric_Amount'] >= 0, 'Status'] = 'Received'
    df.loc[df['Numeric_Amount'] < 0, 'Status'] = 'Paid'
    print("   Created 'Status' column.")

    # Format 'Amount' column string
    df.loc[~failed_amount_mask, 'Amount'] = df.loc[~failed_amount_mask, 'Numeric_Amount'].abs().map('₹{:.2f}'.format)
    df.loc[failed_amount_mask, 'Amount'] = original_amount_str[failed_amount_mask] # Use original if conversion failed
    print("   Formatted 'Amount' column with ₹ symbol (kept original for errors).")

    # Drop temporary numeric column
    if 'Numeric_Amount' in df.columns: df.drop(columns=['Numeric_Amount'], inplace=True)
else:
    print("Warning: 'Amount' column not found. Skipping Amount/Status processing.")

# 4c. Format 'Date' Column
if 'Date' in df.columns:
    print("Formatting 'Date' column...")
    original_dates = df['Date'].astype(str).copy()
    
    def format_date(date_str):
        try:
            # First try parsing as DD/MM/YYYY
            if '/' in date_str:
                parts = date_str.split('/')
                if len(parts) == 3:
                    day, month, year = parts
                    # Convert to datetime using explicit format
                    dt = pd.to_datetime(f"{year}-{month}-{day}")
                    return dt.strftime('%b %d %Y')
            
            # If not DD/MM/YYYY format, try parsing as is
            dt = pd.to_datetime(date_str)
            return dt.strftime('%b %d %Y')
        except:
            return date_str  # Return original if parsing fails
    
    # Apply the custom formatting function
    df['Date'] = df['Date'].apply(format_date)
    print("   Formatted 'Date' column successfully.")
else:
    print("Warning: 'Date' column not found. Skipping Date formatting.")

# 4d. Format 'Time' Column
if 'Time' in df.columns:
    print("Formatting 'Time' column...")
    original_times = df['Time'].astype(str).copy()
    try:
        # Attempt conversion to datetime objects to leverage formatting
        datetime_times = pd.to_datetime(original_times, errors='coerce')
        failed_mask = datetime_times.isna()

        # Format successfully converted times
        if (~failed_mask).any():
             formatted_times = datetime_times[~failed_mask].dt.strftime('%I:%M:%S %p')
             # Remove leading zero from hour (e.g., 03 PM -> 3 PM)
             formatted_times = formatted_times.apply(lambda x: x[1:] if isinstance(x, str) and x.startswith('0') else x)
             df.loc[~failed_mask, 'Time'] = formatted_times
        else:
            print("   Warning: No times were successfully parsed for formatting.")

        # Keep original string for failed times
        df.loc[failed_mask, 'Time'] = original_times[failed_mask]
        print("   Formatted 'Time' column (kept original for errors).")
    except Exception as e:
         print(f"   Warning: Error during 'Time' formatting: {e}. Keeping original values.")
         df['Time'] = original_times # Revert to original if unexpected error
else:
    print("Warning: 'Time' column not found. Skipping Time formatting.")


# --- Step 5: Final Column Reordering ---
print("\n--- Finalizing Column Order ---")
final_columns_order_requested = [
    'Title', 'Status', 'Amount', 'Recipent/Sender Info',
    'Payment Method', 'Date', 'Time'
]

# Check which requested columns actually exist now
available_columns = df.columns.tolist()
final_columns_order_existing = [col for col in final_columns_order_requested if col in available_columns]

print(f"   Columns available before reordering: {available_columns}")
print(f"   Final columns selected in order: {final_columns_order_existing}")

# Create the final DataFrame with only the desired columns in the specified order
try:
    df_cleaned = df[final_columns_order_existing].copy()
    print("Step 12 logic: Reordered columns to final structure.")
except KeyError as e:
    print(f"Error reordering columns - missing expected column: {e}. Outputting available columns.")
    df_cleaned = df.copy() # Fallback to show all remaining columns


# --- Final Output ---
print("\n--- Cleaned DataFrame head ---")
if not df_cleaned.empty:
    # Make sure pandas display settings show enough content
    pd.set_option('display.max_rows', 10) # Show more rows if needed
    pd.set_option('display.max_columns', None) # Ensure all columns are shown
    pd.set_option('display.width', 1000) # Wider display
    print(df_cleaned.head())
else:
    print("Cleaned DataFrame is empty.")

print("\n--- Sample rows from cleaned data ---")
if not df_cleaned.empty:
    sample_size = min(10, len(df_cleaned))
    if sample_size > 0:
        print(df_cleaned.sample(sample_size))
else:
    print("Cleaned DataFrame is empty, cannot show sample.")

print("\n--- Cleaned DataFrame info ---")
df_cleaned.info()


# --- Optional: Save the cleaned DataFrame ---
output_filename_csv = 'paytm.csv'
if not df_cleaned.empty:
    try:
        df_cleaned.to_csv(output_filename_csv, index=False)
        print(f"\nCleaned data saved to CSV: '{output_filename_csv}'")
    except Exception as e:
        print(f"\nError saving cleaned data to CSV: {e}")
else:
    print("\nSkipping save because the cleaned DataFrame is empty.")
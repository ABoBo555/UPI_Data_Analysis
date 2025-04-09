# Import necessary libraries
import pandas as pd
from bs4 import BeautifulSoup
import re # Import regular expressions library
import os # To check if file exists

# --- Configuration ---
# CHANGE THIS TO YOUR ACTUAL FILENAME (including spaces if any)
html_file_path = 'My Activity.html' # Or the correct path to your file
excel_filename = 'scraped_data.xlsx'
csv_filename = 'scraped_data.csv'

# --- Step 1: Read HTML Content from File ---
html_content = None
if not os.path.exists(html_file_path):
    print(f"Error: HTML file not found at '{html_file_path}'")
    exit() # Stop the script if the file doesn't exist

try:
    # Try common encodings if utf-8 fails
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodings_to_try:
        try:
            with open(html_file_path, 'r', encoding=enc) as f:
                html_content = f.read()
            print(f"Successfully read HTML content from '{html_file_path}' using {enc} encoding.")
            break # Stop trying encodings if successful
        except UnicodeDecodeError:
            # print(f"Failed to decode using {enc}...") # Optional: uncomment for debugging encoding issues
            continue # Try next encoding
    if html_content is None:
         # Provide more context if reading failed
         raise Exception(f"Could not decode the file '{html_file_path}' with any of the attempted encodings: {encodings_to_try}")

except Exception as e:
    print(f"Error reading HTML file: {e}")
    exit() # Stop if reading fails

if not html_content:
    print("HTML content is empty. Exiting.")
    exit()

# --- Step 2: Parse the HTML ---
# Using 'lxml' is generally faster if installed, otherwise 'html.parser'
try:
    soup = BeautifulSoup(html_content, 'lxml')
    print("Using lxml parser.")
except ImportError: # Correctly catch ImportError if lxml is not installed
    print("lxml parser not found, using html.parser (might be slower).")
    soup = BeautifulSoup(html_content, 'html.parser')
except Exception as e:
     print(f"Error initializing BeautifulSoup: {e}")
     exit()


# --- Step 3: Find all transaction blocks ---
transaction_blocks = soup.find_all('div', class_='outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp')
if not transaction_blocks:
    print("Warning: No transaction blocks found with class 'outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp'. Check the HTML structure and class names.")

# --- Step 4: Extract Data Points ---
extracted_data = []

for block_index, block in enumerate(transaction_blocks): # Added index for better logging
    # Initialize dictionary to store data for this transaction
    data = {
        'Title': None,
        'Status': None,
        'Amount': None,
        'Recipient/Sender Info': None,
        'Payment Method': None,
        'Date': None,
        'Time': None
    }

    # 1. Extract Title
    title_tag = block.find('p', class_='mdl-typography--title')
    if title_tag:
        data['Title'] = title_tag.get_text(strip=True)
    else:
        print(f"Warning [Block {block_index}]: Could not find title tag.")


    # Find the *inner* grid first, then the specific content cell within it
    inner_grid = block.find('div', class_='mdl-grid')
    if not inner_grid:
        print(f"Warning [Block {block_index}]: Could not find inner 'mdl-grid'.")
        extracted_data.append(data) # Append partially filled data or empty data
        continue # Skip to the next block

    # Find the main content cell containing the transaction details
    content_cell = inner_grid.find('div', class_='content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1')

    if content_cell:
        # Get text nodes separated by <br>, stripping whitespace
        lines = [text for text in content_cell.stripped_strings]

        if len(lines) >= 1: # Need at least the main line
            main_line = lines[0]

            # --- Revised Main Line Parsing (2-Step Approach) ---
            # Step A: Extract Status and Amount (required) and the rest of the line
            # CORRECTED regex for amount part (Group 2)
            match_base = re.match(
                r'^(Paid|Sent|Received)\s+'             # Group 1: Status
                r'(₹[\d,]+\.\d{2,})\s*'                 # Group 2: Amount (allows digits and commas freely before '.') - CORRECTED
                r'(.*)$',                               # Group 3: The rest of the line
                main_line,
                re.IGNORECASE | re.DOTALL
            )

            if match_base:
                data['Status'] = match_base.group(1).capitalize()
                data['Amount'] = match_base.group(2)
                remainder = match_base.group(3).strip()

                # Step B: Parse the remainder for optional 'Recipient/Sender' and 'Payment Method'
                recipient_info = None
                payment_method = None

                # Look for 'using' part first
                # Use re.IGNORECASE for 'using'
                using_match = re.search(r'\susing\s+(.*)$', remainder, re.IGNORECASE)
                if using_match:
                    payment_method = f"using {using_match.group(1).strip()}"
                    # The part before ' using ' is potentially the recipient info
                    potential_recipient = remainder[:using_match.start()].strip()
                    # Check if it actually looks like recipient info (starts with to/from)
                    if potential_recipient.lower().startswith('to ') or potential_recipient.lower().startswith('from '):
                         # Validate prefix matches status (e.g., don't assign 'from X' if status is Paid)
                         prefix_lower = potential_recipient[:potential_recipient.find(' ')+1].lower() # "to " or "from "
                         status_lower = data['Status'].lower()
                         if (status_lower in ['paid', 'sent'] and prefix_lower == 'to ') or \
                            (status_lower == 'received' and prefix_lower == 'from '):
                             recipient_info = potential_recipient
                         else:
                              print(f"Warning [Block {block_index}]: Recipient prefix '{prefix_lower.strip()}' mismatch with status '{status_lower}' in '{remainder}'. Recipient not assigned.")

                    elif potential_recipient: # It exists but doesn't start with to/from - Could be recipient name directly?
                        # We should add the 'to'/'from' prefix based on status
                        prefix = 'to' if data['Status'] in ['Paid', 'Sent'] else 'from'
                        recipient_info = f"{prefix} {potential_recipient}"

                else:
                    # No 'using' found. The entire remainder might be recipient info
                    potential_recipient = remainder
                    if potential_recipient.lower().startswith('to ') or potential_recipient.lower().startswith('from '):
                         # Validate prefix matches status
                         prefix_lower = potential_recipient[:potential_recipient.find(' ')+1].lower() # "to " or "from "
                         status_lower = data['Status'].lower()
                         if (status_lower in ['paid', 'sent'] and prefix_lower == 'to ') or \
                            (status_lower == 'received' and prefix_lower == 'from '):
                             recipient_info = potential_recipient
                         else:
                             print(f"Warning [Block {block_index}]: Recipient prefix '{prefix_lower.strip()}' mismatch with status '{status_lower}' in '{remainder}'. Recipient not assigned.")
                    elif potential_recipient: # Remainder exists but no prefix and no 'using' - e.g. "Received ₹5.00 Google"
                         prefix = 'to' if data['Status'] in ['Paid', 'Sent'] else 'from'
                         recipient_info = f"{prefix} {potential_recipient}"


                data['Recipient/Sender Info'] = recipient_info
                data['Payment Method'] = payment_method

            else:
                 # This warning should now only appear if Status/Amount format is completely unexpected
                 print(f"Warning [Block {block_index}]: Could not parse base transaction (Status/Amount) from: '{main_line}'")


            # --- Date/Time Parsing (Using comma splitting) ---
            if len(lines) >= 2:
                date_time_line = lines[1]
                # Example: Apr 9, 2025, 9:22:11 PM GMT+05:30
                parts = date_time_line.split(',')
                if len(parts) >= 3:
                    # Combine Month Day, Year
                    data['Date'] = f"{parts[0].strip()} {parts[1].strip()}"

                    # Extract time from the third part using regex
                    time_part_str = parts[2].strip()
                    # Regex looks for HH:MM:SS followed optionally by space/special char and AM/PM
                    time_match = re.search(r'(\d{1,2}:\d{2}:\d{2}(\s*|\u202f)[AP]M)', time_part_str)
                    if time_match:
                         # Clean up the matched time string
                         time_str = time_match.group(1)
                         # Replace known unusual spaces and trim
                         time_str = time_str.replace('\u202f', ' ').replace(' ', ' ').strip()
                         # Collapse multiple spaces potentially introduced
                         time_str = re.sub(r'\s+', ' ', time_str)
                         data['Time'] = time_str
                    else:
                        print(f"Warning [Block {block_index}]: Could not parse time pattern (HH:MM:SS AM/PM) from: '{time_part_str}'")
                else:
                     # Handle cases where the split doesn't yield enough parts
                     print(f"Warning [Block {block_index}]: Could not parse date/time line (expected 3+ parts after comma split): '{date_time_line}'")
            else:
                # Only one line found in content cell (main transaction only)
                 print(f"Info [Block {block_index}]: Only one line found in content cell, no date/time line present for: '{main_line}'")

        else:
             print(f"Warning [Block {block_index}]: Content cell is empty or contains no text strings: {content_cell}")

    else:
        # This message indicates the primary div holding the transaction text wasn't found
        print(f"Warning [Block {block_index}]: Could not find the primary content cell within the inner grid.")

    # Add the extracted data dictionary to our list
    extracted_data.append(data)

# --- Step 5: Create Pandas DataFrame ---
if not extracted_data:
     print("\nNo data was extracted. Cannot create DataFrame.")
     df = pd.DataFrame() # Create empty DataFrame
else:
    df = pd.DataFrame(extracted_data)
    # Reorder columns for consistency
    df = df[['Title', 'Status', 'Amount', 'Recipient/Sender Info', 'Payment Method', 'Date', 'Time']]


# --- Step 6: Save Files ---
if not df.empty:
    print(f"\nAttempting to save {len(df)} rows of data.")
    # Save to Excel (.xlsx)
    try:
        # Ensure openpyxl is installed: pip install openpyxl
        df.to_excel(excel_filename, index=False, engine='openpyxl')
        print(f"Data successfully saved to {excel_filename}")
    except ImportError:
        print(f"\nCould not save to Excel ('{excel_filename}'). Please install 'openpyxl': pip install openpyxl")
    except Exception as e:
        print(f"\nAn error occurred while saving to Excel ('{excel_filename}'): {e}")

    # Save to CSV (.csv)
    try:
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
        print(f"Data successfully saved to {csv_filename}")
    except Exception as e:
        print(f"\nAn error occurred while saving to CSV ('{csv_filename}'): {e}")

    # Display the first few rows and summary
    print("\n--- Scraped Data Preview (First 5 Rows) ---")
    print(df.head())
    print("--- Scraped Data Preview (Last 5 Rows) ---")
    print(df.tail())
    print("-------------------------------------------")
    # Check for rows where parsing might have failed completely
    null_rows = df[df['Status'].isnull() & df['Amount'].isnull()].shape[0]
    print(f"\nTotal rows scraped: {len(df)}")
    if null_rows > 0:
        print(f"Warning: Found {null_rows} rows where Status and Amount could not be parsed.")
    # Check how many rows are missing time
    missing_time_rows = df['Time'].isnull().sum()
    if missing_time_rows > 0:
         print(f"Info: Found {missing_time_rows} rows missing extracted Time data.")

else:
    print("\nNo data was extracted or DataFrame is empty, CSV/Excel files not created.")
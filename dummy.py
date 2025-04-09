# Import necessary libraries
import pandas as pd
from bs4 import BeautifulSoup
import re # Import regular expressions library
import os # To check if file exists

# --- Configuration ---
html_file_path = 'My Activity.html' # Make sure this file is in the same directory as the script, or provide the full path
excel_filename = 'scraped_data.xlsx'
csv_filename = 'scraped_data.csv'

# --- Step 1: Read HTML Content from File ---
html_content = None
if not os.path.exists(html_file_path):
    print(f"Error: HTML file not found at '{html_file_path}'")
    exit() # Stop the script if the file doesn't exist

try:
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    print(f"Successfully read HTML content from '{html_file_path}'")
except Exception as e:
    print(f"Error reading HTML file: {e}")
    exit() # Stop if reading fails

if not html_content:
    print("HTML content is empty. Exiting.")
    exit()

# --- Step 2: Parse the HTML ---
soup = BeautifulSoup(html_content, 'html.parser')

# --- Step 3: Find all transaction blocks ---
transaction_blocks = soup.find_all('div', class_='outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp')
if not transaction_blocks:
    print("Warning: No transaction blocks found with class 'outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp'. Check the HTML structure and class names.")

# --- Step 4: Extract Data Points ---
extracted_data = []

for block in transaction_blocks:
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
        print("Warning: Could not find title tag in a block.")


    # Find the *inner* grid first, then the specific content cell within it
    inner_grid = block.find('div', class_='mdl-grid')
    if not inner_grid:
        print("Warning: Could not find inner 'mdl-grid' within an outer block.")
        extracted_data.append(data) # Append partially filled data or empty data
        continue # Skip to the next block

    # Find the main content cell containing the transaction details
    # It's the first div with these classes that doesn't have 'mdl-typography--text-right'
    content_cell = inner_grid.find('div', class_='content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1')

    if content_cell:
        lines = list(content_cell.stripped_strings)

        if len(lines) >= 1: # Need at least the main line
            main_line = lines[0]

            # 2, 3, 4, 5. Extract Status, Amount, Recipient/Sender, Payment Method
            # Revised Regex: Handles missing "to/from" and "using" parts
            # - Group 1: Status (Paid|Sent|Received)
            # - Group 2: Amount (₹X.XX)
            # - Group 3: (Optional) Recipient/Sender info (preceded by "to" or "from")
            # - Group 4: (Optional) Payment method details (preceded by "using")
            pattern = re.compile(
                r'^(Paid|Sent|Received)\s+'                       # Status
                r'(₹\d+\.\d{2,})\s*'                             # Amount (allowing optional space after)
                r'(?:(?:to|from)\s+(.*?))?\s*'                   # Optional Recipient/Sender (non-capturing group for "to/from")
                r'(?:using\s+(.*?))?$'                           # Optional Payment Method (non-capturing group for "using")
                , re.IGNORECASE
            )
            match = pattern.match(main_line)

            if match:
                data['Status'] = match.group(1).capitalize()
                data['Amount'] = match.group(2)
                # Check if optional groups were captured
                if match.group(3):
                    data['Recipient/Sender Info'] = f"{'to' if data['Status'] == 'Paid' or data['Status'] == 'Sent' else 'from'} {match.group(3).strip()}" # Add 'to' or 'from' back based on Status
                if match.group(4):
                    data['Payment Method'] = f"using {match.group(4).strip()}"
            else:
                 print(f"Warning: Could not parse main transaction line: {main_line}")


            # 6, 7. Extract Date and Time from the *second* line, if it exists
            if len(lines) >= 2:
                date_time_line = lines[1]
                # Example: Apr 9, 2025, 9:22:11 PM GMT+05:30
                parts = date_time_line.split(',')
                if len(parts) >= 3:
                    data['Date'] = f"{parts[0].strip()} {parts[1].strip()}"

                    # Extract time (handle potential extra spaces or characters like  )
                    time_match = re.search(r'(\d{1,2}:\d{2}:\d{2}\s?.*?\s?[AP]M)', parts[2]) # Made space before AM/PM optional
                    if time_match:
                         # Clean up potential HTML entities like   which is a narrow non-breaking space (\u202f)
                         time_str = time_match.group(1).replace('\u202f', ' ').replace(' ', ' ').strip()
                         # Further clean up potential multiple spaces introduced
                         time_str = re.sub(r'\s+', ' ', time_str)
                         data['Time'] = time_str
                    else:
                        print(f"Warning: Could not parse time from: {parts[2]}")
                else:
                     print(f"Warning: Could not parse date/time line: {date_time_line}")
            else:
                print(f"Warning: Only one line found in content cell, skipping date/time extraction for: {main_line}")

        else:
             print(f"Warning: Content cell is empty or contains no text: {content_cell}")

    else:
        # This message should appear less often now with the refined selector
        print("Warning: Could not find the primary content cell (div.content-cell.mdl-cell--6-col.mdl-typography--body-1) within the inner grid of a block.")

    # Add the extracted data dictionary to our list
    extracted_data.append(data)

# --- Step 5: Create Pandas DataFrame ---
# Filter out any potential rows where critical info might be missing if needed,
# although the current setup adds rows even if some fields are None.
df = pd.DataFrame(extracted_data)

# --- Step 6: Save Files ---
if not df.empty:
    # Save to Excel (.xlsx)
    try:
        # Ensure openpyxl is installed: pip install openpyxl
        df.to_excel(excel_filename, index=False, engine='openpyxl')
        print(f"\nData successfully saved to {excel_filename}")
    except ImportError:
        print("\nCould not save to Excel. Please install 'openpyxl': pip install openpyxl")
    except Exception as e:
        print(f"\nAn error occurred while saving to Excel: {e}")

    # Save to CSV (.csv)
    try:
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
        print(f"Data successfully saved to {csv_filename}")
    except Exception as e:
        print(f"An error occurred while saving to CSV: {e}")

    # Display the first few rows of the DataFrame (optional)
    print("\n--- Scraped Data Preview ---")
    print(df.head())
    print("\n--------------------------")
else:
    print("\nNo data was extracted, CSV/Excel files not created.")
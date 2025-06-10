import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
from bs4 import BeautifulSoup
import importlib.util
import re  # Add this import for regular expressions

def load_python_file(file_path):
    """Load a Python file as a module"""
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def process_gpay_data(html_content):
    """Process Google Pay HTML data with exact logic from gpay.py"""
    try:
        # Parse HTML content
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except ImportError:
            soup = BeautifulSoup(html_content, 'html.parser')

        # Find all transaction blocks
        transaction_blocks = soup.find_all('div', class_='outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp')
        if not transaction_blocks:
            st.warning("No transaction blocks found. Check if this is the correct Google Pay Activity file.")
            return None

        # Extract Data Points
        extracted_data = []

        for block_index, block in enumerate(transaction_blocks):
            data = {
                'Title': None,
                'Status': None,
                'Amount': None,
                'Recipent/Sender Info': None,
                'Payment Method': None,
                'Date': None,
                'Time': None
            }

            # Extract Title
            title_tag = block.find('p', class_='mdl-typography--title')
            if title_tag:
                data['Title'] = title_tag.get_text(strip=True)

            # Find inner grid and content cell
            inner_grid = block.find('div', class_='mdl-grid')
            if not inner_grid:
                extracted_data.append(data)
                continue

            content_cell = inner_grid.find('div', class_='content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1')

            if content_cell:
                lines = [text for text in content_cell.stripped_strings]

                if len(lines) >= 1:
                    main_line = lines[0]

                    # Parse Status and Amount
                    match_base = re.match(
                        r'^(Paid|Sent|Received)\s+'
                        r'(₹[\d,]+\.\d{2,})\s*'
                        r'(.*)$',
                        main_line,
                        re.IGNORECASE | re.DOTALL
                    )

                    if match_base:
                        data['Status'] = match_base.group(1).capitalize()
                        data['Amount'] = match_base.group(2)
                        remainder = match_base.group(3).strip()

                        # Parse recipient and payment method
                        using_match = re.search(r'\susing\s+(.*)$', remainder, re.IGNORECASE)
                        if using_match:
                            payment_method = f"using {using_match.group(1).strip()}"
                            potential_recipient = remainder[:using_match.start()].strip()
                            
                            if potential_recipient.lower().startswith('to ') or potential_recipient.lower().startswith('from '):
                                prefix_lower = potential_recipient[:potential_recipient.find(' ')+1].lower()
                                status_lower = data['Status'].lower()
                                if (status_lower in ['paid', 'sent'] and prefix_lower == 'to ') or \
                                   (status_lower == 'received' and prefix_lower == 'from '):
                                    data['Recipent/Sender Info'] = potential_recipient
                            elif potential_recipient:
                                prefix = 'to' if data['Status'] in ['Paid', 'Sent'] else 'from'
                                data['Recipent/Sender Info'] = f"{prefix} {potential_recipient}"

                            data['Payment Method'] = payment_method
                        else:
                            potential_recipient = remainder
                            if potential_recipient.lower().startswith('to ') or potential_recipient.lower().startswith('from '):
                                prefix_lower = potential_recipient[:potential_recipient.find(' ')+1].lower()
                                status_lower = data['Status'].lower()
                                if (status_lower in ['paid', 'sent'] and prefix_lower == 'to ') or \
                                   (status_lower == 'received' and prefix_lower == 'from '):
                                    data['Recipent/Sender Info'] = potential_recipient
                            elif potential_recipient:
                                prefix = 'to' if data['Status'] in ['Paid', 'Sent'] else 'from'
                                data['Recipent/Sender Info'] = f"{prefix} {potential_recipient}"

                    # Parse Date and Time
                    if len(lines) >= 2:
                        date_time_line = lines[1]
                        parts = date_time_line.split(',')
                        if len(parts) >= 3:
                            data['Date'] = f"{parts[0].strip()} {parts[1].strip()}"
                            time_part_str = parts[2].strip()
                            time_match = re.search(r'(\d{1,2}:\d{2}:\d{2}(\s*|\u202f)[AP]M)', time_part_str)
                            if time_match:
                                time_str = time_match.group(1)
                                time_str = time_str.replace('\u202f', ' ').replace(' ', ' ').strip()
                                time_str = re.sub(r'\s+', ' ', time_str)
                                data['Time'] = time_str

            extracted_data.append(data)

        # Create DataFrame and clean data
        if not extracted_data:
            st.error("No data was extracted from the file")
            return None

        df = pd.DataFrame(extracted_data)
        df = df[['Title', 'Status', 'Amount', 'Recipent/Sender Info', 'Payment Method', 'Date', 'Time']]

        # Data Cleansing
        empty_both = df['Recipent/Sender Info'].isna() & df['Payment Method'].isna()
        status_sent_paid = df['Status'].isin(['Sent', 'Paid'])
        status_received = df['Status'] == 'Received'

        df.loc[empty_both & status_sent_paid, 'Recipent/Sender Info'] = 'to Unknown'
        df.loc[empty_both & status_received, 'Recipent/Sender Info'] = 'from Unknown'
        df.loc[empty_both, 'Payment Method'] = 'Unknown'
        df.loc[df['Payment Method'].isna(), 'Payment Method'] = 'Unknown'

        # Fix misaligned bank account information
        bank_account_mask = df['Recipent/Sender Info'].str.contains('to using Bank Account', na=False)
        for idx in df[bank_account_mask].index:
            recipient_value = df.loc[idx, 'Recipent/Sender Info']
            payment_method = recipient_value.replace('to ', '', 1)
            df.loc[idx, 'Payment Method'] = payment_method
            df.loc[idx, 'Recipent/Sender Info'] = 'to Unknown'

        return df

    except Exception as e:
        st.error(f"Error processing Google Pay data: {str(e)}")
        return None

def process_paytm_data(excel_content):
    """Process Paytm data from uploaded Excel file using same logic as paytm.py"""
    try:
        # Read Excel content - specifically the "Passbook Payment History" sheet
        df = pd.read_excel(excel_content, sheet_name='Passbook Payment History', dtype=str)
        
        # --- Step 1: Initial Cleanup ---
        # Trim Whitespace
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Rename Columns
        rename_map = {
            'Transaction Details': 'Recipent/Sender Info',
            'Your Account': 'Payment Method'
        }
        for old_name, new_name in rename_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Drop Non-Essential Columns
        cols_to_drop = ['UPI Ref No.', 'Order ID', 'Remarks', 'Tags']
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df.drop(columns=existing_cols_to_drop, inplace=True)
        
        # --- Step 2: Filter Comments ---
        comment_to_remove = "This is not included in total paid and received calculations."
        if 'Comment' in df.columns:
            df['Comment'] = df['Comment'].astype(str).fillna('')
            df = df[df['Comment'] != comment_to_remove].copy()
            df.drop(columns=['Comment'], inplace=True)
        
        # --- Step 3: Apply Transformations ---
        # Add Title
        df['Title'] = 'Paytm'
        
        # Process Amount and Status
        if 'Amount' in df.columns:
            original_amount_str = df['Amount'].astype(str).copy()
            df['Numeric_Amount'] = df['Amount'].str.replace(r'[+,"]', '', regex=True).str.strip()
            df['Numeric_Amount'] = pd.to_numeric(df['Numeric_Amount'], errors='coerce')
            
            # Create Status based on Amount
            df['Status'] = 'Unknown'
            df.loc[df['Numeric_Amount'] >= 0, 'Status'] = 'Received'
            df.loc[df['Numeric_Amount'] < 0, 'Status'] = 'Paid'
            
            # Format Amount
            failed_amount_mask = df['Numeric_Amount'].isna()
            df.loc[~failed_amount_mask, 'Amount'] = df.loc[~failed_amount_mask, 'Numeric_Amount'].abs().map('₹{:.2f}'.format)
            df.loc[failed_amount_mask, 'Amount'] = original_amount_str[failed_amount_mask]
            
            df.drop(columns=['Numeric_Amount'], inplace=True)
        
        # Format Date
        if 'Date' in df.columns:
            def format_date(date_str):
                try:
                    if '/' in date_str:
                        parts = date_str.split('/')
                        if len(parts) == 3:
                            day, month, year = parts
                            dt = pd.to_datetime(f"{year}-{month}-{day}")
                            return dt.strftime('%b %d %Y')
                    dt = pd.to_datetime(date_str)
                    return dt.strftime('%b %d %Y')
                except:
                    return date_str
            df['Date'] = df['Date'].apply(format_date)
        
        # Format Time
        if 'Time' in df.columns:
            original_times = df['Time'].astype(str).copy()
            try:
                datetime_times = pd.to_datetime(original_times, errors='coerce')
                failed_mask = datetime_times.isna()
                if (~failed_mask).any():
                    formatted_times = datetime_times[~failed_mask].dt.strftime('%I:%M:%S %p')
                    formatted_times = formatted_times.apply(lambda x: x[1:] if isinstance(x, str) and x.startswith('0') else x)
                    df.loc[~failed_mask, 'Time'] = formatted_times
                df.loc[failed_mask, 'Time'] = original_times[failed_mask]
            except:
                df['Time'] = original_times
        
        # --- Step 4: Final Column Ordering ---
        final_columns = [
            'Title', 'Status', 'Amount', 'Recipent/Sender Info',
            'Payment Method', 'Date', 'Time'
        ]
        
        # Keep only columns that exist
        final_columns = [col for col in final_columns if col in df.columns]
        df_cleaned = df[final_columns].copy()
        
        if df_cleaned.empty:
            st.error("No valid transactions found in Paytm data")
            return None
            
        return df_cleaned
        
    except Exception as e:
        st.error(f"Error processing Paytm data: {str(e)}")
        return None


def data_cleansing(combine_df):
    try:
        # Make a copy to avoid modifying the original dataframe
        df = combine_df.copy()

        # Convert to lowercase and strip spaces
        df['Recipent/Sender Info'] = df['Recipent/Sender Info'].str.lower().str.strip()
        
        # Text transformation mappings
        transformations = {
            r'recharge of jio mobile \d+': 'to jio prepaid',
            r'to jio prepaid( recharges)?': 'to jio prepaid',
            r'purchase of delhi metro qr tickets': 'to delhi metro',
            r'to delhi metro rail cor(poration ltd)?': 'to delhi metro',
            r'to dmrc( limited)?': 'to delhi metro',
            r'to abhibus( com)?': 'to abhibus',
            r'to amazon pay groceries': 'to amazon india',
            r'to amazon india': 'to amazon india',
            r'to archaeological survey of india( 02)?': 'to archaeological survey of india',
            r'to blu\s?smart mobility private limited': 'to blusmart mobility private limited',
            r'to delhivery\s?(?:private)?limited': 'to delhivery limited',
            r'to (?:hotel )?hari piorko( grand)?': 'to hotel hari piorko',
            r'to lords trading co(?:mpany)?': 'to lords trading company',
            r'to sanjay  kumar': 'to sanjay kumar',
            r'to yarraphagari pranay( theja)?': 'to yarraphagari pranay',
            r'(?:paid )?to www tatacliq com': 'to tatacliq'
        }
        
        # Apply transformations
        for pattern, replacement in transformations.items():
            df['Recipent/Sender Info'] = df['Recipent/Sender Info'].str.replace(
                pattern, replacement, regex=True
            )
        
        # Rows to delete
        rows_to_delete = [
            'money sent to ali elsayed abakar eldago',
            'received from ali elsayed abakar eldago',
            'to tata cliq'
        ]
        
        # Remove specified rows
        df = df[~df['Recipent/Sender Info'].isin(rows_to_delete)]

        df['Amount'] = df['Amount'].str.replace('₹', '').str.replace(',', '').astype(float)
            
        # Convert date to datetime
        def parse_date(date_str):
            try:
                return pd.to_datetime(date_str, format='%b %d %Y')
            except:
                try:
                    return pd.to_datetime(date_str, format='%b %d, %Y')
                except:
                    return pd.NaT
            
        df['Date'] = df['Date'].apply(parse_date)
            
        # Add useful columns for analysis
        df['Month'] = df['Date'].dt.strftime('%Y-%m')
        df['DayOfWeek'] = df['Date'].dt.day_name()
            
        # Clean up Payment Method
        mask_0572 = df['Payment Method'].str.contains('0572', na=False)
        mask_1552 = df['Payment Method'].str.contains('1552', na=False)
            
        df.loc[mask_0572, 'Payment Method'] = 'Jammu and Kashmir Bank - 0572'
        df.loc[mask_1552, 'Payment Method'] = 'State Bank Of India - 1552'

        # convert Status value 'Sent' to 'Paid'
        sent_df = df['Status'].str.contains('Sent',na=False)
        df.loc[sent_df,'Status'] = 'Paid'

        return df
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


def plot_overall_flow(sent_paid_amount, received_amount):
    """Plot overall transaction flow"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    net_flow = received_amount - sent_paid_amount
    
    # Pie Chart
    if sent_paid_amount > 0 or received_amount > 0:
        ax1.pie(
            [sent_paid_amount, received_amount],
            labels=['Sent/Paid', 'Received'],
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("pastel")[0:2]
        )
    ax1.set_title(f'Overall Flow (Net: ₹{net_flow:,.2f})')
    
    # Bar Chart
    categories = ['Sent/Paid', 'Received']
    values = [sent_paid_amount, received_amount]
    sns.barplot(x=categories, y=values, ax=ax2, palette="pastel")
    ax2.set_title('Total Sent/Paid vs. Received')
    ax2.set_ylabel('Amount (₹)')
    
    for i, v in enumerate(values):
        ax2.text(i, v, f'₹{v:,.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_monthly_trends(monthly_sent_paid, monthly_received):
    """Plot monthly transaction trends"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if not monthly_sent_paid.empty and not monthly_received.empty:
        monthly_summary = pd.DataFrame({
            'MonthStr': sorted(set(monthly_sent_paid.index) | set(monthly_received.index)),
            'TotalSentPaid': [monthly_sent_paid.get(m, 0) for m in sorted(set(monthly_sent_paid.index) | set(monthly_received.index))],
            'TotalReceived': [monthly_received.get(m, 0) for m in sorted(set(monthly_sent_paid.index) | set(monthly_received.index))]
        })
        
        sns.lineplot(data=monthly_summary, x='MonthStr', y='TotalSentPaid', 
                    marker='o', label='Sent/Paid', ax=ax)
        sns.lineplot(data=monthly_summary, x='MonthStr', y='TotalReceived', 
                    marker='o', label='Received', ax=ax)
        
        for idx, row in monthly_summary.iterrows():
            ax.annotate(f'₹{row["TotalSentPaid"]:,.0f}',
                       (idx, row['TotalSentPaid']),
                       textcoords="offset points",
                       xytext=(0,10),
                       ha='center',
                       fontsize=8)
            ax.annotate(f'₹{row["TotalReceived"]:,.0f}',
                       (idx, row['TotalReceived']),
                       textcoords="offset points",
                       xytext=(0,-15),
                       ha='center',
                       fontsize=8)
    
    ax.set_title('Monthly Transaction Trends')
    ax.set_xlabel('Month')
    ax.set_ylabel('Amount (₹)')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def plot_transaction_distribution(df):
    """Plot transaction amount distribution"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if not df.empty:
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
                200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 
                10000, 20000, 50000, 100000]
        labels = ['0-10', '11-20', '21-30', '31-40', '41-50', 
                 '51-60', '61-70', '71-80', '81-90', '91-100',
                 '101-200', '201-300', '301-400', '401-500',
                 '501-1K', '1K-2K', '2K-3K', '3K-4K', '4K-5K',
                 '5K-10K', '10K-20K', '20K-50K', '50K-100K']
        
        df['AmountBin'] = pd.cut(df['Amount'], bins=bins, labels=labels, right=True)
        counts = df['AmountBin'].value_counts().reindex(labels, fill_value=0)
        
        bars = ax.bar(range(len(labels)), counts.values, 
                     color='skyblue', alpha=0.7,
                     edgecolor='black', linewidth=1)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom')
        
        ax.set_title('Distribution of Transaction Amounts')
        ax.set_xlabel('Amount Ranges (₹)')
        ax.set_ylabel('Number of Transactions')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def plot_top_recipients(df):
    """Plot top recipients/senders"""
    sent_paid_df = df[df['Status'].isin(['Sent', 'Paid'])]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_recipients = sent_paid_df['Recipent/Sender Info'].value_counts().head(15)
    if not top_recipients.empty:
        bars = sns.barplot(y=top_recipients.index, x=top_recipients.values, 
                          ax=ax, palette="viridis", orient='h')
        
        for bar in bars.patches:
            ax.annotate(f'{int(bar.get_width())}',
                       (bar.get_width(), bar.get_y() + bar.get_height() / 2.),
                       ha='left', va='center',
                       xytext=(5, 0),
                       textcoords='offset points')
    
    ax.set_title('Top 15 Recipients/Senders')
    ax.set_xlabel('Number of Transactions')
    ax.set_ylabel('Recipient/Sender')
    plt.tight_layout()
    return fig

def plot_payment_methods(df):
    """Plot payment method distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    payment_method_counts = df['Payment Method'].value_counts()
    if not payment_method_counts.empty:
        bars = sns.barplot(y=payment_method_counts.index, x=payment_method_counts.values, 
                          ax=ax, palette="magma", orient='h')
        
        for bar in bars.patches:
            ax.annotate(f'{int(bar.get_width())}',
                       (bar.get_width(), bar.get_y() + bar.get_height() / 2.),
                       ha='left', va='center',
                       xytext=(5, 0),
                       textcoords='offset points')
    
    ax.set_title('Payment Method Usage')
    ax.set_xlabel('Number of Transactions')
    ax.set_ylabel('Payment Method')
    plt.tight_layout()
    return fig

def plot_daily_spending(df):
    """Plot daily spending patterns"""
    sent_paid_df = df[df['Status'].isin(['Sent', 'Paid'])]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    weekday_summary = sent_paid_df.groupby('DayOfWeek')['Amount'].sum().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    ax = sns.barplot(x=weekday_summary.index, y=weekday_summary.values, color='steelblue')
    
    for i, val in enumerate(weekday_summary.values):
        ax.text(i, val + (weekday_summary.max() * 0.02),
                f'₹{val:,.2f}', 
                ha='center', 
                va='bottom', 
                fontweight='bold')
    
    plt.title('Total Spending by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Total Amount Spent/Paid (₹)')
    plt.tight_layout()
    return fig




# Set page configuration
st.set_page_config(
    page_title="UPI Transaction Analysis",
    page_icon="💳",
    layout="wide"
)

# Custom CSS and Footer styles
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        margin-bottom: 60px;
    }
    .stSidebar {
        padding: 2rem;
        background-color: #f5f5f5;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f2f6;
        padding: 10px;
        text-align: right;
        padding-right: 20px;
        border-top: 1px solid #e0e0e0;
        font-size: 0.8em;
        z-index: 999;
    }
    .footer a {
        color: #000;
        text-decoration: none;
        margin: 0 5px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .footer img {
        vertical-align: middle;
        margin-right: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 UPI Transaction Analysis")
st.sidebar.write("Upload your transaction data files below:")

# File uploaders
gpay_file = st.sidebar.file_uploader("Upload Google Pay Activity (HTML)", type=['html'])
paytm_file = st.sidebar.file_uploader("Upload Paytm Transaction Data (XLSX)", type=['xlsx'])

# Generate Visuals Button
st.sidebar.markdown("---")
generate_button = st.sidebar.button("Generate Visualizations", type="primary")

if generate_button:
    if not gpay_file and not paytm_file:
        st.error("Please upload at least one data file (Google Pay or Paytm).")
        st.stop()

    # Process uploaded files
    df_gpay = None
    df_paytm = None
    
    with st.spinner('Processing data...'):
        if gpay_file:
            try:
                html_content = gpay_file.getvalue().decode('utf-8')
                df_gpay = process_gpay_data(html_content)
                if df_gpay is None:
                    st.error("Failed to process Google Pay data.")
            except Exception as e:
                st.error(f"Error processing Google Pay data: {str(e)}")

        if paytm_file:
            try:
                df_paytm = process_paytm_data(paytm_file)
                if df_paytm is None:
                    st.error("Failed to process Paytm data.")
            except Exception as e:
                st.error(f"Error processing Paytm data: {str(e)}")

    # Combine the dataframes if both exist
    dfs = []
    if df_gpay is not None:
        dfs.append(df_gpay)
    if df_paytm is not None:
        dfs.append(df_paytm)
    if not dfs:
        st.error("No data was successfully processed.")
        st.stop()
    
    # Combine dataframes
    df_combined = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    
    # Perform data cleansing
    df_combined = data_cleansing(df_combined)

    # Main content area
    st.title("UPI Transaction Analysis Dashboard")

    # Calculate metrics
    sent_paid_df = df_combined[df_combined['Status'].isin(['Sent', 'Paid'])]
    received_df = df_combined[df_combined['Status'] == 'Received']

    total_sent_paid = sent_paid_df['Amount'].sum()
    total_received = received_df['Amount'].sum()
        
    col1, col2, col3, col4 = st.columns(4)
        
    with col1:
        st.metric("Total Transactions", f"{len(df_combined):,}")
    with col2:
        st.metric("Total Paid Amount", f"₹{total_sent_paid:,.2f}")
    with col3:
        st.metric("Total Received Amount", f"₹{total_received:,.2f}")
    with col4:
        st.metric("Net Flow", f"₹{total_received - total_sent_paid:,.2f}")
    
    
    # Display summaries
    st.subheader("Overall Transaction Flow")
    st.pyplot(plot_overall_flow(total_sent_paid, total_received))
    
    st.subheader("Monthly Transaction Trends")
    monthly_sent_paid = sent_paid_df.groupby('Month')['Amount'].sum()
    monthly_received = received_df.groupby('Month')['Amount'].sum()
    st.pyplot(plot_monthly_trends(monthly_sent_paid, monthly_received))
   

    col1, col2 = st.columns(2)
        
    with col1:
        st.subheader("Transaction Distribution")
        st.pyplot(plot_transaction_distribution(df_combined))
            
        st.subheader("Payment Methods")
        st.pyplot(plot_payment_methods(df_combined))
        
    with col2:
        st.subheader("Top Recipients/Senders")
        st.pyplot(plot_top_recipients(df_combined))
            
        st.subheader("Daily Spending Patterns")
        st.pyplot(plot_daily_spending(df_combined))

    # Add download button for processed data
    st.sidebar.markdown("---")
    st.sidebar.subheader("Download Processed Data")
    csv = df_combined.to_csv(index=False)
    st.sidebar.download_button(
        "Download Combined Data",
        csv,
        "combined_transactions.csv",
        "text/csv",
        key='download-csv'
    )
    

else:
    # Display instructions when files are not uploaded
    st.markdown("""
    <span style="font-weight:bold; font-size:1.5em;">📊 UPI Data Analyzer</span>
    <ul>
        <li><b>UPI Data Analyzer</b> is a web application built using Python, Pandas, and Streamlit to help users extract and visualize transaction data from GPay and Paytm.</li>
        <li>It parses <code>.html</code> files from GPay and <code>.xlsx</code> files from Paytm, cleans and aligns the data, then outputs insightful charts showing monthly expenses, category-wise spending, and top merchants.</li>
        <li>Designed to give you a clearer picture of your digital spending habits with just a few clicks.</li>
    </ul>
    <b>📥 Data Download Guide</b>
    <ul>
        <li><b>How to download your GPay transaction data:</b>
            <ul>
                <li>Visit Google Takeout: <a href="https://takeout.google.com/" target="_blank">https://takeout.google.com/</a></li>
                <li>Deselect all, then scroll down and select Google Pay.</li>
                <li>Click Next step → Choose export type as "Export once".</li>
                <li>Set file type to .zip and delivery method to your email.</li>
                <li>Download the ZIP file from your email, extract it, and locate the file: <code>Google Pay/My Activity/My Activity.html</code>.</li>
            </ul>
        </li>
        <li><b>How to download your Paytm transaction data:</b>
            <ul>
                <li>Open the Paytm app on your mobile.</li>
                <li>Go to Balance & History → Passbook → Choose the wallet/bank account.</li>
                <li>Tap on "Statement" or "Download statement".</li>
                <li>Choose the date range you want and download the statement as <code>.xlsx</code>.</li>
            </ul>
        </li>
    </ul>
    <b>📚 Additional Resources</b>
    <ul>
        <li>🔗 GitHub Repository: UPI-Data-Scrap-Viz</li>
        <li>💻 Try on Google Colab: Colab Notebook</li>
    </ul>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <div style="text-align: right; padding-right: 20px;">
            Developed by 
            <a href="https://www.linkedin.com/in/raj-kapoor-aung-bo-bo-a34b47146" target="_blank">
                Raj(ABB)
            </a> | 
            <a href="https://github.com/ABoBo555/UPI-Data-Scrap-Viz" target="_blank">
                <img src="https://github.com/favicon.ico" width="16" height="16">
                GitHub
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)

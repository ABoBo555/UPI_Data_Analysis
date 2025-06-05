import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re
from datetime import datetime

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

def process_gpay_data(html_content):
    """Process Google Pay data from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    transaction_blocks = soup.find_all('div', class_='outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp')
    
    extracted_data = []
    
    for block in transaction_blocks:
        data = {
            'Title': None,
            'Status': None,
            'Amount': None,
            'Recipent/Sender Info': None,
            'Payment Method': None,
            'Date': None,
            'Time': None
        }
        
        title_tag = block.find('p', class_='mdl-typography--title')
        if title_tag:
            data['Title'] = title_tag.get_text(strip=True)
        
        content_cell = block.find('div', class_='content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1')
        if content_cell:
            lines = [text for text in content_cell.stripped_strings]
            if lines:
                main_line = lines[0]
                match_base = re.match(
                    r'^(Paid|Sent|Received)\s+(₹[\d,]+\.\d{2,})\s*(.*)$',
                    main_line,
                    re.IGNORECASE | re.DOTALL
                )
                
                if match_base:
                    data['Status'] = match_base.group(1).capitalize()
                    data['Amount'] = match_base.group(2)
                    remainder = match_base.group(3).strip()
                    
                    using_match = re.search(r'\susing\s+(.*)$', remainder, re.IGNORECASE)
                    if using_match:
                        data['Payment Method'] = f"using {using_match.group(1).strip()}"
                        potential_recipient = remainder[:using_match.start()].strip()
                        if potential_recipient.lower().startswith(('to ', 'from ')):
                            data['Recipent/Sender Info'] = potential_recipient
                        else:
                            prefix = 'to' if data['Status'] in ['Paid', 'Sent'] else 'from'
                            data['Recipent/Sender Info'] = f"{prefix} {potential_recipient}"
                    else:
                        if remainder:
                            prefix = 'to' if data['Status'] in ['Paid', 'Sent'] else 'from'
                            data['Recipent/Sender Info'] = f"{prefix} {remainder}"
                
                if len(lines) >= 2:
                    date_time_line = lines[1]
                    parts = date_time_line.split(',')
                    if len(parts) >= 3:
                        data['Date'] = f"{parts[0].strip()} {parts[1].strip()}"
                        time_part_str = parts[2].strip()
                        time_match = re.search(r'(\d{1,2}:\d{2}:\d{2}(\s*|\u202f)[AP]M)', time_part_str)
                        if time_match:
                            data['Time'] = time_match.group(1).replace('\u202f', ' ').strip()
        
        extracted_data.append(data)
    
    return pd.DataFrame(extracted_data)

def process_paytm_data(df):
    """Process Paytm data from DataFrame"""
    df = df.copy()
    
    # Rename columns for consistency
    rename_map = {
        'Transaction Details': 'Recipent/Sender Info',
        'Your Account': 'Payment Method'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Add Title column
    df['Title'] = 'Paytm'
    
    # Process Amount and create Status
    if 'Amount' in df.columns:
        # Convert Amount to string first to handle mixed types
        df['Amount'] = df['Amount'].astype(str)
        df['Numeric_Amount'] = df['Amount'].str.replace(r'[+,"]', '', regex=True).str.strip()
        df['Numeric_Amount'] = pd.to_numeric(df['Numeric_Amount'], errors='coerce')
        
        df['Status'] = 'Unknown'
        df.loc[df['Numeric_Amount'] >= 0, 'Status'] = 'Received'
        df.loc[df['Numeric_Amount'] < 0, 'Status'] = 'Paid'
        
        df.loc[df['Numeric_Amount'].notna(), 'Amount'] = df.loc[df['Numeric_Amount'].notna(), 'Numeric_Amount'].abs().map('₹{:.2f}'.format)
        df.drop(columns=['Numeric_Amount'], inplace=True)
    
    # Format Date and Time
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df['Date'] = df['Date'].dt.strftime('%b %d %Y')
    
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p', errors='coerce').dt.strftime('%I:%M:%S %p')
        df['Time'] = df['Time'].str.replace('^0', '', regex=True)
    
    # Reorder columns
    columns_order = ['Title', 'Status', 'Amount', 'Recipent/Sender Info', 'Payment Method', 'Date', 'Time']
    return df[columns_order]

def prepare_combined_data(gpay_df, paytm_df):
    """Combine and prepare data for analysis"""
    if gpay_df is not None and paytm_df is not None:
        df = pd.concat([gpay_df, paytm_df], ignore_index=True)
    elif gpay_df is not None:
        df = gpay_df.copy()
    elif paytm_df is not None:
        df = paytm_df.copy()
    else:
        raise ValueError("At least one dataframe must be provided")
    
    # FIXED: Clean amount values properly handling mixed types
    def clean_amount(amount_val):
        if pd.isna(amount_val):
            return np.nan
        
        # Convert to string to handle both string and numeric inputs
        amount_str = str(amount_val)
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[₹,]', '', amount_str).strip()
        
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return np.nan
    
    df['Amount'] = df['Amount'].apply(clean_amount)
    
    # Convert date to datetime
    def parse_date(date_str):
        if pd.isna(date_str):
            return pd.NaT
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
    
    # Payment Method standardization
    mask_0572 = df['Payment Method'].str.contains('0572', na=False)
    mask_1552 = df['Payment Method'].str.contains('1552', na=False)
    
    df.loc[mask_0572, 'Payment Method'] = 'Jammu and Kashmir Bank - 0572'
    df.loc[mask_1552, 'Payment Method'] = 'State Bank Of India - 1552'
    
    return df

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
    """Plot transaction amount distribution - COMPLETELY FIXED"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if not df.empty:
        # Create a temporary dataframe to avoid modifying the original
        plot_df = df.copy()
        
        # Ensure Amount is numeric and remove any NaN values
        plot_df['Amount'] = pd.to_numeric(plot_df['Amount'], errors='coerce')
        plot_df = plot_df.dropna(subset=['Amount'])
        
        if plot_df.empty:
            ax.text(0.5, 0.5, 'No valid amount data to display', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # CORRECTED: Define bin edges and labels properly
        bins = [0.99, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
                200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 
                10000, 20000, 50000, 100000, float('inf')]
        
        labels = ['1-10', '11-20', '21-30', '31-40', '41-50', 
                 '51-60', '61-70', '71-80', '81-90', '91-100',
                 '101-200', '201-300', '301-400', '401-500',
                 '501-1K', '1K-2K', '2K-3K', '3K-4K', '4K-5K',
                 '5K-10K', '10K-20K', '20K-50K', '50K-100K', '100K+']
        
        # Create bins with right-inclusive intervals
        plot_df['AmountBin'] = pd.cut(
            plot_df['Amount'],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True
        )
        
        # Count transactions in each bin
        counts = plot_df['AmountBin'].value_counts().reindex(labels, fill_value=0)
        
        # Create the bar plot
        bars = ax.bar(range(len(labels)), counts.values, 
                     color='skyblue', alpha=0.7,
                     edgecolor='black', linewidth=1)
        
        # Add count labels on bars
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
        
        # Debug: Print some sample mappings
        sample_amounts = [1, 10, 11, 20, 21, 30, 60, 150, 300.9]
        print("Debug - Amount to Bin mapping:")
        for amt in sample_amounts:
            if amt in plot_df['Amount'].values:
                bin_val = plot_df[plot_df['Amount'] == amt]['AmountBin'].iloc[0]
                print(f"Amount {amt} -> {bin_val}")
    
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
    try:
        combined_df = None
        
        # Process Google Pay data if available
        if gpay_file:
            gpay_content = gpay_file.read().decode('utf-8')
            gpay_df = process_gpay_data(gpay_content)
            combined_df = prepare_combined_data(gpay_df, None)
            st.sidebar.success("Google Pay data processed successfully!")
        
        # Process Paytm data if available
        if paytm_file:
            paytm_df = pd.read_excel(paytm_file, sheet_name='Passbook Payment History')
            paytm_df = process_paytm_data(paytm_df)
            if combined_df is not None:
                # If we already have GPay data, combine with it
                combined_df = prepare_combined_data(gpay_df, paytm_df)
                st.sidebar.success("Data combined successfully!")
            else:
                # If only Paytm data is available
                combined_df = prepare_combined_data(None, paytm_df)
                st.sidebar.success("Paytm data processed successfully!")
        
        if combined_df is None:
            st.error("Please upload at least one data file (Google Pay or Paytm).")
            st.stop()
            
        # Main content area
        st.title("UPI Transaction Analysis Dashboard")
        
        # Key metrics
        sent_paid_df = combined_df[combined_df['Status'].isin(['Sent', 'Paid'])]
        received_df = combined_df[combined_df['Status'] == 'Received']
        
        total_sent_paid = sent_paid_df['Amount'].sum()
        total_received = received_df['Amount'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(combined_df):,}")
        with col2:
            st.metric("Total Sent/Paid", f"₹{total_sent_paid:,.2f}")
        with col3:
            st.metric("Total Received", f"₹{total_received:,.2f}")
        with col4:
            st.metric("Net Flow", f"₹{total_received - total_sent_paid:,.2f}")
        
        # Visualizations
        st.subheader("Overall Transaction Flow")
        st.pyplot(plot_overall_flow(total_sent_paid, total_received))
        
        st.subheader("Monthly Transaction Trends")
        monthly_sent_paid = sent_paid_df.groupby('Month')['Amount'].sum()
        monthly_received = received_df.groupby('Month')['Amount'].sum()
        st.pyplot(plot_monthly_trends(monthly_sent_paid, monthly_received))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Distribution")
            st.pyplot(plot_transaction_distribution(combined_df))
            
            st.subheader("Payment Methods")
            st.pyplot(plot_payment_methods(combined_df))
        
        with col2:
            st.subheader("Top Recipients/Senders")
            st.pyplot(plot_top_recipients(combined_df))
            
            st.subheader("Daily Spending Patterns")
            st.pyplot(plot_daily_spending(combined_df))
            
        # Download processed data
        st.sidebar.markdown("---")
        st.sidebar.subheader("Download Processed Data")
        
        @st.cache_data
        def convert_df_to_csv():
            return combined_df.to_csv(index=False).encode('utf-8')
        
        st.sidebar.download_button(
            "Download Combined Data (CSV)",
            convert_df_to_csv(),
            "upi_transactions.csv",
            "text/csv",
            key='download-csv'
        )
        
    except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
        st.write("Debug info:")
        st.write(str(e))
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
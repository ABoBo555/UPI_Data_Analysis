import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
from bs4 import BeautifulSoup
import importlib.util

def load_python_file(file_path):
    """Load a Python file as a module"""
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def process_gpay_data(html_content):
    """Process GPay data using gpay.py"""
    # Create a temporary file to store the HTML content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(html_content)
        temp_html_path = temp_file.name

    try:
        # Load gpay.py as a module
        gpay_module = load_python_file('gpay.py')
        
        # Modify the module's file path variable
        gpay_module.html_file_path = temp_html_path
        
        # Process will create gpay.csv
        if hasattr(gpay_module, 'process_data'):
            gpay_module.process_data()
        
        # Read the resulting CSV
        if os.path.exists('gpay.csv'):
            return pd.read_csv('gpay.csv')
        return None
    finally:
        # Clean up temporary file
        os.unlink(temp_html_path)

def process_paytm_data(excel_content):
    """Process Paytm data using paytm.py"""
    # Create a temporary file to store the Excel content
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
        temp_file.write(excel_content.getvalue())
        temp_excel_path = temp_file.name

    try:
        # Load paytm.py as a module
        paytm_module = load_python_file('paytm.py')
        
        # Modify the module's file path variable
        paytm_module.excel_file_path = temp_excel_path
        
        # Process will create paytm.csv
        if hasattr(paytm_module, 'process_data'):
            paytm_module.process_data()
        
        # Read the resulting CSV
        if os.path.exists('paytm.csv'):
            return pd.read_csv('paytm.csv')
        return None
    finally:
        # Clean up temporary file
        os.unlink(temp_excel_path)


def data_cleansing(combine_df):
    try:
        """
    Perform data cleansing on the Recipent/Sender Info column.
    Includes standardizing text, replacing variants, and removing specific rows.
    """
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style for better visualizations
plt.style.use('default')  # Reset to default style first
sns.set_theme()  # Apply seaborn theme
sns.set_palette('pastel')  # Set color palette

def save_chart(name):
    """Save the current figure if it doesn't already exist"""
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Saved_charts')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Check if any file with this chart name already exists
    existing_files = [f for f in os.listdir(save_dir) if f.startswith(name + '_')]
    if existing_files:
        print(f"Chart '{name}' already exists, skipping save")
        plt.close()
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Saved {name} chart as {filepath}")
    plt.close()

def load_and_process_data():
    """Load and process the combined transaction data"""
    try:
        df_paytm = pd.read_csv('paytm.csv')
        df_gpay = pd.read_csv('gpay.csv')
        combined_df = pd.concat([df_paytm, df_gpay], ignore_index=True)
        combined_df.to_csv('combined_data.csv', index=False)
        print("Combined data saved to combined_data.csv")
    except FileNotFoundError:
        print("Warning: One or both of the input CSV files (paytm.csv, gpay.csv) were not found. Trying to load combined_data.csv directly.")
    except Exception as e:
        print(f"An error occurred during initial merge: {e}")

    try:
        # Load the combined data
        df = pd.read_csv('combined_data.csv', engine='python', skip_blank_lines=True)
        
        # Clean 'Amount' - remove ₹ and commas, convert to numeric
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
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def plot_transaction_analysis(df):
    """Create all visualizations for the transaction analysis"""
    if df is None:
        print("No data available for analysis")
        return
        
    # Prepare data
    sent_paid_df = df[df['Status'].isin(['Sent', 'Paid'])]
    received_df = df[df['Status'] == 'Received']
    
    total_sent_paid = sent_paid_df['Amount'].sum()
    total_received = received_df['Amount'].sum()
    net_flow = total_received - total_sent_paid

    # 1. Overall Flow - Pie Chart
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    if total_sent_paid > 0 or total_received > 0:
        ax.pie(
            [total_sent_paid, total_received],
            labels=['Sent/Paid', 'Received'],
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("pastel")[0:2]
        )
        ax.set_title(f'Overall Flow (Net: ₹{net_flow:,.2f})')
    else:
        ax.text(0.5, 0.5, 'No Transaction Data', ha='center', va='center')
        ax.set_title('Overall Flow')
    ax.axis('equal')
    plt.tight_layout()
    save_chart('overall_flow_pie')

    # 2. Overall Flow - Bar Chart
    plt.figure(figsize=(7, 5))
    categories = ['Sent/Paid', 'Received']
    values = [total_sent_paid, total_received]
    bars = sns.barplot(x=categories, y=values, palette="pastel")
    plt.title(f'Total Sent/Paid vs. Received (Net: ₹{net_flow:,.2f})')
    plt.ylabel('Amount (₹)')
    for bar in bars.patches:
        plt.annotate(f'₹{bar.get_height():,.2f}',
                    xy=(bar.get_x() + bar.get_width()/2., bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom')
    plt.tight_layout()
    save_chart('overall_flow_bar')

    # 3. Monthly Trends
    plt.figure(figsize=(12, 6))
    monthly_sent_paid = sent_paid_df.groupby('Month')['Amount'].sum()
    monthly_received = received_df.groupby('Month')['Amount'].sum()
    
    monthly_summary = pd.DataFrame({
        'MonthStr': sorted(set(monthly_sent_paid.index) | set(monthly_received.index)),
        'TotalSentPaid': [monthly_sent_paid.get(m, 0) for m in sorted(set(monthly_sent_paid.index) | set(monthly_received.index))],
        'TotalReceived': [monthly_received.get(m, 0) for m in sorted(set(monthly_sent_paid.index) | set(monthly_received.index))]
    })
    
    if not monthly_summary.empty:
        sns.lineplot(data=monthly_summary, x='MonthStr', y='TotalSentPaid', 
                    marker='o', label='Sent/Paid')
        sns.lineplot(data=monthly_summary, x='MonthStr', y='TotalReceived', 
                    marker='o', label='Received')
        
        for idx, row in monthly_summary.iterrows():
            if pd.notna(row['TotalSentPaid']):
                plt.annotate(f'₹{row["TotalSentPaid"]:,.0f}',
                           (idx, row['TotalSentPaid']),
                           textcoords="offset points",
                           xytext=(0,10),
                           ha='center',
                           fontsize=8)
            if pd.notna(row['TotalReceived']):
                plt.annotate(f'₹{row["TotalReceived"]:,.0f}',
                           (idx, row['TotalReceived']),
                           textcoords="offset points",
                           xytext=(0,-15),
                           ha='center',
                           fontsize=8)
        
        plt.title('Monthly Sent/Paid vs. Received')
        plt.xlabel('Month')
        plt.ylabel('Amount (₹)')
        plt.xticks(rotation=45)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No Monthly Data', ha='center', va='center')
    plt.tight_layout()
    save_chart('monthly_trends')

    # 4. Transaction Distribution
    plt.figure(figsize=(12, 7))
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
        
        bars = plt.bar(range(len(labels)), counts.values, 
                     color='skyblue', alpha=0.7,
                     edgecolor='black', linewidth=1)
        
        plt.title('Distribution of Transaction Amounts', fontsize=12, pad=20)
        plt.xlabel('Amount Ranges (₹)', fontsize=10)
        plt.ylabel('Number of Transactions', fontsize=10)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width()/2., height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.gca().set_axisbelow(True)
    else:
        plt.text(0.5, 0.5, 'No Transaction Data', ha='center', va='center')
    plt.tight_layout()
    save_chart('transaction_distribution')

    # 5. Top Recipients
    plt.figure(figsize=(10, 8))
    top_recipients = sent_paid_df['Recipent/Sender Info'].value_counts().head(15)
    
    if not top_recipients.empty:
        bars = sns.barplot(y=top_recipients.index, x=top_recipients.values, 
                          palette="viridis", orient='h')
        plt.title('Top 15 Recipients/Senders (Frequency)')
        plt.xlabel('Number of Transactions')
        plt.ylabel('Recipient/Sender')
        
        for bar in bars.patches:
            plt.annotate(f'{int(bar.get_width())}',
                        xy=(bar.get_width(), bar.get_y() + bar.get_height()/2.),
                        xytext=(5, 0), textcoords='offset points',
                        ha='left', va='center')
    else:
        plt.text(0.5, 0.5, 'No Recipient Data', ha='center', va='center')
    plt.tight_layout()
    save_chart('top_recipients')

    # 6. Payment Methods
    plt.figure(figsize=(10, 6))
    payment_method_counts = df['Payment Method'].value_counts()
    
    if not payment_method_counts.empty:
        bars = sns.barplot(y=payment_method_counts.index, x=payment_method_counts.values, 
                          palette="magma", orient='h')
        plt.title('Payment Method Usage')
        plt.xlabel('Number of Transactions')
        plt.ylabel('Payment Method')
        
        for bar in bars.patches:
            plt.annotate(f'{int(bar.get_width())}',
                        xy=(bar.get_width(), bar.get_y() + bar.get_height()/2.),
                        xytext=(5, 0), textcoords='offset points',
                        ha='left', va='center')
    else:
        plt.text(0.5, 0.5, 'No Payment Method Data', ha='center', va='center')
    plt.tight_layout()
    save_chart('payment_methods')

    # 7. Daily Spending
    plt.figure(figsize=(12, 6))
    weekday_summary = sent_paid_df.groupby('DayOfWeek')['Amount'].sum().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    bars = sns.barplot(x=weekday_summary.index, y=weekday_summary.values, color='steelblue')
    
    # Fixed data label placement - using a small fixed offset instead of percentage
    for i, val in enumerate(weekday_summary.values):
        plt.annotate(f'₹{val:,.2f}',
                    xy=(i, val),
                    xytext=(0, 5),  # Fixed 5-point offset instead of percentage
                    textcoords='offset points',
                    ha='center', 
                    va='bottom',
                    fontweight='bold')
    
    plt.title('Total Spending by Day of Week', fontsize=16)
    plt.xlabel('Day of Week')
    plt.ylabel('Total Amount Spent/Paid (₹)')
    plt.tight_layout()
    save_chart('daily_spending')

    # 8. Insights Dashboard
    plt.figure(figsize=(15, 8))
    plt.suptitle('Transaction Analysis Dashboard - Key Insights', fontsize=16, y=0.95)
    
    insights = [
        f"Total Transactions: {len(df):,}",
        f"Total Amount Sent/Paid: ₹{total_sent_paid:,.2f}",
        f"Total Amount Received: ₹{total_received:,.2f}",
        f"Net Flow: ₹{net_flow:,.2f}",
        f"Average Transaction Amount: ₹{df['Amount'].mean():,.2f}",
        f"Most Common Payment Method: {payment_method_counts.index[0]}",
        f"Most Active Day: {weekday_summary.idxmax()}",
        f"Number of Unique Recipients: {sent_paid_df['Recipent/Sender Info'].nunique():,}"
    ]
    
    plt.text(0.5, 0.5, '\n'.join(insights),
             ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.8, 
                      edgecolor='gray', boxstyle='round,pad=1'),
             fontsize=11,
             linespacing=1.3,
             family='Arial')
    
    plt.axis('off')
    plt.tight_layout()
    save_chart('insights_dashboard')

def main():
    """Main function to run the visualization process"""
    print("Loading and processing data...")
    df = load_and_process_data()
    if df is not None:
        print("Creating visualizations...")
        plot_transaction_analysis(df)
        print("All visualizations have been generated and saved.")
    else:
        print("Could not proceed with visualization due to data loading error.")

if __name__ == "__main__":
    main()
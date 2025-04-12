import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for handling potential NaN values
import os
from datetime import datetime

# Create Saved_charts directory if it doesn't exist
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Saved_charts')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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

# Set a consistent and aesthetically pleasing style
sns.set_theme(style="whitegrid")

# --- Data Loading and Initial Merging ---
# This part remains the same if you still need to merge paytm and gpay initially.
# If combined_data.csv already exists and is the primary source, you might skip this.
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


# --- Load and Clean Combined Data ---
try:
    # Load the combined data, skipping problematic header rows and fixing column parsing
    df = pd.read_csv('combined_data.csv', skiprows=40, engine='python', skip_blank_lines=True)
    
    # If we got only one column, try splitting it
    if len(df.columns) == 1:
        # Split the single column into multiple columns
        df = pd.read_csv('combined_data.csv', skiprows=40, engine='python', sep=',', skip_blank_lines=True)
        
    print("\nInitial data check:")
    print("Number of columns:", len(df.columns))
    print("Columns:", df.columns.tolist())
    
    # Rename columns if they exist, otherwise assign default names
    expected_columns = ['Source', 'Status', 'Amount', 'RecipientSenderInfo', 'PaymentMethod', 'Date', 'Time']
    if len(df.columns) >= len(expected_columns):
        df = df.iloc[:, :len(expected_columns)]  # Take only the columns we need
        df.columns = expected_columns
    
    # Clean 'Amount' - remove ₹ and commas, convert to numeric
    df['Amount'] = df['Amount'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    # Clean 'Date' and 'Time', combine them
    df['Date'] = df['Date'].astype(str).str.strip()
    df['Time'] = df['Time'].astype(str).str.strip()
    datetime_str = df['Date'] + ' ' + df['Time']

    # Attempt parsing with multiple formats
    # Format 1: Month Day Year (e.g., Apr 12 2025)
    format1 = '%b %d %Y %I:%M:%S %p'
    # Format 2: Day/Month/Year (e.g., 12/04/2025)
    format2 = '%d/%m/%Y %I:%M:%S %p'

    # Apply parsing, trying format1 then format2
    df['DateTime'] = pd.to_datetime(datetime_str, format=format1, errors='coerce')
    mask_failed_format1 = df['DateTime'].isna()
    df.loc[mask_failed_format1, 'DateTime'] = pd.to_datetime(datetime_str[mask_failed_format1], format=format2, errors='coerce')

    # Check if any dates still failed to parse
    if df['DateTime'].isna().any():
        print("Warning: Some Date/Time values could not be parsed with known formats.")
        # Optional: print the failed rows for debugging
        # print(df[df['DateTime'].isna()][['Date', 'Time']])

    # Drop rows where essential data (Amount, DateTime, Status) is missing
    df.dropna(subset=['Amount', 'DateTime', 'Status'], inplace=True)

    # Extract useful date/time features
    df['Month'] = df['DateTime'].dt.to_period('M')
    df['Year'] = df['DateTime'].dt.year
    df['DayOfWeek'] = df['DateTime'].dt.day_name()
    df['Hour'] = df['DateTime'].dt.hour

    # Clean 'RecipientSenderInfo' and 'PaymentMethod' (basic cleaning)
    df['RecipientSenderInfo'] = df['RecipientSenderInfo'].astype(str).str.strip().replace('nan', 'Unknown')
    df['PaymentMethod'] = df['PaymentMethod'].astype(str).str.strip().replace('nan', 'Unknown')
    # Example: Extract bank account info if pattern exists
    df['PaymentMethod'] = df['PaymentMethod'].str.extract(r'(Bank Account XXXXXX\d+)', expand=False).fillna('Other/Unknown')

    # After creating DateTime
    print("\nUnique days of week before processing:")
    df['DayOfWeek'] = df['DateTime'].dt.day_name()
    print(sorted(df['DayOfWeek'].unique().tolist()))

    print("Data loaded and cleaned successfully.")
    print(f"Total valid transactions: {len(df)}")
    # print("\nSample cleaned data:")
    # print(df.head())
    # print("\nData types:")
    # print(df.info())


    # --- Data Analysis ---
    sent_paid_df = df[df['Status'].isin(['Sent', 'Paid'])]
    received_df = df[df['Status'] == 'Received']

    total_sent_paid = sent_paid_df['Amount'].sum()
    total_received = received_df['Amount'].sum()
    net_flow = total_received - total_sent_paid

    # Monthly aggregation
    monthly_summary = df.groupby('Month').agg(
        TotalSentPaid=('Amount', lambda x: x[df.loc[x.index, 'Status'].isin(['Sent', 'Paid'])].sum()),
        TotalReceived=('Amount', lambda x: x[df.loc[x.index, 'Status'] == 'Received'].sum())
    ).reset_index()
    monthly_summary['MonthStr'] = monthly_summary['Month'].astype(str)

    # Top Recipients/Senders (for Sent/Paid transactions)
    top_recipients = sent_paid_df['RecipientSenderInfo'].value_counts().nlargest(15)

    # Payment Method Usage
    payment_method_counts = df['PaymentMethod'].value_counts()

    # Spending by Day of Week - With Debug Output
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Debug: Print unique days before processing
    print("\nDebug - Unique days in data:", sorted(sent_paid_df['DayOfWeek'].unique()))
    
    # Force the DayOfWeek to be exactly these values
    sent_paid_df['DayOfWeek'] = pd.Categorical(
        sent_paid_df['DayOfWeek'],
        categories=day_order,
        ordered=True
    )
    
    # Create daily spending with explicit aggregation
    daily_spending = (sent_paid_df
        .groupby('DayOfWeek', observed=True)['Amount']
        .sum()
        .reindex(day_order, fill_value=0)
        .reset_index())
    
    # Debug: Print final daily_spending DataFrame
    print("\nDebug - Daily spending data:")
    print(daily_spending)
    print("\nDebug - Shape:", daily_spending.shape)

    # --- Visualization ---
    print("\nGenerating visualizations...")

    # --- Plot 1: Overall Sent/Paid vs Received (Pie Chart) ---
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

    # --- Plot 1.5: Overall Sent/Paid vs Received (Bar Chart) ---
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    categories = ['Sent/Paid', 'Received']
    values = [total_sent_paid, total_received]
    bars = sns.barplot(x=categories, y=values, ax=ax, palette="pastel")
    ax.set_title(f'Total Sent/Paid vs. Received (Net: ₹{net_flow:,.2f})')
    ax.set_ylabel('Amount (₹)')
    # Add data labels
    for bar in bars.patches:
        ax.annotate(f'₹{bar.get_height():,.2f}',
                    (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    save_chart('overall_flow_bar')

    # --- Plot 2: Monthly Sent/Paid vs Received (Line Chart) ---
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    if not monthly_summary.empty:
        # Create line plots
        sent_line = sns.lineplot(data=monthly_summary, x='MonthStr', y='TotalSentPaid', marker='o', label='Sent/Paid', ax=ax)
        received_line = sns.lineplot(data=monthly_summary, x='MonthStr', y='TotalReceived', marker='o', label='Received', ax=ax)
        
        # Add value labels for Sent/Paid
        for idx, row in monthly_summary.iterrows():
            if pd.notna(row['TotalSentPaid']):
                ax.annotate(f'₹{row["TotalSentPaid"]:,.0f}',
                          (idx, row['TotalSentPaid']),
                          textcoords="offset points",
                          xytext=(0,10),
                          ha='center',
                          fontsize=8)
        
        # Add value labels for Received
        for idx, row in monthly_summary.iterrows():
            if pd.notna(row['TotalReceived']):
                ax.annotate(f'₹{row["TotalReceived"]:,.0f}',
                          (idx, row['TotalReceived']),
                          textcoords="offset points",
                          xytext=(0,-15),
                          ha='center',
                          fontsize=8)
        
        ax.set_title('Monthly Sent/Paid vs. Received')
        ax.set_xlabel('Month')
        ax.set_ylabel('Amount (₹)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No Monthly Data', ha='center', va='center')
        ax.set_title('Monthly Sent/Paid vs. Received')
    plt.tight_layout()
    save_chart('monthly_trends')

    # --- Plot 3: Transaction Amount Distribution (Custom Bins Histogram) ---
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    if not df.empty:
        # Define custom bins and labels
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
                200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 
                10000, 20000, 50000, 100000]
        
        labels = ['0-10', '10-20', '20-30', '30-40', '40-50', 
                 '50-60', '60-70', '70-80', '80-90', '90-100',
                 '100-200', '200-300', '300-400', '400-500',
                 '500-1K', '1K-2K', '2K-3K', '3K-4K', '4K-5K',
                 '5K-10K', '10K-20K', '20K-50K', '50K-100K']
        
        # Calculate the histogram data
        counts, _ = np.histogram(df['Amount'], bins=bins)
        
        # Create the bar plot
        bars = ax.bar(range(len(labels)), counts, 
                     color='skyblue', alpha=0.7,
                     edgecolor='black', linewidth=1)
        
        # Customize the plot
        ax.set_title('Distribution of Transaction Amounts', fontsize=12, pad=20)
        ax.set_xlabel('Amount Ranges (₹)', fontsize=10)
        ax.set_ylabel('Number of Transactions', fontsize=10)
        
        # Set x-axis labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show label if there are transactions in that range
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
        
        # Add grid for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, 'No Transaction Data', ha='center', va='center')
        ax.set_title('Distribution of Transaction Amounts')
    
    save_chart('transaction_distribution')

    # --- Plot 4: Top Recipients/Senders (Horizontal Bar Chart) ---
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    if not top_recipients.empty:
        bars = sns.barplot(y=top_recipients.index, x=top_recipients.values, ax=ax, palette="viridis", orient='h')
        ax.set_title('Top 15 Recipients/Senders (Frequency)')
        ax.set_xlabel('Number of Transactions')
        ax.set_ylabel('Recipient/Sender')
        # Add data labels
        for bar in bars.patches:
            ax.annotate(f'{int(bar.get_width())}', # Format as integer for frequency
                        (bar.get_width(), bar.get_y() + bar.get_height() / 2.),
                        ha='left', va='center',
                        xytext=(5, 0),
                        textcoords='offset points')
    else:
        ax.text(0.5, 0.5, 'No Recipient Data', ha='center', va='center')
        ax.set_title('Top 15 Recipients/Senders (Frequency)')
    plt.tight_layout()
    save_chart('top_recipients')

    # --- Plot 5: Payment Method Usage ---
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    if not payment_method_counts.empty:
        bars = sns.barplot(y=payment_method_counts.index, x=payment_method_counts.values, ax=ax, palette="magma", orient='h')
        ax.set_title('Payment Method Usage')
        ax.set_xlabel('Number of Transactions')
        ax.set_ylabel('Payment Method')
        # Add data labels
        for bar in bars.patches:
            ax.annotate(f'{int(bar.get_width())}', # Format as integer for frequency
                        (bar.get_width(), bar.get_y() + bar.get_height() / 2.),
                        ha='left', va='center',
                        xytext=(5, 0),
                        textcoords='offset points')
    else:
        ax.text(0.5, 0.5, 'No Payment Method Data', ha='center', va='center')
        ax.set_title('Payment Method Usage')
    plt.tight_layout()
    save_chart('payment_methods')

    # --- Plot 6: Spending by Day of Week ---
    plt.figure(figsize=(10, 6))
    
    # Simplest possible implementation
    weekday_spending = sent_paid_df.groupby('DayOfWeek')['Amount'].sum()
    
    # Print debug info
    print("\nWeekday spending data:")
    print(weekday_spending)
    
    # Create plot with basic DataFrame
    plt.bar(range(7), [weekday_spending.get(day, 0) for day in day_order])
    plt.xticks(range(7), day_order, rotation=45)
    plt.title('Total Spending by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Total Amount Spent/Paid (₹)')
    
    # Add value labels
    for i, v in enumerate([weekday_spending.get(day, 0) for day in day_order]):
        plt.text(i, v, f'₹{v:,.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_chart('daily_spending')

    # After all individual plots, create final dashboard with insights
    def calculate_insights(df, sent_paid_df, received_df):
        insights = []
        
        # 1. Transaction Patterns
        avg_transaction = sent_paid_df['Amount'].mean()
        most_common_range = pd.cut(sent_paid_df['Amount'], 
                                 bins=[0, 100, 500, 1000, 5000, float('inf')],
                                 labels=['₹0-100', '₹100-500', '₹500-1000', '₹1000-5000', '₹5000+']
                                ).mode().iloc[0]
        
        # 2. Timing Patterns
        busy_day = sent_paid_df.groupby('DayOfWeek')['Amount'].count().idxmax()
        busy_hour = sent_paid_df.groupby('Hour')['Amount'].count().idxmax()
        
        # 3. Financial Flow
        net_monthly_flow = monthly_summary['TotalReceived'] - monthly_summary['TotalSentPaid']
        worst_month = monthly_summary.loc[net_monthly_flow.idxmin(), 'MonthStr']
        best_month = monthly_summary.loc[net_monthly_flow.idxmax(), 'MonthStr']
        
        # 4. Top Transactions
        top_expense = sent_paid_df.nlargest(1, 'Amount').iloc[0]
        top_income = received_df.nlargest(1, 'Amount').iloc[0]
        
        # 5. Regular Transactions
        regular_recipients = sent_paid_df['RecipientSenderInfo'].value_counts().head(3)
        
        # Format insights without emojis and with better grouping
        insights.extend([
            "TRANSACTION PATTERNS",
            f"• Average Transaction Amount: ₹{avg_transaction:,.2f}",
            f"• Most Common Range: {most_common_range}",
            "",
            "TIMING PATTERNS",
            f"• Most Active Day: {busy_day}",
            f"• Peak Hour: {busy_hour:02d}:00",
            "",
            "FINANCIAL FLOW",
            f"• Most Profitable Month: {best_month}",
            f"• Highest Spending Month: {worst_month}",
            "",
            "NOTABLE TRANSACTIONS",
            f"• Largest Expense: ₹{top_expense['Amount']:,.2f}",
            f"  To: {top_expense['RecipientSenderInfo']}",
            f"• Largest Received-Amount: ₹{top_income['Amount']:,.2f}",
            "",
            "TOP 3 FREQUENT RECIPIENTS",
            *[f"• {name}: {count} transactions" for name, count in regular_recipients.items()]
        ])
        
        return insights

    # Create final dashboard with insights
    plt.figure(figsize=(15, 8))
    plt.suptitle('Transaction Analysis Dashboard - Key Insights', fontsize=16, y=0.95)
    
    # Calculate insights
    insights = calculate_insights(df, sent_paid_df, received_df)
    
    # Create text box with insights
    insight_text = '\n'.join(insights)
    plt.text(0.5, 0.5, insight_text,
             ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.8, 
                      edgecolor='gray', boxstyle='round,pad=1'),
             fontsize=11,
             linespacing=1.3,
             family='Arial')  # Explicitly set font family
    
    plt.axis('off')
    plt.tight_layout()
    save_chart('insights_dashboard')

    print("Visualizations and insights dashboard generated.")

except FileNotFoundError:
    print("Error: combined_data.csv not found. Please ensure the file exists.")
except pd.errors.EmptyDataError:
    print("Error: combined_data.csv is empty or header is incorrect after skipping rows.")
except KeyError as e:
    print(f"Error: A required column is missing: {e}. Check CSV headers and skiprows.")
except Exception as e:
    print(f"An unexpected error occurred during analysis or visualization: {e}")



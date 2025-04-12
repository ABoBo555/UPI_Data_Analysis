import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Assuming 'paytm.csv' and 'gpay.csv' are in the current directory
# If not, adjust the file paths accordingly

try:
    df_paytm = pd.read_csv('paytm.csv')
    df_gpay = pd.read_csv('gpay.csv')

    # Merge the DataFrames
    combined_df = pd.concat([df_paytm, df_gpay], ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv('combined_data.csv', index=False)
    print("Combined data saved to combined_data.csv")

except FileNotFoundError:
    print("Error: One or both of the input CSV files were not found.")
except Exception as e:
    print(f"An error occurred: {e}")


# Now, let's visualize the combined data using matplotlib and seaborn

# Load the data
df = pd.read_csv('combined_data.csv')

# Data preprocessing (if needed)
# Convert 'Amount' column to numeric, handling potential errors
df['Amount'] = df['Amount'].str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)

# Convert 'Date' to datetime objects
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Create 'Month' column
df['Month'] = df['Date'].dt.to_period('M')

# 1. (Sent + Paid) vs. Received
sent_paid = df[df['Status'].isin(['Sent', 'Paid'])]['Amount'].sum()
received = df[df['Status'] == 'Received']['Amount'].sum()

fig1 = plt.figure(figsize=(6, 4))  # Reduced figure size
plt.bar(['Sent/Paid', 'Received'], [sent_paid, received], color=['skyblue', 'lightcoral'])
plt.title('Total Sent/Paid vs. Received Amount')
plt.xlabel('Transaction Type')
plt.ylabel('Amount (₹)')

# Add data points to the bars
for i, v in enumerate([sent_paid, received]):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)  # Smaller font size

st.pyplot(fig1)

# 2. Monthly (Sent + Paid) Spend vs. Received
monthly_data = df.groupby('Month').agg({'Amount': 'sum', 'Status': lambda x: x.value_counts().index[0] if not x.empty else None})

monthly_data['Sent+Paid'] = monthly_data.apply(lambda row: row['Amount'] if row['Status'] in ['Sent', 'Paid'] else 0, axis=1)
monthly_data['Received'] = monthly_data.apply(lambda row: row['Amount'] if row['Status'] == 'Received' else 0, axis=1)

fig2 = plt.figure(figsize=(10, 6))  # Adjusted figure size
plt.plot(monthly_data.index.astype(str), monthly_data['Sent+Paid'], marker='o', label='Sent/Paid')
plt.plot(monthly_data.index.astype(str), monthly_data['Received'], marker='o', label='Received')
plt.title('Monthly Sent/Paid vs. Received Amount')
plt.xlabel('Month')
plt.ylabel('Amount (₹)')
plt.xticks(rotation=45, ha="right")
plt.legend()
st.pyplot(fig2)

# 3. Top 5 Months with Most Expense
top_5_months = monthly_data.nlargest(5, 'Sent+Paid')
fig3 = plt.figure(figsize=(8, 6))
plt.bar(top_5_months.index.astype(str), top_5_months['Sent+Paid'], color='skyblue')
plt.title('Top 5 Months with Highest Spending')
plt.xlabel('Month')
plt.ylabel('Total Spent/Paid Amount (₹)')
plt.xticks(rotation=45, ha="right")

# Add data points to the bars
for i, v in enumerate(top_5_months['Sent+Paid']):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)  # Smaller font size

st.pyplot(fig3)

# 4. Recipient/Sender Info Visualization (Word Cloud or Bar Chart)
recipient_counts = df['Recipent/Sender Info'].value_counts().nlargest(10)  # Top 10
fig4 = plt.figure(figsize=(10, 6))
plt.bar(recipient_counts.index, recipient_counts.values, color='skyblue')
plt.title('Top 10 Recipents/Senders')
plt.xlabel('Recipent/Sender')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha="right")

# Add data points to the bars
for i, v in enumerate(recipient_counts.values):
    plt.text(i, v, f'{v}', ha='center', va='bottom', fontsize=8)  # Smaller font size

st.pyplot(fig4)

# 5. Frequency of Amount (Histogram)
fig5 = plt.figure(figsize=(10, 6))
plt.hist(df['Amount'], bins=20, color='skyblue', edgecolor='black')
plt.title('Frequency of Transaction Amounts')
plt.xlabel('Transaction Amount (₹)')
plt.ylabel('Frequency')
plt.show()

# 6. Monthly Bar Chart Comparison
fig6 = plt.figure(figsize=(10, 6))  # Adjusted figure size
plt.bar(monthly_data.index.astype(str), monthly_data['Sent+Paid'], label='Sent/Paid', color='skyblue')
fig5 = plt.figure(figsize=(10, 6))
plt.bar(monthly_data.index.astype(str), monthly_data['Received'], label='Received', color='lightcoral', bottom=monthly_data['Sent+Paid'])
plt.title('Monthly Comparison of Sent/Paid and Received Amounts')
plt.xlabel('Month')
plt.ylabel('Amount (₹)')
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.show()

fig7=plt.figure(figsize=(6, 4))  # Reduced figure size
plt.pie(
    [sent_paid, received],
    labels=['Sent/Paid', 'Received'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['skyblue', 'lightcoral']
)
plt.title('Overall Sent/Paid vs. Received Distribution')
plt.axis('equal')  # Ensures pie is a circle
st.pyplot(fig7)


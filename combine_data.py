import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime



def cleansing_data(df):
    """
    Perform data cleansing on the Recipent/Sender Info column.
    Includes standardizing text, replacing variants, and removing specific rows.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
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
    
    return df

def load_and_process_data(combine_df):
    
    try:
        # First apply data cleansing
        combine_df = cleansing_data(combine_df)
        
        combine_df['Amount'] = combine_df['Amount'].str.replace('₹', '').str.replace(',', '').astype(float)
        
        # Convert date to datetime
        def parse_date(date_str):
            try:
                return pd.to_datetime(date_str, format='%b %d %Y')
            except:
                try:
                    return pd.to_datetime(date_str, format='%b %d, %Y')
                except:
                    return pd.NaT
        
        combine_df['Date'] = combine_df['Date'].apply(parse_date)
        
        # Add useful columns for analysis
        combine_df['Month'] = combine_df['Date'].dt.strftime('%Y-%m')
        combine_df['DayOfWeek'] = combine_df['Date'].dt.day_name()
        
        # Clean up Payment Method
        mask_0572 = combine_df['Payment Method'].str.contains('0572', na=False)
        mask_1552 = combine_df['Payment Method'].str.contains('1552', na=False)
        
        combine_df.loc[mask_0572, 'Payment Method'] = 'Jammu and Kashmir Bank - 0572'
        combine_df.loc[mask_1552, 'Payment Method'] = 'State Bank Of India - 1552'

        # convert Status value 'Sent' to 'Paid'
        sent_df = combine_df['Status'].str.contains('Sent',na=False)
        combine_df.loc[sent_df,'Status'] = 'Paid'
        
        # Save the processed data
        combine_df.to_csv('final2.csv', index=False)
        
        # Return the processed DataFrame
        return combine_df
               
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":

    try:
        df_paytm = pd.read_csv('paytm.csv')
        df_gpay = pd.read_csv('gpay.csv')
        combine_df = pd.concat([df_paytm, df_gpay], ignore_index=True)
    except FileNotFoundError:
        print("Warning: One or both of the input CSV files (paytm.csv, gpay.csv) were not found. Trying to load combined_data.csv directly.")
    except Exception as e:
        print(f"An error occurred during initial merge: {e}")

    df = load_and_process_data(combine_df)
import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv('/Users/gopikasriram/Desktop/LBMAGold/lbma.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Convert the 'Date' column to a datetime object
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')


# Set the 'Date' column as the DataFrame index
df.set_index('Date', inplace=True)

# Convert the 'USD (AM)', 'USD (PM)', 'EURO (AM)', 'EURO (PM)', 'GBP (AM)', and 'GBP (PM)' columns to floats
cols_to_convert = ['USD (AM)', 'USD (PM)', 'EURO (AM)', 'EURO (PM)', 'GBP (AM)', 'GBP (PM)']
for col in cols_to_convert:
    df[col] = df[col].astype(float)

# Add new columns with the average of 'USD (AM)' and 'USD (PM)', 'EURO (AM)' and 'EURO (PM)', and 'GBP (AM)' and 'GBP (PM)'
df['USD (Average)'] = (df['USD (AM)'] + df['USD (PM)']) / 2
df['EURO (Average)'] = (df['EURO (AM)'] + df['EURO (PM)']) / 2
df['GBP (Average)'] = (df['GBP (AM)'] + df['GBP (PM)']) / 2


# Drop the 'USD (AM)', 'USD (PM)', 'EURO (AM)', 'EURO (PM)', 'GBP (AM)', and 'GBP (PM)' columns
cols_to_drop = ['USD (AM)', 'USD (PM)', 'EURO (AM)', 'EURO (PM)', 'GBP (AM)', 'GBP (PM)']
df.drop(columns=cols_to_drop, inplace=True)

# Save the cleaned DataFrame to a new CSV file
df.to_csv('/Users/gopikasriram/Desktop/LBMAGold/clean.csv')

# Print the cleaned DataFrame
print(df.tail())
print(df.head())

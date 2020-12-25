import sys
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    # Read the two csv files
    categories_df = pd.read_csv(categories_filepath)
    messages_df = pd.read_csv(messages_filepath)
    # Merge the two dataframes on id column
    consolidated_df = pd.merge(left=categories_df, right=messages_df, how='left', left_on='id', right_on='id')
    #print(consolidated_df.shape)
    return consolidated_df


def clean_data(df):
    #Transform data to have individual columns for the categories values
    # iterate over the dataframe row by row
    for index_label, row_series in df.iterrows():
        #Split the categories text into cats (category and values)
        cats = [cat.strip().split('-') for cat in row_series.categories.split(';')]
        # Add new columns for each cat and update the value
        for cat in cats:
            df.at[index_label , cat[0]] = cat[1]
    #Drop the categories column
    df.drop(columns=['categories'], inplace=True)
    #Find and delete duplicate rows in the dataframe
    # Deduplicate the dataframe by deleting the duplicate rows
    duplicate_index = df[df.duplicated(keep='first')].index
    # drop these row indexes from dataFrame 
    df.drop(duplicate_index, axis=0, inplace = True) 
    # delete any rows that are null so as to not break the code
    df.dropna(inplace=True)
    #print(df.shape)
    return df


def save_data(df, database_filename):
    # Load the processsed data
    import sqlite3
    conn = sqlite3.connect(database_filename)
    conn.execute('CREATE TABLE IF NOT EXISTS MESSAGES ('+ str(list(df.columns)) +')')
    conn.commit()
    df.to_sql('MESSAGES', conn, if_exists='replace', index = False)
    #print(pd.read_sql('SELECT count(*) FROM MESSAGES', con = conn))
    conn.close()
    
    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
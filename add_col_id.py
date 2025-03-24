import pandas as pd

def add_column_id(input_file, output_file):
    """
    Add a column_id field to the CSV based on source_id and position.
    The column_id is formatted as source_id_position where position
    is the sequential count of that source_id in the dataset.
    
    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str): Path to save the output CSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Initialize variables to track current source_id and position
    current_source_id = None
    position_counter = 0
    column_ids = []
    
    # Iterate through the rows to create column_id
    for _, row in df.iterrows():
        source_id = row['source_id']
        
        # Reset counter when source_id changes
        if source_id != current_source_id:
            current_source_id = source_id
            position_counter = 1
        else:
            position_counter += 1
            
        # Create and store the column_id
        column_id = f"{source_id}_{position_counter}"
        column_ids.append(column_id)
    
    # Add the new column_id to the dataframe
    df['column_id'] = column_ids
    
    new_order = ['uid', 'source_id', 'column_id', 'column_name', 'column_desc','table_name','table_desc', 'metadata']
    
    df = df[new_order]
    # Save the updated dataframe to a new CSV file
    df.to_csv(output_file, index=False)
    
    print(f"Added column_id to {len(df)} rows and saved to {output_file}")
    
    # Preview some rows
    # print("\nFirst few rows of updated data:")
    # print(df[['source_id', 'chunk', 'column_id']].head())
    
    return df

# Example usage
if __name__ == "__main__":
    input_file = "/Users/arvindsivaram/column-desc/grouped_chunks_with_column_desc.csv"
    output_file = "/Users/arvindsivaram/column-desc/g1_with_column_id_and_desc.csv"
    
    # Add column_id to the CSV
    updated_df = add_column_id(input_file, output_file)
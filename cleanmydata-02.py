import streamlit as st
import pandas as pd
import numpy as np
import openai

# Set your OpenAI API key here
openai.api_key = "sk-proj-XATwByNWpeow89UpAVTGT3BlbkFJOFC0HGF6XsREvBPbxmJl"  # Replace with your actual API key

# Function to get cleaning suggestions from OpenAI
def get_cleaning_suggestions(df):
    prompt = f"Here is a sample of the dataset:\n{df.head(5).to_string()}\n\n"
    prompt += "What are the data cleaning methods that can be performed to make this data more efficient?"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a data cleaning expert."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message['content'].strip()

# Function to handle missing numerical values
def handle_numerical_missing(df):
    numeric_method = st.sidebar.selectbox('Select method for numerical columns', ['Fill with mean', 'Fill with median', 'Fill with mode'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if numeric_method == 'Fill with mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        st.sidebar.write('Filled missing numeric values with mean.')
    elif numeric_method == 'Fill with median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        st.sidebar.write('Filled missing numeric values with median.')
    elif numeric_method == 'Fill with mode':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        st.sidebar.write('Filled missing numeric values with mode.')

# Function to handle missing categorical values
def handle_categorical_missing(df):
    categorical_method = st.sidebar.selectbox('Select method for categorical columns', ['Impute with ffill', 'Impute with bfill'])
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if categorical_method == 'Impute with ffill':
        df[categorical_cols] = df[categorical_cols].fillna(method='ffill')
        st.sidebar.write('Imputed missing categorical values with forward fill (ffill).')
    elif categorical_method == 'Impute with bfill':
        df[categorical_cols] = df[categorical_cols].fillna(method='bfill')
        st.sidebar.write('Imputed missing categorical values with backward fill (bfill).')

# Function to remove trailing spaces from columns
def remove_trailing_spaces(df):
    if st.sidebar.checkbox('Remove trailing spaces from all columns'):
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        st.write('Removed trailing spaces from all columns.')

# Function to capitalize all strings in the dataset
def capitalize_strings(df):
    if st.sidebar.checkbox('Capitalize all strings in the dataset'):
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.upper()
        st.write('Capitalized all strings in the dataset.')

# Function to remove duplicate rows
def remove_duplicates(df):
    initial_duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    st.write(f'Removed {initial_duplicates} duplicate rows.')
    return df

# Function to convert column to datetime
def convert_to_datetime(df):
    st.sidebar.subheader('Change Column to DateTime Format')
    col = st.sidebar.selectbox('Select column to convert to datetime', df.columns)
    try:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
        st.sidebar.write(f'Converted column {col} to datetime format (YYYY-MM-DD).')
    except Exception as e:
        st.error(f'Error converting column {col} to datetime: {e}')

# Function to change column datatype
def change_column_datatype(df):
    st.sidebar.subheader('Change Column Datatype')
    col = st.sidebar.selectbox('Select column to change datatype', df.columns)
    dtype = st.sidebar.selectbox('Select new datatype', ['int', 'float', 'str'])
    try:
        if dtype == 'int':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            st.sidebar.write(f'Changed datatype of column {col} to int.')
        elif dtype == 'float':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            st.sidebar.write(f'Changed datatype of column {col} to float.')
        elif dtype == 'str':
            df[col] = df[col].astype(str)
            st.sidebar.write(f'Changed datatype of column {col} to str.')
    except Exception as e:
        st.error(f'Error changing datatype of column {col}: {e}')

# Function to remove text before or after a delimiter
def remove_text_before_after_delimiter(df):
    st.sidebar.subheader('Remove Text Before/After Delimiter')
    col = st.sidebar.selectbox('Select column', df.columns)
    delimiter_expanded = st.sidebar.checkbox('Click to Remove Text Before/After Delimiter')
    
    if delimiter_expanded:
        delimiter = st.sidebar.text_input('Enter delimiter')
        action = st.sidebar.radio('Select action', ['Remove before delimiter', 'Remove after delimiter'])
        
        try:
            if action == 'Remove before delimiter':
                df[col] = df[col].str.split(delimiter).str[-1]
                st.sidebar.write(f'Removed text before delimiter "{delimiter}" in column {col}.')
            elif action == 'Remove after delimiter':
                df[col] = df[col].str.split(delimiter).str[0]
                st.sidebar.write(f'Removed text after delimiter "{delimiter}" in column {col}.')
        except Exception as e:
            st.error(f'Error removing text: {e}')
    else:
        st.sidebar.write('Expand to remove text before or after delimiter.')

# Function to impute missing values based on selected strategy
def impute_missing_values(df):
    st.sidebar.subheader('Handle Missing Values')
    if st.sidebar.checkbox('Handle Missing Values for Numerical Columns'):
        handle_numerical_missing(df)
    if st.sidebar.checkbox('Handle Missing Values for Categorical Columns'):
        handle_categorical_missing(df)
    if st.sidebar.checkbox('Drop Rows with Null Values'):
        df.dropna(inplace=True)
        st.write('Dropped rows with null values.')

# Function to set the first row as the header
def set_first_row_as_header(df):
    if st.sidebar.checkbox('Set first row as header'):
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        st.write('Set the first row as the header.')
    return df

# Function to clear all filters applied to the dataset
def clear_filters(df):
    if st.sidebar.button('Clear All Filters'):
        df = original_data.copy()
        st.write('Cleared all filters and restored the original data.')
    return df

# Placeholder for the original and cleaned data
original_data = None
cleaned_data = None

# Main page for upload and settings
st.header("Upload and Clean Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Function to load and display summary statistics
def load_data_display_summary(file):
    # Load data
    df = pd.read_csv(file)

    # Display original data
    st.subheader("Original Data")
    st.write(df)

    # Number of rows and columns in original data
    st.write(f"Number of rows in original data: {df.shape[0]}")
    st.write(f"Number of columns in original data: {df.shape[1]}")

    # Display EDA (Exploratory Data Analysis)
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("Data Summary:")
    st.write(df.describe())
    # st.write("Data Information:")
    # st.write(df.info())

    # Display data types and non-null counts
    st.subheader("Data Types and Non-Null Counts")
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("Non-Null Count:")
    st.write(df.notnull().sum())

    return df

if uploaded_file is not None:
    try:
        original_data = load_data_display_summary(uploaded_file)
        cleaned_data = original_data.copy()

        # Button to get cleaning suggestions
        if st.button('Get Cleaning Suggestions'):
            suggestions = get_cleaning_suggestions(original_data)
            st.subheader("Cleaning Suggestions from OpenAI")
            st.write(suggestions)

        # Sidebar for data cleaning techniques
        st.sidebar.header('Data Cleaning Techniques')

        # Perform data cleaning operations in the specified order
        st.sidebar.subheader('Set First Row as Header')
        cleaned_data = set_first_row_as_header(cleaned_data)

        # st.sidebar.subheader('Handle Missing Values')
        impute_missing_values(cleaned_data)

        st.sidebar.subheader('Change Column Datatype')
        if st.sidebar.checkbox('Change Column Datatype'):
            change_column_datatype(cleaned_data)

        st.sidebar.subheader('Change Column to DateTime Format')
        if st.sidebar.checkbox('Change Column to DateTime Format'):
            convert_to_datetime(cleaned_data)

        st.sidebar.subheader('Remove Trailing Spaces')
        remove_trailing_spaces(cleaned_data)

        st.sidebar.subheader('Capitalize Strings')
        capitalize_strings(cleaned_data)

        st.sidebar.subheader('Remove Text Before/After Delimiter')
        remove_text_before_after_delimiter(cleaned_data)

        # Clear filters
        cleaned_data = clear_filters(cleaned_data)

        # Display cleaned data
        st.subheader("Cleaned Data")
        st.write(cleaned_data)

        # Number of rows and columns in cleaned data
        st.write(f"Number of rows after cleaning: {cleaned_data.shape[0]}")
        st.write(f"Number of columns after cleaning: {cleaned_data.shape[1]}")

        # Download cleaned data
        csv = cleaned_data.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Dataset",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("Built by [Sai Raam](https://www.linkedin.com/in/srinrealyf/) - A [StatBir](https://twitter.com/statbir) Product | Made with ❤️ in India")

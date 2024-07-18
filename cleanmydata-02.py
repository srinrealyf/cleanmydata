import streamlit as st
import pandas as pd
import numpy as np
import openai

# Set your OpenAI API key here
openai.api_key = "sk-proj-c3HbJbkQB6f4NsFaBzIsT3BlbkFJFa0hJxdDGBWuU0FcQO9c"  # Replace with your actual API key

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
def handle_numerical_missing(df, selected_cols):
    numeric_method = st.sidebar.selectbox('Select method for numerical columns', ['Fill with mean', 'Fill with median', 'Fill with mode'])
    numeric_cols = selected_cols if selected_cols else df.select_dtypes(include=[np.number]).columns
    
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
def handle_categorical_missing(df, selected_cols):
    categorical_method = st.sidebar.selectbox('Select method for categorical columns', ['Impute with ffill', 'Impute with bfill'])
    categorical_cols = selected_cols if selected_cols else df.select_dtypes(include=['object']).columns
    
    if categorical_method == 'Impute with ffill':
        df[categorical_cols] = df[categorical_cols].fillna(method='ffill')
        st.sidebar.write('Imputed missing categorical values with forward fill (ffill).')
    elif categorical_method == 'Impute with bfill':
        df[categorical_cols] = df[categorical_cols].fillna(method='bfill')
        st.sidebar.write('Imputed missing categorical values with backward fill (bfill).')

# Function to remove trailing spaces from columns
def remove_trailing_spaces(df, selected_cols):
    df[selected_cols] = df[selected_cols].apply(lambda x: x.str.strip())
    st.sidebar.write('Removed trailing spaces from selected columns.')

# Function to capitalize all strings in the dataset or specific columns
def capitalize_strings(df, selected_cols):
    if selected_cols:
        for col in selected_cols:
            df[col] = df[col].apply(lambda x: x.upper() if isinstance(x, str) else x)
        st.sidebar.write(f'Capitalized strings in columns: {", ".join(selected_cols)}')
    else:
        df[df.select_dtypes(include=['object']).columns] = df[df.select_dtypes(include=['object']).columns].apply(lambda x: x.str.upper() if isinstance(x, str) else x)
        st.sidebar.write('Capitalized all strings in the dataset.')


# Function to lowercase all strings in the dataset or specific columns
def lowercase_strings(df, selected_cols):
    if selected_cols:
        for col in selected_cols:
            df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
        st.sidebar.write(f'Lowercased strings in columns: {", ".join(selected_cols)}')
    else:
        df[df.select_dtypes(include=['object']).columns] = df[df.select_dtypes(include=['object']).columns].apply(lambda x: x.str.lower() if isinstance(x, str) else x)
        st.sidebar.write('Lowercased all strings in the dataset.')

# Function to remove text before or after a delimiter
def remove_text_before_after_delimiter(df, selected_cols):
    delimiter = st.sidebar.text_input('Enter delimiter')
    action = st.sidebar.radio('Select action', ['Remove before delimiter', 'Remove after delimiter'])
    
    for col in selected_cols:
        if action == 'Remove before delimiter':
            df[col] = df[col].str.split(delimiter).str[-1]
        elif action == 'Remove after delimiter':
            df[col] = df[col].str.split(delimiter).str[0]
    st.sidebar.write(f'Removed text {action} delimiter "{delimiter}" in columns: {", ".join(selected_cols)}')

# Function to impute missing values based on selected strategy for specific columns
def impute_missing_values(df):
    numeric_cols = st.sidebar.multiselect('Select numerical columns to handle missing values', df.select_dtypes(include=[np.number]).columns)
    if numeric_cols:
        handle_numerical_missing(df, numeric_cols)
    
    categorical_cols = st.sidebar.multiselect('Select categorical columns to handle missing values', df.select_dtypes(include=['object']).columns)
    if categorical_cols:
        handle_categorical_missing(df, categorical_cols)
    
    if st.sidebar.checkbox('Drop Rows with Null Values'):
        df.dropna(inplace=True)
        st.sidebar.write('Dropped rows with null values.')

# Function to set the first row as the header
def set_first_row_as_header(df):
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    st.sidebar.write('Set the first row as the header.')
    return df

# Function to clear all filters applied to the dataset
def clear_filters(df):
    df = original_data.copy()
    st.sidebar.write('Cleared all filters and restored the original data.')
    return df

# Function to change column datatype
def change_column_datatype(df, selected_cols):
    if selected_cols:
        dtype = st.sidebar.selectbox('Select new datatype', ['int', 'float', 'str'])
        try:
            for col in selected_cols:
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
            st.sidebar.error(f'Error changing datatype: {e}')


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

    # Display data types and non-null counts side by side
    st.subheader("Data Types and Non-Null Counts")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Data Types:")
        st.write(df.dtypes)

    with col2:
        st.write("Non-Null Count:")
        st.write(df.notnull().sum())

    # Button to get cleaning suggestions
    if st.button('Get AI Cleaning Suggestions'):
        suggestions = get_cleaning_suggestions(df)
        st.subheader("Cleaning Suggestions from OpenAI")
        st.write(suggestions)

    return df

if uploaded_file is not None:
    try:
        original_data = load_data_display_summary(uploaded_file)
        cleaned_data = original_data.copy()

        # Sidebar for data cleaning techniques
        st.sidebar.header('Data Cleaning Techniques')

        # Perform data cleaning operations in the specified order
        if st.sidebar.checkbox('Set First Row as Header'):
            st.subheader('Set First Row as Header')
            cleaned_data = set_first_row_as_header(cleaned_data)
        
        if st.sidebar.checkbox('Impute Missing Values'):
            st.subheader('Impute Missing Values')
            impute_missing_values(cleaned_data)
        
        if st.sidebar.checkbox('Change Column Datatype'):
            st.subheader('Change Column Datatype')
            selected_cols = st.sidebar.multiselect('Select columns to change datatype', cleaned_data.columns)
            if selected_cols:
                change_column_datatype(cleaned_data, selected_cols)
        
        if st.sidebar.checkbox('Remove Trailing Spaces'):
            st.subheader('Remove Trailing Spaces')
            selected_cols = st.sidebar.multiselect('Select columns to remove trailing spaces', cleaned_data.select_dtypes(include=['object']).columns)
            if selected_cols:
                remove_trailing_spaces(cleaned_data, selected_cols)
        
        if st.sidebar.checkbox('Capitalize Strings'):
            st.subheader('Capitalize Strings')
            selected_cols = st.sidebar.multiselect('Select columns to capitalize strings', cleaned_data.select_dtypes(include=['object']).columns)
            if selected_cols:
                capitalize_strings(cleaned_data, selected_cols)
        
        if st.sidebar.checkbox('Lowercase Strings'):
            st.subheader('Lowercase Strings')
            selected_cols = st.sidebar.multiselect('Select columns to lowercase strings', cleaned_data.select_dtypes(include=['object']).columns)
            if selected_cols:
                lowercase_strings(cleaned_data, selected_cols)
        
        if st.sidebar.checkbox('Remove Text Before/After Delimiter'):
            st.subheader('Remove Text Before/After Delimiter')
            selected_cols = st.sidebar.multiselect('Select columns to remove text before/after delimiter', cleaned_data.columns)
            if selected_cols:
                remove_text_before_after_delimiter(cleaned_data, selected_cols)
        
        if st.sidebar.button('Clear All Filters'):
            cleaned_data = clear_filters(cleaned_data)
        
        # Display cleaned data
        st.subheader("Cleaned Data")
        st.write(cleaned_data)

        # Number of rows and columns in cleaned data
        st.write(f"Number of rows after cleaning: {cleaned_data.shape[0]}")
        st.write(f"Number of columns after cleaning: {cleaned_data.shape[1]}")

        # Display data types and non-null counts after cleaning
        st.subheader("Data Types and Non-Null Counts After Cleaning")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Data Types:")
            st.write(cleaned_data.dtypes)

        with col2:
            st.write("Non-Null Count:")
            st.write(cleaned_data.notnull().sum())

        # Download cleaned data
        st.download_button(
            label="Download Cleaned Data as CSV",
            data=cleaned_data.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

        # Embed Google Form
        st.markdown('<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSeOo0Js0LTPBCCmZPaeYfff119SpLE7qshxID209x0sJUhmcQ/viewform?embedded=true" width="640" height="2442" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown('<footer style="text-align: center;">Built by Sai Raam, a StatBir product.</footer>', unsafe_allow_html=True)

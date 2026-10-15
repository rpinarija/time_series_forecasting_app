"""Data loading functions for different data sources."""

import re

import pandas as pd
import requests
import streamlit as st


def fix_dtypes(df):
    """Convert object columns to datetime or numeric where possible (Arrow compatibility)."""
    for col in df.columns:
        # Text columns are 'object' in pandas < 3 and 'str' in pandas >= 3
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue

        # Only treat columns as dates when their name suggests it; otherwise
        # numeric-looking strings like "12" would be parsed as timestamps.
        if any(word in str(col).lower() for word in ['date', 'time', 'period', 'month', 'year']):
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().all():
                df[col] = converted
                continue

        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
            continue

        df[col] = df[col].astype(str)

    return df


def set_datetime_index(df):
    """Use the first datetime column as a sorted index, if there is one."""
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if datetime_cols:
        df = df.set_index(datetime_cols[0]).sort_index()
    return df


@st.cache_data(show_spinner=False)
def read_csv_url(url):
    """Read a CSV from a URL (cached so reruns don't re-download it)."""
    return pd.read_csv(url)


def load_data(source_type, uploaded_file=None, file_url=None):
    """Load data with proper type handling."""
    try:
        if source_type == "upload" and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif source_type == "github" and file_url:
            df = read_csv_url(file_url).copy()
        else:
            return None

        df = fix_dtypes(df)
        return set_datetime_index(df)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def get_dtypes_info(df):
    """Get DataFrame data types information."""
    dtypes_info = pd.DataFrame({
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Sample Values': [str(df[col].iloc[0]) if len(df) > 0 else None for col in df.columns]
    })
    return dtypes_info


@st.cache_data(ttl=3600, show_spinner=False)
def _list_github_csvs(api_url, branch):
    """Call the GitHub contents API (cached; raises on failure so errors aren't cached)."""
    response = requests.get(api_url, params={'ref': branch} if branch else None, timeout=15)
    response.raise_for_status()
    return [
        (item['name'], item['download_url'])
        for item in response.json()
        if item.get('type') == 'file' and item['name'].lower().endswith('.csv')
    ]


def get_github_files(repo_url):
    """
    Fetch CSV files from a GitHub repository folder using the GitHub contents API.

    Args:
        repo_url (str): URL like https://github.com/<owner>/<repo>/tree/<branch>/<path>

    Returns:
        list: List of tuples containing (filename, raw_url)
    """
    match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)/?(.*))?",
        repo_url.strip().rstrip('/')
    )
    if not match:
        st.error("Enter a URL like https://github.com/<owner>/<repo>/tree/<branch>/<folder>")
        return []

    owner, repo, branch, path = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path or ''}"

    try:
        csv_files = _list_github_csvs(api_url, branch)
        if not csv_files:
            st.warning("No CSV files found at that location.")
        return csv_files
    except Exception as e:
        st.error(f"Error fetching GitHub files: {str(e)}")
        return []


@st.cache_data(show_spinner=False)
def _read_example_csv(url):
    df = pd.read_csv(url, index_col=0, header=[0, 1])['Close']
    df.index = pd.to_datetime(df.index)
    df.columns.name = None
    return df.sort_index()


def load_example_data(url):
    """
    Load the example yfinance dataset (closing prices for several tickers).

    Args:
        url (str): URL of the example dataset

    Returns:
        pd.DataFrame: Loaded dataframe or None if error
    """
    try:
        return _read_example_csv(url).copy()
    except Exception as e:
        st.error(f"Error loading example data: {str(e)}")
        return None

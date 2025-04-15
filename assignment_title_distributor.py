import streamlit as st
import pandas as pd
import numpy as np
import io
import random
from typing import List, Tuple, Dict, Any, Optional
import base64
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="Assignment Title Distributor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to improve UI appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #0277BD;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #43A047;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FFA000;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #E53935;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #BBDEFB;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2196F3 !important;
        color: white !important;
    }
    /* Additional styling for file uploader */
    .css-1v0mbdj.e115fcil1 {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
def generate_random_assignment_order(total_assignments: int) -> List[int]:
    """Generate a random order for assignments."""
    assignment_indices = list(range(total_assignments))
    random.shuffle(assignment_indices)
    return assignment_indices

def identify_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Try to identify which columns are for student names, IDs, and assignment titles.
    Returns column names for (name_col, id_col, title_col)
    """
    # Heuristics for identification
    possible_name_cols = []
    possible_id_cols = []
    possible_title_cols = []
    
    for col in df.columns:
        # Check column header names
        col_lower = str(col).lower()
        
        # Name column heuristics
        if any(name_keyword in col_lower for name_keyword in ['name', 'student', 'person']):
            possible_name_cols.append(col)
        
        # ID column heuristics
        if any(id_keyword in col_lower for id_keyword in ['id', 'number', 'roll', 'enrollment']):
            possible_id_cols.append(col)
        
        # Title column heuristics
        if any(title_keyword in col_lower for title_keyword in ['title', 'assignment', 'topic', 'project']):
            possible_title_cols.append(col)
    
    # If heuristics didn't work, try to identify by data types and content
    if not possible_name_cols:
        for col in df.columns:
            # Names are usually strings with multiple characters and spaces
            if df[col].dtype == 'object' and df[col].str.contains(' ').any():
                possible_name_cols.append(col)
    
    if not possible_id_cols:
        for col in df.columns:
            # IDs are usually numeric or alphanumeric, no spaces
            if df[col].dtype in ['int64', 'float64'] or \
               (df[col].dtype == 'object' and not df[col].str.contains(' ').any()):
                possible_id_cols.append(col)
    
    if not possible_title_cols:
        for col in df.columns:
            # Titles are usually longer strings
            if df[col].dtype == 'object' and df[col].astype(str).str.len().mean() > 10:
                possible_title_cols.append(col)
    
    # Default to first three columns if can't identify
    if len(df.columns) >= 3:
        return (
            possible_name_cols[0] if possible_name_cols else df.columns[0],
            possible_id_cols[0] if possible_id_cols else df.columns[1],
            possible_title_cols[0] if possible_title_cols else df.columns[2]
        )
    else:
        st.error("The uploaded file doesn't have enough columns.")
        return df.columns[0], df.columns[0], df.columns[0]

def extract_extra_titles(df: pd.DataFrame, name_col: str, id_col: str, title_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """Extract rows with 'EXTRA' in name or ID column and return regular rows and extra titles."""
    # Find rows with 'EXTRA' in the name or ID column
    extra_mask = df[name_col].astype(str).str.contains('EXTRA', case=False, na=False) | \
                df[id_col].astype(str).str.contains('EXTRA', case=False, na=False)
    
    # Extract extra titles
    extra_titles = df.loc[extra_mask, title_col].tolist()
    
    # Get regular student rows
    regular_df = df.loc[~extra_mask].copy()
    
    return regular_df, extra_titles

def randomize_assignments(df: pd.DataFrame, extra_titles: List[str], name_col: str, id_col: str, title_col: str) -> pd.DataFrame:
    """
    Randomly redistribute assignment titles.
    Returns new DataFrame with randomized assignments.
    """
    # Get all titles including student titles and extra titles
    all_titles = df[title_col].tolist() + extra_titles
    
    # Generate random order
    random_indices = generate_random_assignment_order(len(all_titles))
    random_titles = [all_titles[i] for i in random_indices]
    
    # Create new DataFrame
    result_df = df.copy()
    result_df[title_col] = random_titles[:len(df)]
    
    # If there are remaining titles (more titles than students), add them as new rows
    remaining_titles = random_titles[len(df):]
    if remaining_titles:
        extra_rows = pd.DataFrame({
            name_col: ['EXTRA'] * len(remaining_titles),
            id_col: ['EXTRA'] * len(remaining_titles),
            title_col: remaining_titles
        })
        result_df = pd.concat([result_df, extra_rows], ignore_index=True)
    
    return result_df

def download_link(df: pd.DataFrame, filename: str, text: str, excel: bool = True) -> str:
    """Generate a download link for the DataFrame."""
    if excel:
        # Generate Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Randomized Assignments')
            # Auto-adjust columns' width
            worksheet = writer.sheets['Randomized Assignments']
            for i, col in enumerate(df.columns):
                column_width = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, column_width)
        
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
    else:
        # Generate CSV file
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'data:file/csv;base64,{b64}'
    
    return f'<a href="{href}" download="{filename}" class="download-button">{text}</a>'

def analyze_distribution(original_df: pd.DataFrame, randomized_df: pd.DataFrame, 
                       title_col: str) -> Tuple[float, Dict[str, int]]:
    """
    Analyze how well the titles were distributed.
    Returns: (percentage_changed, position_changes)
    """
    # Calculate percentage of titles that changed position
    original_titles = original_df[title_col].tolist()
    randomized_titles = randomized_df[title_col].tolist()
    
    changes = 0
    position_changes = {}
    
    for i, (orig, rand) in enumerate(zip(original_titles, randomized_titles)):
        if orig != rand:
            changes += 1
            
            # Find where the original title went
            if rand in original_titles:
                orig_pos = original_titles.index(rand)
                shift = abs(orig_pos - i)
                position_changes[rand] = shift
    
    percentage_changed = (changes / len(original_titles)) * 100
    
    return percentage_changed, position_changes

def create_wordcloud(df: pd.DataFrame, title_col: str) -> plt.Figure:
    """Create a word cloud visualization based on assignment titles."""
    # Combine all titles into one text
    text = ' '.join(df[title_col].astype(str).tolist())
    
    # Create the wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud of Assignment Titles', fontsize=16)
    
    return fig

def calculate_title_statistics(original_df: pd.DataFrame, randomized_df: pd.DataFrame, title_col: str) -> Dict[str, float]:
    """
    Calculate various statistics about the titles.
    Returns dictionary with percentage overlap and uniqueness metrics.
    """
    original_titles = original_df[title_col].astype(str).tolist()
    randomized_titles = randomized_df[title_col].astype(str).tolist()
    
    # Calculate overlap percentage (titles that didn't change)
    overlap_count = sum(1 for orig, rand in zip(original_titles, randomized_titles) if orig == rand)
    overlap_percentage = (overlap_count / len(original_titles)) * 100
    
    # Calculate uniqueness percentage
    unique_titles = len(set(original_titles))
    uniqueness_percentage = (unique_titles / len(original_titles)) * 100
    
    # Count word frequency
    all_words = []
    for title in original_titles:
        all_words.extend([word.lower() for word in title.split() if len(word) > 3])
    
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(5)
    
    return {
        'overlap_percentage': overlap_percentage,
        'uniqueness_percentage': uniqueness_percentage,
        'most_common_words': most_common_words
    }

def main():
    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            "Main Menu", 
            ["Home", "Assignment Distributor", "About"], 
            icons=["house", "shuffle", "info-circle"], 
            menu_icon="cast",
            default_index=1,
            styles={
                "container": {"padding": "5!important", "background-color": "#f0f2f6"},
                "icon": {"color": "#1E88E5", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#e3f2fd"},
                "nav-link-selected": {"background-color": "#1E88E5"},
            }
        )
    
    if selected == "Home":
        st.markdown('<h1 class="main-header">Assignment Title Distributor</h1>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        Welcome to the Assignment Title Distributor tool! This application helps educators fairly distribute 
        assignment titles among students by randomizing the assignments.
        
        ### Why randomize assignments?
        
        - **Fairness**: Ensures that assignments are not biased based on student ID or position in the roster
        - **Equal difficulty distribution**: Prevents students with specific ID ranges from always getting easier or harder assignments
        - **Reduced bias**: Eliminates any perception of favoritism in assignment distribution
        
        ### Key Features
        
        - Upload assignment sheets in Excel format
        - Automatically detect student names, IDs, and assignment titles
        - Identify extra assignments marked with "EXTRA"
        - Randomly redistribute assignments while preserving student order
        - Download the newly organized assignment sheet
        - Analyze the distribution's effectiveness
        
        Navigate to the **Assignment Distributor** section to get started!
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # No image here as requested

    elif selected == "Assignment Distributor":
        st.markdown('<h1 class="main-header">Assignment Title Distributor</h1>', unsafe_allow_html=True)
        
        # File upload section
        st.markdown('<h2 class="sub-header">Step 1: Upload Your Assignment Sheet</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        Upload your Excel file containing:
        - Student names
        - Student ID numbers
        - Assignment titles
        
        **Note**: If you have extra assignment titles, mark them with 'EXTRA' in the name or ID column.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            # Read the file
            try:
                df = pd.read_excel(uploaded_file)
                
                # Store original dataframe in session state
                st.session_state.original_df = df.copy()
                
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.write(f"✅ File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display first few rows
                st.markdown('<h3 class="section-header">Preview of Uploaded Data</h3>', unsafe_allow_html=True)
                st.dataframe(df.head(5), use_container_width=True)
                
                # Column identification
                st.markdown('<h2 class="sub-header">Step 2: Identify Data Columns</h2>', unsafe_allow_html=True)
                
                # Try to automatically identify columns
                name_col, id_col, title_col = identify_columns(df)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    name_col = st.selectbox(
                        "Select Student Name Column",
                        options=df.columns.tolist(),
                        index=df.columns.tolist().index(name_col)
                    )
                
                with col2:
                    id_col = st.selectbox(
                        "Select Student ID Column",
                        options=df.columns.tolist(),
                        index=df.columns.tolist().index(id_col)
                    )
                
                with col3:
                    title_col = st.selectbox(
                        "Select Assignment Title Column",
                        options=df.columns.tolist(),
                        index=df.columns.tolist().index(title_col)
                    )
                
                # Extract extra titles
                regular_df, extra_titles = extract_extra_titles(df, name_col, id_col, title_col)
                
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.write(f"Found {len(regular_df)} regular student assignments and {len(extra_titles)} extra titles.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Randomization options
                st.markdown('<h2 class="sub-header">Step 3: Randomization Options</h2>', unsafe_allow_html=True)
                
                seed = st.number_input("Random Seed (for reproducibility)", min_value=0, max_value=10000, value=42)
                random.seed(seed)
                
                shuffle_button = st.button("Randomize Assignments", type="primary")
                
                if shuffle_button:
                    # Randomize assignments
                    randomized_df = randomize_assignments(regular_df, extra_titles, name_col, id_col, title_col)
                    
                    # Store randomized df in session state
                    st.session_state.randomized_df = randomized_df
                    
                    st.markdown('<h2 class="sub-header">Step 4: Review Randomized Assignments</h2>', unsafe_allow_html=True)
                    
                    # Display tabs for original vs randomized
                    tab1, tab2 = st.tabs(["Original Assignments", "Randomized Assignments"])
                    
                    with tab1:
                        st.dataframe(df, use_container_width=True)
                    
                    with tab2:
                        st.dataframe(randomized_df, use_container_width=True)
                    
                    # Analysis of distribution
                    percentage_changed, position_changes = analyze_distribution(
                        regular_df, randomized_df, title_col
                    )
                    
                    st.markdown('<h2 class="sub-header">Step 5: Distribution Analysis</h2>', unsafe_allow_html=True)
                    
                    # Calculate title statistics
                    title_stats = calculate_title_statistics(regular_df, randomized_df, title_col)
                    
                    # Display metrics in a 2x2 grid
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.metric("Assignments Changed", f"{percentage_changed:.1f}%")
                        st.write(f"Out of {len(regular_df)} assignments, {int(percentage_changed * len(regular_df) / 100)} were changed.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        avg_shift = sum(position_changes.values()) / len(position_changes) if position_changes else 0
                        st.metric("Average Position Shift", f"{avg_shift:.1f} positions")
                        st.write(f"Titles moved an average of {avg_shift:.1f} positions from their original place.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.metric("Title Overlap", f"{title_stats['overlap_percentage']:.1f}%")
                        st.write(f"Percentage of titles that remained in the same position.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.metric("Unique Titles", f"{title_stats['uniqueness_percentage']:.1f}%")
                        st.write(f"Percentage of titles that are unique in the dataset.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add word cloud visualization
                    st.markdown('<h3 class="section-header">Word Cloud of Assignment Titles</h3>', unsafe_allow_html=True)
                    st.pyplot(create_wordcloud(randomized_df, title_col))
                    
                    # Download options
                    st.markdown('<h2 class="sub-header">Step 6: Download Randomized Assignments</h2>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        excel_download = download_link(
                            randomized_df, 
                            "randomized_assignments.xlsx", 
                            "📥 Download as Excel", 
                            excel=True
                        )
                        st.markdown(excel_download, unsafe_allow_html=True)
                    
                    with col2:
                        csv_download = download_link(
                            randomized_df, 
                            "randomized_assignments.csv", 
                            "📥 Download as CSV", 
                            excel=False
                        )
                        st.markdown(csv_download, unsafe_allow_html=True)
                    
                    # Show statistics
                    with st.expander("View Detailed Distribution Statistics"):
                        # Calculate title length distribution
                        title_lengths = randomized_df[title_col].astype(str).apply(len)
                        
                        stats_col1, stats_col2 = st.columns(2)
                        
                        with stats_col1:
                            st.metric("Shortest Title Length", f"{title_lengths.min()} chars")
                            st.metric("Longest Title Length", f"{title_lengths.max()} chars")
                        
                        with stats_col2:
                            st.metric("Average Title Length", f"{title_lengths.mean():.1f} chars")
                            st.metric("Median Title Length", f"{title_lengths.median():.1f} chars")
                        
                        # Most common words
                        st.subheader("Most Common Words in Titles")
                        for word, count in title_stats['most_common_words']:
                            st.write(f"• **{word}**: appears {count} times")
                        
            except Exception as e:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error(f"Error processing the file: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected == "About":
        st.markdown('<h1 class="main-header">About Assignment Title Distributor</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Purpose
        
        This application was developed to help educators fairly distribute assignment titles among students. 
        Traditional assignment distribution methods often follow a sequential pattern, which can result in 
        students with specific ID ranges (typically at the beginning or end) consistently receiving easier 
        or more difficult assignments.
        
        ### How It Works
        
        1. **Upload**: You upload your Excel sheet containing student information and assignment titles.
        2. **Identify**: The app identifies which columns contain student names, IDs, and assignment titles.
        3. **Extract**: The app separates regular assignments from extra titles marked with 'EXTRA'.
        4. **Randomize**: The app shuffles the assignment titles while maintaining the original student order.
        5. **Analyze**: The app provides statistics on how effectively the titles were redistributed.
        6. **Download**: You can download the randomized assignments in Excel or CSV format.
        
        ### Features
        
        - **Intelligent Column Detection**: Automatically identifies the purpose of each column.
        - **Extra Title Handling**: Properly processes any additional titles marked with 'EXTRA'.
        - **Distribution Analysis**: Provides metrics on how effectively the assignments were shuffled.
        - **Multiple Download Formats**: Download your randomized assignments in Excel or CSV format.
        
        ### Privacy
        
        This application processes all data in your browser. No data is stored on any server.
        
        ### Feedback
        
        Have suggestions for improvement? Please reach out to us!
        
        ### Version
        
        Version 1.0.0
        """)

if __name__ == "__main__":
    main()

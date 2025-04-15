# Assignment-Title-Distributor
# Assignment Title Distributor

## Overview
The Assignment Title Distributor is a Streamlit web application designed to help educators fairly distribute assignment titles among students. It addresses the common issue where students with specific ID numbers (typically at the beginning or end of the roster) may consistently receive easier or more difficult assignments.

By randomly redistributing assignment titles while maintaining student order, the app ensures fairness and equal difficulty distribution across all students.

![Assignment Title Distributor](https://raw.githubusercontent.com/yourusername/assignment-title-distributor/main/screenshot.png)

## Features

- **Intuitive User Interface**: Clean, organized, and responsive design with a three-section layout
- **Smart Column Detection**: Automatically identifies student names, IDs, and assignment titles
- **Extra Title Handling**: Properly processes additional titles marked with "EXTRA"
- **Flexible File Support**: Works with Excel files (.xlsx, .xls)
- **Word Cloud Visualization**: Visual representation of common terms in assignment titles
- **Title Statistics**: Shows overlap percentage and uniqueness of assignment titles
- **Distribution Analysis**: Provides metrics on how effectively assignments were shuffled
- **Random Seed Option**: Set a seed value for reproducible randomization
- **Detailed Statistics**: Analyzes title length distribution and most common words
- **Download Options**: Export randomized assignments as Excel or CSV

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/assignment-title-distributor.git
   cd assignment-title-distributor
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage Guide

### Step 1: Upload Your Excel File
- Click on the file upload area to select your Excel file
- The file should contain columns for student names, ID numbers, and assignment titles
- If you have extra assignment titles, mark them with 'EXTRA' in the name or ID column

### Step 2: Identify Data Columns
- The app will attempt to automatically identify which columns contain student names, IDs, and titles
- You can adjust these selections using the dropdown menus if needed

### Step 3: Set Randomization Options
- Optionally set a random seed for reproducible results
- Click the "Randomize Assignments" button

### Step 4: Review Randomized Assignments
- Compare the original and randomized assignments using the tabs
- Verify that the distribution meets your requirements

### Step 5: Analyze the Distribution
- Review statistics about how many assignments changed and how far they moved
- Examine the word cloud visualization of title keywords
- Check title overlap percentage and uniqueness percentage

### Step 6: Download Results
- Download the randomized assignments as an Excel or CSV file

## Example Excel Format

Your Excel file should follow this general structure:

| Student Name     | Student ID | Assignment Title                |
|------------------|------------|--------------------------------|
| John Smith       | 10001      | Introduction to Data Structures |
| Maria Garcia     | 10002      | Algorithm Analysis              |
| Samuel Johnson   | 10003      | Database Design Principles      |
| EXTRA            | EXTRA      | Extra Assignment Title 1        |
| EXTRA            | EXTRA      | Extra Assignment Title 2        |

The rows marked with "EXTRA" will be treated as additional titles that can be assigned to students.

## Key Metrics

The app provides several metrics to help you understand the distribution:

- **Assignments Changed**: Percentage of assignments that were moved to different students
- **Average Position Shift**: How far, on average, assignments moved from their original position
- **Title Overlap**: Percentage of titles that remained with the same student
- **Unique Titles**: Percentage of titles that are unique in the dataset

## Word Cloud Visualization

The app generates a word cloud visualization based on the words found in your assignment titles. This helps identify common themes and topics in your assignments.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Openpyxl
- XlsxWriter
- Streamlit-option-menu
- Plotly
- WordCloud
- Matplotlib

## Troubleshooting

### Common Issues

1. **File Upload Errors**
   - Make sure your Excel file is not corrupted
   - Ensure the file has the correct columns for student names, IDs, and assignment titles

2. **Column Detection Issues**
   - If automatic column detection fails, manually select the correct columns
   - Ensure your column headers are descriptive (e.g., "Student Name", "ID", "Assignment Title")

3. **Extra Titles Not Recognized**
   - Make sure rows with extra titles have "EXTRA" in both the name and ID columns

### Support

If you encounter any issues or have questions about the application, please open an issue on the GitHub repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

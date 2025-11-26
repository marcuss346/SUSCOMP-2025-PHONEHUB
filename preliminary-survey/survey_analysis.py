import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('survey.csv')

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Extract written answers for later analysis
written_answers = []

# Function to collect written answers


def collect_written_answers(row):
    for col in df.columns:
        if isinstance(row[col], str) and len(row[col]) > 50:  # Longer text responses
            written_answers.append({
                'column': col,
                'answer': row[col],
                'timestamp': row['Časovni žig'] if 'Časovni žig' in row else 'Unknown'
            })


# Apply the function to each row
df.apply(collect_written_answers, axis=1)

# 1. DEMOGRAPHIC ANALYSIS
print("\n" + "="*50)
print("DEMOGRAPHIC ANALYSIS")
print("="*50)

# Age distribution
plt.figure(figsize=(15, 12))

plt.subplot(3, 3, 1)
age_data = df['Koliko ste stari?\nHow old are you?'].dropna()
# Convert age to numeric, handling '26+' values
age_numeric = []
for age in age_data:
    if str(age).endswith('+'):
        age_numeric.append(26)
    else:
        try:
            age_numeric.append(int(age))
        except:
            age_numeric.append(np.nan)
age_numeric = [x for x in age_numeric if not np.isnan(x)]
plt.hist(age_numeric, bins=range(19, 31), edgecolor='black', alpha=0.7)
plt.title('Age Distribution of Respondents')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(range(19, 31))

# Faculty distribution
plt.subplot(3, 3, 2)
faculty_cols = [col for col in df.columns if 'fakultet' in col.lower()
                or 'faculty' in col.lower()]
faculty_data = []
for col in faculty_cols:
    faculty_data.extend(df[col].dropna().tolist())

faculty_clean = [f for f in faculty_data if pd.notna(f) and f != '']
faculty_series = pd.Series(faculty_clean)
faculty_counts = faculty_series.value_counts().head(8)  # Top 8 faculties

plt.pie(faculty_counts.values, labels=faculty_counts.index, autopct='%1.1f%%')
plt.title('Faculty Distribution')

# Study year distribution
plt.subplot(3, 3, 3)
year_cols = [col for col in df.columns if 'letnik' in col.lower()
             or 'year' in col.lower()]
year_data = []
for col in year_cols:
    year_data.extend(df[col].dropna().tolist())

year_clean = [y for y in year_data if pd.notna(
    y) and y != '' and 'letnik' in str(y).lower()]
year_series = pd.Series(year_clean)
year_counts = year_series.value_counts()

plt.bar(range(len(year_counts)), year_counts.values)
plt.title('Study Year Distribution')
plt.xlabel('Study Year')
plt.ylabel('Count')
plt.xticks(range(len(year_counts)), year_counts.index, rotation=45)

# 2. PHONE MODEL ANALYSIS
print("\n" + "="*50)
print("PHONE MODEL ANALYSIS")
print("="*50)

# Phone brands
plt.subplot(3, 3, 4)
brand_cols = [col for col in df.columns if 'znamk' in col.lower()
              or 'brand' in col.lower()]
brand_data = []
for col in brand_cols:
    brand_data.extend(df[col].dropna().tolist())

brand_clean = [b for b in brand_data if pd.notna(
    b) and b != '' and b != 'Drugo (Other)']
brand_series = pd.Series(brand_clean)
brand_counts = brand_series.value_counts().head(10)

plt.barh(range(len(brand_counts)), brand_counts.values)
plt.title('Top 10 Phone Brands')
plt.xlabel('Count')
plt.yticks(range(len(brand_counts)), brand_counts.index)

# Price distribution
plt.subplot(3, 3, 5)
price_col = [col for col in df.columns if 'cen' in col.lower()
             or 'price' in col.lower()][0]
price_data = df[price_col].dropna()
price_counts = price_data.value_counts()

plt.bar(range(len(price_counts)), price_counts.values)
plt.title('Phone Price Distribution')
plt.xlabel('Price Range (€)')
plt.ylabel('Count')
plt.xticks(range(len(price_counts)), price_counts.index, rotation=45)

# Purchase year analysis
plt.subplot(3, 3, 6)
purchase_col = [col for col in df.columns if 'kupili' in col.lower()
                or 'bought' in col.lower()][0]
purchase_data = df[purchase_col].dropna()

# Extract years from purchase data
years = []
for text in purchase_data:
    if '2020' in str(text):
        years.append(2020)
    elif '2021' in str(text):
        years.append(2021)
    elif '2022' in str(text):
        years.append(2022)
    elif '2023' in str(text):
        years.append(2023)
    elif '2024' in str(text):
        years.append(2024)
    elif '2025' in str(text):
        years.append(2025)

year_counts = pd.Series(years).value_counts().sort_index()
plt.plot(year_counts.index, year_counts.values, marker='o')
plt.title('Phone Purchase Year Distribution')
plt.xlabel('Year')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

# 3. PHONE LIFECYCLE AND DISPOSAL ANALYSIS
print("\n" + "="*50)
print("PHONE LIFECYCLE AND DISPOSAL ANALYSIS")
print("="*50)

# Replacement plans
plt.subplot(3, 3, 7)
replace_col = [col for col in df.columns if 'namen zamenjati' in col.lower(
) or 'plan to replace' in col.lower()][0]
replace_data = df[replace_col].dropna()
replace_counts = replace_data.value_counts().head(6)

plt.bar(range(len(replace_counts)), replace_counts.values)
plt.title('Phone Replacement Plans')
plt.xlabel('Replacement Timeline')
plt.ylabel('Count')
plt.xticks(range(len(replace_counts)), [str(x)[
           :20] + '...' if len(str(x)) > 20 else str(x) for x in replace_counts.index], rotation=45)

# Reasons for replacement
plt.subplot(3, 3, 8)
reason_col = [col for col in df.columns if 'zakaj' in col.lower(
) and 'namen' in col.lower() or 'why' in col.lower() and 'plan' in col.lower()][0]
reason_data = df[reason_col].dropna()

# Categorize reasons
replacement_reasons = {
    'Performance/Slowness': ['počas', 'slow', 'outdated', 'zastarel'],
    'Battery': ['baterij', 'battery'],
    'Damage': ['poškod', 'damage', 'pokvaril'],
    'New Model': ['novejš', 'newer model', 'želim nov'],
    'Other': []  # Catch-all
}

reason_categories = []
for reason in reason_data:
    reason_str = str(reason).lower()
    categorized = False
    for category, keywords in replacement_reasons.items():
        if any(keyword in reason_str for keyword in keywords):
            reason_categories.append(category)
            categorized = True
            break
    if not categorized:
        reason_categories.append('Other')

reason_counts = pd.Series(reason_categories).value_counts()
plt.pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%')
plt.title('Reasons for Phone Replacement')

# Phone disposal plans
plt.subplot(3, 3, 9)
disposal_col = [col for col in df.columns if 'kam ga boste' in col.lower(
) or 'what will you do' in col.lower()][0]
disposal_data = df[disposal_col].dropna()
disposal_counts = disposal_data.value_counts().head(6)

plt.bar(range(len(disposal_counts)), disposal_counts.values)
plt.title('Phone Disposal Plans')
plt.xlabel('Disposal Method')
plt.ylabel('Count')
plt.xticks(range(len(disposal_counts)), [str(x)[
           :15] + '...' if len(str(x)) > 15 else str(x) for x in disposal_counts.index], rotation=45)

plt.tight_layout()
plt.show()

# Additional detailed analysis
print("\n" + "="*50)
print("DETAILED STATISTICS")
print("="*50)

# Brand popularity by faculty
print("\nTop Brands by Faculty:")
faculty_brand_data = []
for idx, row in df.iterrows():
    faculty = None
    brand = None

    # Get faculty
    for col in faculty_cols:
        if pd.notna(row[col]) and row[col] != '':
            faculty = row[col]
            break

    # Get brand
    for col in brand_cols:
        if pd.notna(row[col]) and row[col] != '' and row[col] != 'Drugo (Other)':
            brand = row[col]
            break

    if faculty and brand:
        faculty_brand_data.append({'faculty': faculty, 'brand': brand})

faculty_brand_df = pd.DataFrame(faculty_brand_data)
if not faculty_brand_df.empty:
    top_faculty_brands = faculty_brand_df.groupby('faculty')['brand'].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else 'None')
    print(top_faculty_brands.head())

# Phone age analysis
print(f"\nAverage respondent age: {np.mean(age_numeric):.1f} years")
print(f"Most common age: {pd.Series(age_numeric).mode().iloc[0]} years")

# Replacement timeline analysis
replacement_timeline = {
    'Immediate (within 6 months)': ['6 mesecih', '6 months', 'naslednjih 6'],
    'Short term (within 1 year)': ['naslednjem letu', 'next year'],
    'Medium term (1-3 years)': ['2 leti', '3 leta', '2 years', '3 years'],
    'Long term (3+ years)': ['4 leta', '5 leta', '4 years', '5 years'],
    'No plans': ['nimam namena', "don't plan", 'dokler deluje']
}

timeline_categories = []
for plan in replace_data:
    plan_str = str(plan).lower()
    for category, keywords in replacement_timeline.items():
        if any(keyword in plan_str for keyword in keywords):
            timeline_categories.append(category)
            break
    else:
        timeline_categories.append('Other')

timeline_counts = pd.Series(timeline_categories).value_counts()
print("\nReplacement Timeline Categories:")
print(timeline_counts)

# Display written answers for manual analysis
print("\n" + "="*50)
print("WRITTEN ANSWERS FOR MANUAL ANALYSIS")
print("="*50)
print(f"Total written answers collected: {len(written_answers)}")

if written_answers:
    print("\nSample of written answers:")
    for i, answer in enumerate(written_answers[:5]):  # Show first 5
        print(f"\n--- Answer {i+1} ---")
        print(f"Column: {answer['column']}")
        print(f"Answer: {answer['answer'][:200]}...")  # First 200 characters

    # Save written answers to file for further analysis
    written_df = pd.DataFrame(written_answers)
    written_df.to_csv('written_answers_analysis.csv',
                      index=False, encoding='utf-8')
    print(f"\nAll written answers saved to 'written_answers_analysis.csv'")
else:
    print("No substantial written answers found.")

# Create a summary statistics table
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

summary_stats = {
    'Total Respondents': len(df),
    'Average Age': f"{np.mean(age_numeric):.1f} years",
    'Most Common Brand': brand_counts.index[0] if len(brand_counts) > 0 else 'N/A',
    'Most Common Price Range': price_counts.index[0] if len(price_counts) > 0 else 'N/A',
    'Most Common Replacement Plan': replace_counts.index[0] if len(replace_counts) > 0 else 'N/A',
    'Phones to be Kept as Backup': disposal_counts.get('Obdržal_a ga bom kot rezervni telefon (Keep it as a backup phone)', 0)
}

for stat, value in summary_stats.items():
    print(f"{stat}: {value}")

print("\nAnalysis complete! Check the generated graphs and statistics above.")

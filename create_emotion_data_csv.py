import pandas as pd

# Sample data (replace this with your actual data or other methods to generate it)
data = {
    'text': ['I am happy', 'I am sad', 'I am excited', 'I feel down', 'I am joyful'],
    'emotion': ['happy', 'sad', 'excited', 'sad', 'happy']
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
df.to_csv('emotion_data.csv', index=False)

print("emotion_data.csv file created successfully!")

from bs4 import BeautifulSoup

# Replace 'your_file.html' with the path to your Project Gutenberg HTML file
file_path = 'data/pg2370-images.html'

with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, 'html.parser')

# Extract all text from the HTML file
text = soup.get_text()

# Optionally, save the extracted text to a file
with open('extracted_text.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(text)

print("Text extracted and saved to extracted_text.txt")

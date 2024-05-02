import sqlite3
import json

# Connect to an SQLite database (the database file is located at the path 'your-database-name.db')
connection = sqlite3.connect('data/nep_dict.sqlite3')

# Create a cursor object using the connection
cursor = connection.cursor()

# Execute a query
cursor.execute("SELECT * FROM word")
#cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

# Fetch and print all rows from the query
vocabulary = []
rows = cursor.fetchall()
for row in rows:
    print(row)
    vocabulary.append(row[1])
with open("nep_dict_supp.json", "w") as outj:
    json.dump(vocabulary, outj, indent=4, ensure_ascii=False)

# Don't forget to close the connection
connection.close()

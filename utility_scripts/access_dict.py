import sqlite3
import json

# Connect to an SQLite database (the database file is located at the path 'your-database-name.db')
connection = sqlite3.connect('data/nep_dict.sqlite3')

# Create a cursor object using the connection
cursor = connection.cursor()

# Execute a query
query = "PRAGMA table_info(definition);"
query = """
SELECT word.value, definition.value
FROM word
JOIN definition ON word.id = definition.word_id
"""
#cursor.execute("SELECT * FROM word")
cursor.execute(query)
#cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

#print(cursor.fetchall())

# Fetch and print all rows from the query
vocabulary = {}
rows = cursor.fetchall()
for row in rows:
    if row[0] not in vocabulary:
        vocabulary[row[0]] = [row[1]]
    else:
        vocabulary[row[0]].append(row[1])
with open("nep_supp_wdef.json", "w") as outj:
    json.dump(vocabulary, outj, indent=4, ensure_ascii=False)

# Don't forget to close the connection
connection.close()

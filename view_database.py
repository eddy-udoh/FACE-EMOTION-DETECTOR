import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()

print("="*80)
print("DATABASE CONTENTS:")
print("="*80)

for row in rows:
    print(f"\nID: {row[0]}")
    print(f"Name: {row[1]}")
    print(f"Email: {row[2]}")
    print(f"Age: {row[3]}")
    print(f"Emotion: {row[4]}")
    print(f"Image Path: {row[5]}")
    print(f"Timestamp: {row[6]}")
    print("-"*80)

conn.close()
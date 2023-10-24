import mysql.connector
import pandas as pd

mydb = mysql.connector.connect(
    host="",
    user="",
    password="",
    database=""
)
query="""
    SELECT exhibition_title,exhibition_contents,exhibition_img
    FROM exhibition_board
"""
cursor = mydb.cursor()
cursor.execute(query)
rows = cursor.fetchall()
mlist = []
for i in rows:
    d = {
        "exhibition_title": i[0],
        "exhibition_contents": i[1],
        "exhibition_img": i[2]
     }
    mlist.append(d)
exh_df = pd.DataFrame(mlist)




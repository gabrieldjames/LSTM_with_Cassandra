#Create comment based on post comments, insert into generated_comments table
from get_comments import get_post_comments
from comment_generator import run_lstm
from db_connection import Connection
import datetime
import pandas as pandas
import sys



post_id = str(sys.argv[1])

df = get_comments(post_id)

new_comment, generated_seed = run_lstm(df_comments, maxlen = 15, step = 3, epoch_num = 5, train = True, save = False )
d_now = datetime.datetime.now()

q_new_comment = """
				 insert into askreddit.generated_comments(post_id, lstm_date, seed, generated_comment)
				 values (?, ?, ?, ?)
				"""


q_comment_prepped = connection.session.execute(q_new_comment, (df['post_id'].iloc[0], d_now, generated_seed))

print('Inserted generated comment')

connection.close()


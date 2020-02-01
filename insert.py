### insert Reddit comments into Cassandra via batches
import pandas as pd
from progressbar import Percentage, ProgressBar, Bar, ETA
from db_connection import Connection
from data_collection import pull_Reddit_Posts
from comment_generator import run_lstm
import datetime
datetime.datetime.now()


#pull reddit comments
df_full = pull_Reddit_Posts(subreddit = 'askreddit', num_posts = 1, comment_lim = None)

#create dataframes of comments and posts
df_comments = df_full[['c_id', 'u_id', 'c_timestamp', 'comment', 'title', 'p_id', 'p_timestamp']]
df_posts = df_full[['p_id', 'p_timestamp', 'title', 'score', 'url', 'num_comments']].drop_duplicates()

#establish connection to Cassandra, create prepared inserts
connection = Connection()
q_comments = """
		      begin batch

		      insert into askreddit.comments_by_post(comment_id, comment_user_id, comment_date, comment, title, post_id, post_date) values (?, ?, ?, ?, ?, ?, ?);
		      insert into askreddit.comments_by_user(comment_id, comment_user_id, comment_date, comment, title, post_id, post_date) values (?, ?, ?, ?, ?, ?, ?);

		      apply batch
	         """

q_posts = "insert into askreddit.posts(post_id, post_date, title, upvotes, url, comment_count) values (?, ?, ?, ?, ?, ?);"
q_uc    = "update askreddit.user_comments_log set comment_count = comment_count + 1 where comment_user_id = ?;"
q_pc    = "update askreddit.post_comments_log set comment_count = comment_count + 1 where post_id = ?;"


comments_prepped = connection.session.prepare(q_comments)
posts_prepped = connection.session.prepare(q_posts)

user_counts_prepped = connection.session.prepare(q_uc)
post_counts_prepped = connection.session.prepare(q_pc)



#insert data into Cassandra
print()
print('Insert Comments:')
bar1 = ProgressBar(maxval = len(df_comments), widgets=[Bar('=', '[', ']'), ' ', Percentage(), '  ', ETA()]).start()

for i in range(len(df_comments)):
	
	connection.session.execute(comments_prepped, (df_comments.iloc[i,0], df_comments.iloc[i,1], df_comments.iloc[i,2], df_comments.iloc[i,3], df_comments.iloc[i,4], df_comments.iloc[i,5], df_comments.iloc[i,6], 
												  df_comments.iloc[i,0], df_comments.iloc[i,1], df_comments.iloc[i,2], df_comments.iloc[i,3], df_comments.iloc[i,4], df_comments.iloc[i,5], df_comments.iloc[i,6]))
	connection.session.execute(user_counts_prepped, (df_comments.iloc[i,1],))
	connection.session.execute(post_counts_prepped, (df_comments.iloc[i,5],))
	bar1.update(i+1)


bar1.finish()




print()
print('Insert Posts:')
bar2 = ProgressBar(maxval = len(df_posts), widgets=[Bar('=', '[', ']'), ' ', Percentage(), '  ', ETA()]).start()

for i in range(len(df_posts)):
	connection.session.execute(posts_prepped, (df_posts.iloc[i,0], df_posts.iloc[i,1], df_posts.iloc[i,2], df_posts.iloc[i,3], df_posts.iloc[i,4], df_posts.iloc[i,5]))
	bar2.update(i+1)

bar2.finish()


print()
print('Insert Complete')


print()
print('Training LSTM Model')

new_comment, generated_seed = run_lstm(df_comments, min_word_freq = 5, maxlen = 6, step = 2, epoch_num = 5, train = True)
d_now = datetime.datetime.now()

q_new_comment = """
				 insert into askreddit.generated_comments(post_id, lstm_date, seed, generated_comment)
				 values (?, ?, ?, ?)
				"""


q_comment_prepped = connection.session.execute(q_new_comment, (df['post_id'].iloc[0], d_now, generated_seed))

print('Inserted generated comment')

connection.close()

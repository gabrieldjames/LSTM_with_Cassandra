### get posts and comments from subreddit, apply basic text cleaning for LSTM
import pandas as pd
import praw
from praw.models import MoreComments
import datetime
import warnings
warnings.filterwarnings("ignore")


def pull_Reddit_Posts(subreddit, num_posts = 1, comment_lim = 100):

	print("Pulling comments from ", num_posts, " post(s) on <www.reddit.com/r/", subreddit, ">...", sep='')

	reddit = praw.Reddit(client_id='vBPwE2meEKIqMg', client_secret='BKgyF6otTVkQnrXbC_v2WPndgdk', user_agent='LSTM_Trial')

	subreddit = reddit.subreddit(subreddit)

	posts = []
	comments = []

	#get posts
	for post in subreddit.hot(limit=num_posts):
		if not post.stickied:
			posts.append([post.title, post.score, post.id, post.url, post.num_comments, datetime.datetime.fromtimestamp(post.created)])

	posts = pd.DataFrame(posts,columns=['title', 'score', 'p_id', 'url', 'num_comments', 'p_timestamp'])

	#get all comments
	for post_id in posts['p_id']:
		submission = reddit.submission(id=post_id)
		submission.comments.replace_more(limit=comment_lim)
		for comment in submission.comments.list(): #get all comments
			comments.append([post_id, comment.author, comment.id, comment.body, datetime.datetime.fromtimestamp(comment.created)])

	comments = pd.DataFrame(comments, columns=['p_id', 'u_id', 'c_id', 'comment', 'c_timestamp'])

	#join post and comment dataframes
	df = pd.merge(posts, comments, how='left', on='p_id')
	df['Time_to_Comment'] = df['c_timestamp'] - df['p_timestamp']

	#get comment length, filter out comments with less than 15 words
	df['comment_length'] = 0
	for i in range(len(df)):
		df['comment_length'].iloc[i] = len(df['comment'].iloc[i].split(' '))

	df = df[df['comment_length']>=15]


	#format columns
	df['comment'] = df['comment'].astype(str).apply(clean_text)
	df['title'] = df['title'].astype(str).apply(clean_text)

	df['p_id'] = df['p_id'].astype(str)
	df['u_id'] = df['u_id'].astype(str)
	df['c_id'] = df['c_id'].astype(str)
	df['url'] = df['url'].astype(str)


	print(len(df), " comments successfully pulled from <www.reddit.com/r/", subreddit, ">.", sep='')

	return df


def clean_text(text): #format text columns
    
    text = text.replace('\n', '')
    text = text.replace('\\', '')
    text = text.replace('"', '')
    text = text.replace("*", '')
    text = text.lower()
    return text

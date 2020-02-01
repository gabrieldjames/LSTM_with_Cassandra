#pull comments by post and by user
from db_connection import Connection
import pandas as pd

def get_post_comments(post_id):
	connection = Connection()
	query = "select title, comment from askreddit.comments_by_user where post_id = " + post_id + ";"
	df = pd.DataFrame(list(connection.session.execute(query)))
	return df

def get_user_comments(comment_user_id):
	connection = Connection()
	query = "select title, comment from askreddit.comments_by_user where comment_user_id = " + comment_user_id + ";"
	df = pd.DataFrame(list(connection.session.execute(query)))
	return df

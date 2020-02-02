from db_connection import Connection

connection = Connection()

query = """
		 truncate askreddit.comments_by_post;
		 truncate askreddit.comments_by_user;
		 truncate askreddit.user_comments_log;
		 truncate askreddit.post_comments_log;
		 truncate askreddit.posts;
		"""

print('Type "Truncate All" to truncate all tables')
check = input()
test = 'Truncate All'

if(check==test):
	connection.session.execute(query)

connection.close()
# LSTM_with_Cassandra
This set of Python scripts pulls Reddit comments from the hottest Askreddit post and inserts them into a Cassandra database via Datastax Apollo: https://www.datastax.com/constellation/datastax-apollo

An LSTM trains on these comments, post by post, and generates a comment based on some seed found in each post.  That result is uploaded to a generated comment table.


#### data_collection.py
This script extracts Reddit comments from Askreddit posts, transforms the data for improved training time for the LSTM, and outputs a dataframe.

#### comment_generation.py
This script takes in a dataframe, manipulates the Reddit comments field, and applies the LSTM model to the inputted text.  This script supports training and saving the model and training the model without saving (to enable comparing output of model for particular threads as model trains over time). 

#### insert.py
This script takes input from data_collection.py, inserts the comments and relevant counters for posts and users, and calls comment_generator.py to insert the generated comments for the post inputted to the generated_comments table.

#### generate_post_comment.py
Inserts a comment generated from a particular post into the generated_comments table.

#### get_comments.py
Provides functions for reading data from comments_by_user and comments_by_post tables

#### db_connection.py
Class for connecting to Cassandra database.

#### char_data.csv
Contains character encodings for LSTM

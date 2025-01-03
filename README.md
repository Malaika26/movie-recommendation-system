# Movie Recommendation System using Matrix Factorization and KMeans Clustering

In this notebook, we are building a movie recommendation system using Matrix Factorization and KMeans Clustering to predict movie ratings based on user preferences. This system utilizes the MovieLens dataset, which contains ratings and movie information. The goal is to predict the ratings that a user might give to a movie and group similar movies together using KMeans clustering.

## Dataset:
The dataset used is MovieLens. It contains movie ratings, user data, and movie information, and is freely available for research purposes.

## Reference: 
Harper, F. Maxwell, and Joseph A. Konstan. "The MovieLens Datasets: History and Context." ACM Transactions on Interactive Intelligent Systems (TiiS), 2015. [Link to Paper](https://doi.org/10.1145/2827872)

## Working & Setup:
- **Data Import and Preprocessing:**
Download the dataset, unzip it, and load the movies.csv and ratings.csv files into pandas DataFrames.
After loading the data, calculate some basic statistics about the data, such as the number of unique users, movies, and the full rating matrix's size.
- **Initial Data Inspection:**
Inspect the first few rows of both movies_df and ratings_df to understand their structure.
Then create a dictionary that maps movie IDs to movie titles for easy lookup.
Also calculate the number of unique users and movies in the dataset and the size of the full rating matrix.
- **Matrix Factorization Model:**
Matrix Factorization is a technique where we approximate the user-item interaction matrix by decomposing it into lower-dimensional matrices (user and item embeddings). These embeddings represent latent factors that help in predicting missing values (ratings in our case).
Implement the model using PyTorch, which allows us to take advantage of GPU acceleration for training.
- **Dataset and DataLoader:**
Create a custom PyTorch Dataset to handle the user-item pairs and corresponding ratings. This allows efficient batching of data during model training.
The Loader class maps the user IDs and movie IDs to indices, making it compatible with the embedding layers in our model.
- **Training:**
The model is trained using Mean Squared Error (MSE) Loss and the Adam optimizer.
Train the model over multiple epochs and track the loss to monitor the training process.
- **KMeans Clustering:**
After training the matrix factorization model, use KMeans clustering to group movies that have similar latent embeddings. This allows us to identify similar movies that users might enjoy based on their preferences.

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
import os
from sklearn.metrics import mean_squared_error
import logging
import sentry_sdk

# Configure logging
logger = logging.getLogger(__name__)

class SimpleCFModel:
    def __init__(self):
        self.user_movie_matrix = None
        self.user_similarity = None
        self.movie_similarity = None

    def train(self, train_data):
        logger.info("Starting training process.")
        # Ensure 'rating' column is numeric, coercing errors to NaN
        logger.info("Ensuring 'rating' column is numeric...")
        train_data['rating'] = pd.to_numeric(train_data['rating'], errors='coerce')

        # Identify and handle invalid ratings
        invalid_ratings = train_data[train_data['rating'].isna()]
        if not invalid_ratings.empty:
            logger.warning(f"Found {len(invalid_ratings)} invalid ratings. Setting them to 0.")
            train_data['rating'] = train_data['rating'].fillna(0)

        # Create user-item matrix
        logger.info("Creating user-item matrix...")
        self.user_movie_matrix = train_data.pivot(index='userid', columns='movieid', values='rating').fillna(0)

        # Compute similarity matrices with progress bars
        logger.info("Computing user similarity matrix...")
        for _ in tqdm(range(1), desc="User similarity computation"):
            user_similarity_np = cosine_similarity(self.user_movie_matrix)
            self.user_similarity = pd.DataFrame(
                user_similarity_np,
                index=self.user_movie_matrix.index,
                columns=self.user_movie_matrix.index
            )

        logger.info("Computing movie similarity matrix...")
        for _ in tqdm(range(1), desc="Movie similarity computation"):
            movie_similarity_np = cosine_similarity(self.user_movie_matrix.T)
            self.movie_similarity = pd.DataFrame(
                movie_similarity_np,
                index=self.user_movie_matrix.columns,
                columns=self.user_movie_matrix.columns
            )

        logger.info("Training process completed.")

    def recommend(self, userid, top_n=5):
        if userid not in self.user_movie_matrix.index:
            error_message = f"User ID {userid} not found in training data."
            logger.error(error_message)
            sentry_sdk.capture_message(error_message, level='error')
            raise ValueError(error_message)

        # Add a breadcrumb before computing scores
        sentry_sdk.add_breadcrumb(
            category='model',
            message=f'Computing recommendations for user {userid}',
            level='info',
        )

        # Start a Sentry span
        with sentry_sdk.start_span(op="model_recommend", description="Model Recommendation Calculation"):
            logger.info(f"Calculating recommendation scores for user {userid}...")
            # Compute the weighted sum of all users' ratings for each movie
            scores = self.user_similarity.loc[userid].dot(self.user_movie_matrix) / (self.user_similarity.loc[userid].sum() + 1e-8)

            # Get top N recommendations
            logger.info(f"Selecting top {top_n} recommendations for user {userid}...")
            top_indices = np.argsort(scores)[::-1][:top_n]
            recommended_movies = self.user_movie_matrix.columns[top_indices]

            logger.debug(f"Top recommendations for user {userid}: {recommended_movies.tolist()}")
            return recommended_movies.tolist()

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {file_path}")

    @staticmethod
    def load_model(file_path):
        if not os.path.exists(file_path):
            error_message = f"Model file {file_path} not found."
            logger.error(error_message)
            sentry_sdk.capture_message(error_message, level='error')
            raise FileNotFoundError(error_message)
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {file_path}")
        return model

    def evaluate(self, test_data):
        logger.info("Starting evaluation process.")
        # Ensure 'rating' column is numeric, coercing errors to NaN
        logger.info("Ensuring 'rating' column in test data is numeric...")
        test_data['rating'] = pd.to_numeric(test_data['rating'], errors='coerce')
        test_data['rating'] = test_data['rating'].fillna(0)

        # Create user-item matrix for test data
        logger.info("Creating user-item matrix for test data...")
        test_user_movie_matrix = test_data.pivot(index='userid', columns='movieid', values='rating').fillna(0)

        # Ensure that test users exist in training data
        common_users = set(self.user_movie_matrix.index).intersection(set(test_user_movie_matrix.index))
        if not common_users:
            logger.warning("No common users between training and test sets.")
            return None

        # Align the test user_movie_matrix to training user_movie_matrix
        test_user_movie_matrix = test_user_movie_matrix.loc[list(common_users)]
        test_user_movie_matrix = test_user_movie_matrix.reindex(columns=self.user_movie_matrix.columns, fill_value=0)

        # Predict ratings
        logger.info("Predicting ratings for test data...")
        user_similarity_subset = self.user_similarity.loc[test_user_movie_matrix.index]

        # Prevent division by zero by adding a small epsilon
        epsilon = 1e-8
        predicted_ratings = user_similarity_subset.dot(self.user_movie_matrix) / (user_similarity_subset.sum(axis=1).values[:, np.newaxis] + epsilon)

        # Flatten the matrices for comparison
        y_true = test_user_movie_matrix.values.flatten()
        y_pred = predicted_ratings.values.flatten()

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        logger.info(f"RMSE on test data: {rmse}")
        return rmse

if __name__ == "__main__":
    from tqdm import tqdm

    # Configure logging for the main execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load training data with user IDs as strings
    train_df = pd.read_csv(
        'small_train_set.csv',
        header=None,
        names=['userid', 'movieid', 'rating'],
        skiprows=1,  # Skip the header row if '0,1,2' is a header
        dtype={'userid': str, 'movieid': str}
    )
    train_df['userid'] = train_df['userid'].str.strip()
    train_df['movieid'] = train_df['movieid'].str.strip()

    test_df = pd.read_csv(
        'small_test_set.csv',
        header=None,
        names=['userid', 'movieid', 'rating'],
        skiprows=1,  # Skip the header row if present
        dtype={'userid': str, 'movieid': str}
    )
    test_df['userid'] = test_df['userid'].str.strip()
    test_df['movieid'] = test_df['movieid'].str.strip()

    # Initialize and train the model
    model = SimpleCFModel()
    model.train(train_df)
    logger.info("Model trained successfully.")

    # Save the model to disk
    model.save_model('simple_cf_model.pkl')

    # Load the model back (optional)
    loaded_model = SimpleCFModel.load_model('simple_cf_model.pkl')

    # Example recommendation
    user_id = '536377'
    if user_id in loaded_model.user_movie_matrix.index:
        recommendations = loaded_model.recommend(user_id, top_n=5)
        logger.info(f"Example Recommendations for user {user_id}: {recommendations}")
    else:
        logger.warning(f"User ID {user_id} not found in the training data.")

    # Evaluate the model
    loaded_model.evaluate(test_df)

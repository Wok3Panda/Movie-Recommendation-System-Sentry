
# Movie Recommendation System with Sentry Monitoring

This project is a movie recommendation system that uses a collaborative filtering model to provide movie recommendations. The system integrates Sentry for error tracking and performance monitoring to improve reliability in production environments.

## Project Structure

├── server_flask.py     # Flask server with Sentry integration
├── SimpleCFModel.py    # Collaborative filtering model code
├── simulate_request.py # Script to simulate user requests and trigger errors
├── small_test_set.csv  # Sample test dataset
├── small_train_set.csv # Sample training dataset


## Project Files

### 1. `server_flask.py`
A Flask server that hosts the recommendation API with Sentry integration. Key functionalities include:
- **Endpoints**:
  - `/`: Health check endpoint.
  - `/recommend/<userid>`: Provides movie recommendations for a given user.
  - `/debug-sentry`: Triggers a test error to check Sentry integration.
- **Sentry Integration**: Captures errors, adds breadcrumbs, and monitors performance.

### 2. `SimpleCFModel.py`
Defines the `SimpleCFModel` class for collaborative filtering, which includes:
- **Model Training**: Generates a user-movie matrix and calculates similarity matrices.
- **Recommendation**: Provides top-N recommendations based on user similarity.
- **Evaluation**: Calculates the RMSE of predictions for evaluation purposes.
- **Error Logging**: Logs and reports errors to Sentry.

### 3. `simulate_request.py`
A simulation script to mimic real-world usage and generate requests for the recommendation API. This script:
- Loads user IDs from `small_train_set.csv`.
- Sends requests to the recommendation endpoint.
- Occasionally triggers intentional errors to test Sentry.

### 4. Data Files (`small_test_set.csv` and `small_train_set.csv`)
These CSV files contain sample data for training and testing. Each file includes columns for:
- `userid`: Unique user identifier.
- `movieid`: Unique movie identifier.
- `rating`: Rating given by the user to the movie.

## Getting Started

### Prerequisites

- **Python 3.x**
- **Flask 3.x**
- **Sentry SDK** (install using `pip install sentry-sdk`)
- **dotenv** for environment variable management (install using `pip install python-dotenv`)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Movie-Recommendation-System-Sentry.git
   cd Movie-Recommendation-System-Sentry
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add a .env file with your Sentry DSN:
   ```plaintext
   SENTRY_DSN=<Your_Sentry_DSN>
   ```

4. Run the Flask server:
   ```bash
   python server_flask.py
   ```

5. Use simulate_request.py to test the system:
   ```bash
   python simulate_request.py
   ```

### Monitoring and Troubleshooting
- Sentry Dashboard: You can monitor errors and performance metrics on Sentry’s dashboard. Real-time alerts, breadcrumbs, and contextual information aid in troubleshooting issues.
- Simulated Errors: The /debug-sentry route and simulation script intentionally trigger errors for testing Sentry's capabilities.

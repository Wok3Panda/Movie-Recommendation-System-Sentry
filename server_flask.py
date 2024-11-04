import os
from flask import Flask, jsonify, request
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk import capture_message
from sentry_sdk.integrations.logging import LoggingIntegration
import logging
from SimpleCFModel import SimpleCFModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Sentry with Logging Integration
SENTRY_DSN = os.getenv("SENTRY_DSN")

sentry_logging = LoggingIntegration(
    level=logging.INFO,        # Capture info and above as breadcrumbs
    event_level=logging.ERROR  # Send errors as events
)

sentry_sdk.init(
    dsn=SENTRY_DSN,
    integrations=[FlaskIntegration(), sentry_logging],
    traces_sample_rate=1.0,    # Capture 100% of transactions for performance monitoring
    profiles_sample_rate=1.0,  # Enable profiling
)

# Initialize Flask App
app = Flask(__name__)

# Initialize and Load the Model
MODEL_FILE = 'simple_cf_model.pkl'  # Path to the saved model file

try:
    logger.info("Loading the pre-trained model...")
    model = SimpleCFModel.load_model(MODEL_FILE)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Error loading the model")
    sentry_sdk.capture_exception(e)
    # Exit the application if the model fails to load
    exit(1)

# Route for Health Check
@app.route('/')
def index():
    return "Recommendation System API is Running\n"

# Route for Recommendation Requests
@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    # Set the transaction name for Sentry
    with sentry_sdk.configure_scope() as scope:
        scope.set_transaction_name("/recommend/<userid>")
        scope.set_tag("endpoint", "/recommend/<userid>")

    with sentry_sdk.start_transaction(op="recommendation_request", name="Recommendation Request"):
        try:
            if not userid:
                logger.error("No user_id in request")
                return jsonify({"error": "no user_id in request"}), 400

            logger.info(f"Recommendation request received for user: {userid}")
            sentry_sdk.set_user({"id": userid})

            # Add breadcrumb for incoming request
            sentry_sdk.add_breadcrumb(
                category='request',
                message=f'Received recommendation request for user {userid}',
                level='info',
            )

            with sentry_sdk.start_span(op="compute_recommendation", description="Compute Recommendations"):
                recommendations = model.recommend(userid)

                # Add breadcrumb after successful recommendation
                sentry_sdk.add_breadcrumb(
                    category='recommendation',
                    message=f'Computed recommendations for user {userid}',
                    data={'recommendations': recommendations},
                    level='info',
                )

                logger.debug(f"Recommendations for user {userid}: {recommendations}")

            # Add custom context data
            sentry_sdk.set_context("recommendation_data", {
                "userid": userid,
                "recommendations": recommendations
            })

            return jsonify({"recommendations": recommendations}), 200
        except ValueError as ve:
            logger.error(f"ValueError: {ve}")
            sentry_sdk.capture_exception(ve)
            return jsonify({"error": str(ve)}), 404
        except Exception as e:
            logger.exception(f"Unhandled exception for user {userid}")
            sentry_sdk.capture_exception(e)
            return jsonify({"error": "Internal server error"}), 500

# Route to Trigger a Test Error
@app.route('/debug-sentry')
def trigger_error():
    # Set the transaction name for Sentry
    with sentry_sdk.configure_scope() as scope:
        scope.set_transaction_name("/debug-sentry")
        scope.set_tag("endpoint", "/debug-sentry")

    # Add breadcrumb before triggering error
    sentry_sdk.add_breadcrumb(
        category='debug',
        message='Triggering division by zero error',
        level='info',
    )

    division_by_zero = 1 / 0  # This will raise an exception

# Run the Flask App
if __name__ == "__main__":
    # Ensure the model is loaded before starting the server
    app.run(host='0.0.0.0', port=8888)
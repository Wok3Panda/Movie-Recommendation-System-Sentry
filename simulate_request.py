import requests
import time
from tqdm import tqdm
import random
import datetime
import pandas as pd

API_URL = "http://localhost:8888"

def send_recommendation(userid):
    try:
        response = requests.get(f"{API_URL}/recommend/{userid}")
        if response.status_code == 200:
            print(f"Recommendations for User {userid}: {response.json()['recommendations']}")
        else:
            print(f"Error for User {userid}: {response.json()['error']}")
    except Exception as e:
        print(f"Request failed for user {userid}: {e}")

def trigger_errors():
    try:
        response = requests.get(f"{API_URL}/debug-sentry")
    except Exception as e:
        print(f"Error trigger failed: {e}")

if __name__ == "__main__":
    # Load user IDs from the training data
    train_df = pd.read_csv(
        'small_train_set.csv',
        header=None,
        names=['userid', 'movieid', 'rating'],
        skiprows=1,  # Skip the header row if present
        dtype={'userid': str}
    )
    train_df['userid'] = train_df['userid'].str.strip()
    user_ids = train_df['userid'].unique().tolist()

    # Simulate requests for 30 minutes
    duration_minutes = 30
    end_time = datetime.datetime.now() + datetime.timedelta(minutes=duration_minutes)

    print(f"Starting simulation for {duration_minutes} minutes...")

    with tqdm(total=duration_minutes * 60, desc="Simulation Time", unit="s") as pbar:
        last_time = time.time()
        while datetime.datetime.now() < end_time:
            # Randomly select a user ID
            userid = random.choice(user_ids)

            # Occasionally use an invalid user ID to generate errors
            if random.random() < 0.1:  # 10% chance to use invalid user ID
                userid = 'invalid_user_id'

            send_recommendation(userid)

            # Sleep for a random time between 0.5 and 3 seconds to simulate real-world usage
            sleep_time = random.uniform(0.2, 3.0)
            time.sleep(sleep_time)

            # Update the progress bar
            current_time = time.time()
            elapsed_time = current_time - last_time
            pbar.update(elapsed_time)
            last_time = current_time

    # Trigger an error at the end
    print("\nTriggering an error...")
    trigger_errors()

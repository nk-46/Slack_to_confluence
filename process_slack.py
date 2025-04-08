from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.web import WebClient
import os
from dotenv import load_dotenv
from openai import OpenAI
import logging
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd
import requests
import json
import os
import base64
from datetime import datetime
from requests.auth import HTTPBasicAuth

# Load API Keys and environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OpenAI Assistant ID
ASSISTANT_ID = "asst_3wdSmJAfchbWa6sjoQs2KpCV"
#OPENAI THREAD ID
#THREAD_ID = os.getenv("OPENAI_THREAD_ID")

# Set up logging to file
LOG_FILE = "slack_bot.log"

logging.basicConfig(
    level=logging.DEBUG,  # Log everything from DEBUG and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Timestamp + Log Level + Message
    handlers=[
        logging.FileHandler(LOG_FILE),  # Save logs to a file
        logging.StreamHandler()  # Also print logs to the console
    ]
)

logger = logging.getLogger(__name__)  # Create logger instance


# Get environment variables
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")  # You'll need this new token
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHANNEL_ID = os.getenv("CHANNEL_ID")

# Confluence details (store securely in environment variables)
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")  # e.g., "https://your-domain.atlassian.net/wiki"
EMAIL = os.getenv("CONFLUENCE_EMAIL")  # Your email used for Confluence login
API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")  # Generate from Atlassian API tokens
PAGE_ID = os.getenv("CONFLUENCE_PAGE_ID")  # Replace with the actual Confluence page ID
SPACE_ID = os.getenv("CONFLUENCE_SPACE_ID") 


# Initialize clients
slack_client = WebClient(token=SLACK_BOT_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Create Basic Auth string with email+token format
auth = HTTPBasicAuth(EMAIL , API_TOKEN)

# Headers for authentication and content type
HEADERS = {
    "Authorization" : "auth",
    "Accept" : "application/json",
    "Content-Type": "application/json"
}

#start by fetching the slack message
def on_connect(client: SocketModeClient, response):
    """Called when the client connects successfully"""
    logger.info("Connected to Slack Socket Mode!")
    
def on_disconnect(client: SocketModeClient):
    """Called when the client disconnects"""
    logger.info("Disconnected from Slack Socket Mode!")
    
def on_error(client: SocketModeClient, error):
    """Called when the client encounters an error"""
    logger.error(f"Error in Socket Mode: {error}")

# Load CSV Database (Local File)
CSV_FILE = "cleaned_confluence_data_2.csv"
if not os.path.exists(CSV_FILE):
    print("⚠️ CSV file not found! Please ensure 'cleaned_confluence_data_2.csv' exists.")
    exit()

#load pandas dataframe
articles_df = pd.read_csv(CSV_FILE)

# Extract text data
article_texts = articles_df["Content_Chunks"].fillna("")

# Initialize TF-IDF vectorizer for local text search
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(article_texts)


#classify the message
def classify_message(message_text):
    # Step 1: Create a thread
    thread = openai_client.beta.threads.create()

    # Step 2: Add message to the thread with a strict classification prompt
    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"""
        You are a classification assistant. Your only job is to classify the input message into one of the following categories:
        - "simple notification"
        - "process update"
        - "Product enhancement"
        - "carrier notification"

        DO NOT search any database or documents. Just return the category.

        Message: {message_text}

        Respond with only the category name.
        """
    )
    print(f"Thread ID: {thread.id}")
    # Step 3: Run the Assistant on the thread
    run = openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )

    # Step 4: Polling until the run completes
    while True:
        run_status = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run_status.status in ["completed", "failed"]:
            break
        time.sleep(1)  # Wait before checking again

    # Step 5: Get response message
    messages = openai_client.beta.threads.messages.list(thread_id=thread.id)
    assistant_response = messages.data[0].content[0].text.value.strip().lower()
    
    return assistant_response

#Find related articles from CSV
def find_related_article(message_text):
    """Find the most relevant Confluence article using TF-IDF."""
    user_vector = vectorizer.transform([message_text])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    best_match_index = similarities.argmax()
    best_match_title = articles_df.iloc[best_match_index]["Title"]
    return best_match_title if similarities[best_match_index] > 0.2 else None  # Adjust threshold if needed

def handle_slack_event(client: SocketModeClient, req: SocketModeRequest):
    """Handle incoming Slack events."""
    logger.info(f"Received event type: {req.type}")
    logger.info(f"Full event payload: {req.payload}")

    if req.type == "events_api":
        # Acknowledge the request
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

        event = req.payload["event"]
        logger.info(f"Event received: {event}")

        # Only process messages from the specified channel
        if event["type"] == "message" and event.get("channel") == CHANNEL_ID:
            logger.info(f"🔹 Received Message: {event.get('text')} from {event.get('user')} in {event.get('channel')}")
            #Initialize variables
            category = None
            ai_response = ""
            message_text = ""

            # Ignore bot messages and message updates
            if "bot_id" not in event and "subtype" not in event:
                try:
                    message_text = event.get("text", "")
                    thread_ts = event.get("thread_ts")

                    # Get thread messages if it's in a thread
                    thread_messages = []
                    if thread_ts:
                        try:
                            replies = slack_client.conversations_replies(
                                channel=CHANNEL_ID,
                                ts=thread_ts
                            )
                            thread_messages = [msg["text"] for msg in replies.get("messages", [])[1:]]
                        except Exception as e:
                            print(f"Error fetching thread messages: {e}")

                    # Combine thread messages
                    if thread_messages:
                        message_text += " " + " ".join(thread_messages)

                    # Process message classification
                    try:
                        category = classify_message(message_text)
                        logger.info(f"Message category: {category}")
                        print(f"🔍 Message Category: {category}")
                    except Exception as e:
                        print(f"Error classifying message: {e}")
                        category = None

                    # Only proceed if we have a category
                    if category:
                        if category in ["process update", "product enhancement", "carrier notification"]:
                            # Use hardcoded article title for testing
                            article_title = "Updated Confluence Page"
                            
                            ai_response = f"Category: {category}\n✅ Updating Confluence article: {article_title}"
                            
                            # Format and update Confluence
                            formatted_content = f"""
                            <p>Latest Update from Slack:</p>
                            <blockquote>
                                <p>Category: {category}</p>
                                <p>{message_text}</p>
                                <p><em>Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
                            </blockquote>
                            """
                            
                            # Update Confluence
                            success = update_confluence_page(PAGE_ID, formatted_content)
                            if success:
                                ai_response += "\n✅ Confluence page updated successfully!"
                            else:
                                ai_response += "\n❌ Failed to update Confluence page."
                        else:
                            ai_response = f"Category: {category}✅ No Confluence update needed."
                    else:
                        ai_response = "❌ Could not classify message."

                    # Reply in thread if it's a thread message, otherwise send as a new message
                    if ai_response:
                        thread_ts = event.get("thread_ts", event.get("ts"))
                        try:
                            slack_client.chat_postMessage(
                                channel=CHANNEL_ID,
                                text=ai_response,
                                thread_ts=thread_ts
                            )
                        except Exception as e:
                            print(f"Error sending response to Slack: {e}")

                except Exception as e:
                    print(f"Error processing message: {e}")
                    logger.error(f"Error processing message: {e}")
                    try:
                        thread_ts = event.get("thread_ts", event.get("ts"))
                        slack_client.chat_postMessage(
                            channel= CHANNEL_ID,
                            text = f"Error procesing message: {str(e)}",
                            thread_ts=thread_ts
                        )
                    except:
                        pass



def update_confluence_page(PAGE_ID, formatted_content):
    """Update Confluence page using Atlassian's recommended authentication"""
    try:
        # Confluence details
        CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
        EMAIL = os.getenv("CONFLUENCE_EMAIL")
        API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

        # Set up authentication
        auth = HTTPBasicAuth(EMAIL, API_TOKEN)

        # Headers
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        # First get the current version of the page
        get_url = f"{CONFLUENCE_BASE_URL}/api/v2/pages/{PAGE_ID}"

        print(f"Requesting URL: {get_url}")
        
        # Debug print
        print(f"Fetching page content...")
        
        response = requests.get(
            get_url,
            headers=headers,
            auth=auth
        )
        
        if response.status_code == 404:
            print(f"Page not found. Please check:\n"
                  f"1. Page ID: {PAGE_ID} exists\n"
                  f"2. Base URL: {CONFLUENCE_BASE_URL} is correct\n"
                  f"3. You have permission to access this page")
            return False


        if response.status_code != 200:
            print(f"Error fetching page: {response.status_code}, {response.text}")
            return False

        page_data = response.json()
        current_version = page_data['version']['number']
        page_title = page_data['title']

        #Get page content using the content endpoint
        content_url = f"{CONFLUENCE_BASE_URL}/rest/api/content/{PAGE_ID}?expand=body.storage.version"

        content_response = requests.get(
            content_url,
            headers=headers,
            auth=auth
        )

        if content_response.status_code !=200:
            print(f"Error fetching content: {content_response.status_code}, {content_response.text}")
            return False
        
        content_data = content_response.json()
        existing_content = content_data.get('body', {}).get('storage', {}).get('value', '')
        print(f"Current version: {current_version}")
        print(f"Existing content: {existing_content}")

        #Combine existing content with new content
        combined_content = f"{existing_content}\n\n\n {formatted_content}"
        
        # Prepare update data
        new_version = current_version + 1
        update_data = {
            "id": PAGE_ID,
            "status" : "current",
            "type": "page",
            "title": page_data['title'],
            "space": {"key": SPACE_ID},
            "body": {
                "storage": {
                    "value": combined_content,
                    "representation": "storage"
                }
            },
            "version": {"number": new_version}
        }

        # Update the page
        update_url = f"{CONFLUENCE_BASE_URL}/api/v2/pages/{PAGE_ID}"
        
        print(f"Updating page...")
        response = requests.put(
            update_url,
            headers=headers,
            auth=auth,
            json=update_data
        )

        if response.status_code == 200:
            print("✅ Page updated successfully")
            return True
        else:
            print(f"Error updating page: {response.status_code}, {response.text}")
            return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False






def main():
    """Main function to run the Slack event listener."""
    logger.info("=== Starting Slack Bot ===")

    # Initialize Socket Mode client
    logger.info("Initializing Socket Mode client...")
    app = SocketModeClient(
        app_token=SLACK_APP_TOKEN,
        web_client=slack_client
    )

    # Add event handler
    app.socket_mode_request_listeners.append(handle_slack_event)

    # Start the app
    logger.info("⚡️ Connecting to Slack Socket Mode...")
    try:
        app.connect()
        logger.info("Socket Mode connection initiated...")
    except Exception as e:
        logger.error(f"Failed to connect to Socket Mode: {e}")
        return

    # Keep the program running
    import signal

    def signal_handler(signum, frame):
        logger.info("\nStopping the application...")
        app.disconnect()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Bot is running and waiting for events...")

    # Keep the process alive
    while True:
        try:
            signal.pause()
        except AttributeError:
            import time
            time.sleep(1)


if __name__ == "__main__":
    main()

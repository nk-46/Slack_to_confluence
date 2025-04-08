Overview of the Production Workflow:

Slackbot fetches the new message posted on the #support-important-updates channel

Slack message is cleaned and passed through an AI model.

Pre- trained AI models processes the message using Random Forest Classifier

Uses NLP to extract context (e.g., topic, keywords).

Convert slack message into embeddings.

Matches the topic to a Confluence page.

Confluence articles are fetched using APIs, pre-processed, divided into chunks and converted into vectors

Confluence vectors are stored in ChromaDB

A similarity search using cosinesimilarity() returns a suggested Confluence page to update.

If the similarity between slack message and the confluence chunk > 60%, the confluence page is returned and updated.

How the Code Flows:

🚀 Start the bot: python main.py

📡 Bot connects to Slack and listens for messages.

📝 Message arrives → Bot cleans the message.

🔍 Message is classified → Finds related article using ChromaDB.

📚 Confluence article is updated dynamically.

💬 Slack thread is updated with success/failure status.
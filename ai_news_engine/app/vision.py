# app/vision.py

import os
import base64
import requests
import json
import pandas as pd
from dotenv import load_dotenv

# === CONFIGURATION ===
load_dotenv()  # Load .env file
CSV_FILE = "news_data.csv"
API_KEY = os.getenv("GEMINI_API_KEY")  # Load API key from environment variables
API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"

PROMPT_TEXT = """
Task: Extract all articles from the given newspaper image and return the output strictly in JSON format.

The JSON format for each article should look like this:
{
  "articleId": <sequence_number>,
  "title": "<exact title as shown in the image>",
  "description": "<detailed summary starting with historical background and ending with current developments. It should be factual, clear, and comprehensive. Explain like you're narrating to someone unfamiliar with the topic. Don't leave anything out.>",
  "points": [
    "<key point 1>",
    "<key point 2>"
  ],
  "glossary": {
    "<term1>": "<meaning or expansion>",
    "<term2>": "<meaning or expansion>"
  }
}

Notes:
- articleId: Must continue from previous input.
- title: Must match exactly as in image.
- description: Full story in simple words — It should explain the entire news article clearly and completely. Start with what happened in the past (give relevant background and facts) and then explain what is happening now, as mentioned in the article. Write it like you're telling a story to someone who knows nothing about the topic — be factual, clear, and simple. Make sure there are no gaps or unanswered questions. Everything that is listed in the points section must also be explained here in full detail. The goal is that after reading this description, I should fully understand the topic — both its history and current developments — in just one read.
- points: Key facts useful for UPSC or government exams.
- glossary: All difficult/useful words along with meanings in simple words.

Very Important:
- Extract every article in the image.
- Format output as a JSON array.
"""

# === IMAGE ENCODER ===
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# === MAIN FUNCTION TO CALL FROM FLASK ===
def extract_articles_from_images(image_folder="images"):
    # Create CSV file if not exists
    columns = ["articleId", "title", "description", "points", "glossary"]
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=columns).to_csv(CSV_FILE, index=False)

    article_id_counter = pd.read_csv(CSV_FILE).shape[0] + 1
    extracted_articles = []

    for image_file in sorted(os.listdir(image_folder)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            base64_image = encode_image(image_path)

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": PROMPT_TEXT},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(API_URL, headers=headers, json=payload)

            if response.status_code == 200:
                try:
                    result_text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()

                    # Cleanup markdown formatting like ```json
                    if result_text.startswith("```json"):
                        result_text = result_text.strip("`").split("json")[-1].strip()
                    elif result_text.startswith("```"):
                        result_text = result_text.strip("`").split("\n", 1)[-1].strip()

                    articles = json.loads(result_text)
                    rows = []

                    for art in articles:
                        data = {
                            "articleId": article_id_counter,
                            "title": art.get("title", ""),
                            "description": art.get("description", ""),
                            "points": "\n".join(art.get("points", [])),
                            "glossary": json.dumps(art.get("glossary", {}))
                        }
                        rows.append(data)
                        extracted_articles.append(data)
                        article_id_counter += 1

                    # Append to CSV
                    pd.DataFrame(rows).to_csv(CSV_FILE, mode='a', header=False, index=False)

                except Exception as e:
                    print(f"❌ Failed to parse response for {image_file}: {e}")
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")

    return extracted_articles

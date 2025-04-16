import os
import base64
import requests
import json
import pandas as pd
import logging
from dotenv import load_dotenv
from typing import List
from .classifier import predict_category  # Uses your updated classifier setup

# === CONFIGURATION ===
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"

PROMPT_TEXT = """
Task: Extract all **news articles this is must and should you need to extract all articles expcept ads** from the given newspaper image. Return the output strictly in **JSON format** as a JSON array. Each news article must follow the structure and rules given below.

The format for each article should be:

{
  "articleId": <number>,  
  "title": "<exact title shown in the newspaper image>",

  "involvement": "<List all people, organizations, or groups involved in the article. For each one, write in this format — its name (don't inlcude this label ,mention directly real name): a simple explanation of about it. Use very simple and short sentences.>",

  "past": "<Write a paragraph (minimum 4 lines) explaining the background or past events that led to this news. Use factual and accurate information. Use the internet if needed to get correct context. Write in very simple and clear words — like explaining to someone with no prior knowledge. Do not use bullet points — this should be a short paragraph.>",

  "present": "<Write a detailed paragraph (maximum 10 lines) summarizing what is happening now according to the article. Explain the full content of the article clearly. Keep it short and simple but cover everything. Do not use bullet points — this should be a descriptive paragraph.>",

  "points": [
    "<you should give minimum 5 important points in very simple words. These points should help students preparing for government exams like UPSC, SSC, etc. Each point must be clearly explained and easy to understand. End each point with a full stop.>"
  ],

  "glossary": { (you should give minimum 5 english words only not persons or organisation names these should be covered in Involvement section)
    "<word1>": "<simple meaning or explanation>",
    "<word2>": "<simple meaning or explanation>",
    "<abbreviation1>": "<full form and what it means in simple words>",
    "<...>": "<...>"
  }
}

Important Guidelines:

1. The final result must be a **JSON array** — one object per news article.
2. **title**: Must match exactly as shown in the newspaper image. Do not modify, fix, or translate it.
3. **involvement**:don't label as name or role etc just mention name directly-about it, List and explain all people, organizations, or entities mentioned in the article. Keep explanations short and very simple.
4. **past**: Give a short paragraph with 4 or more lines explaining past events related to the article. Keep it factual and easy to read.
5. **present**: Describe the current news in up to 10 lines. This must be a paragraph that clearly explains everything in the article in simple words.
6. **points**: minimum 5 key takeaways in very easy English. These should help someone preparing for exams like UPSC or SSC. Each point should be meaningful and informative.
7. **glossary**: This is a mandatory object. You must minimum 5 terms and above from each article. These can be:
   - Difficult or unique English words from the article (with simple meanings)
   - Abbreviations with full forms and simple explanations
   - Key terms or names (with short descriptions)
   - don't consider person or group or organisation names glossary should have only english words and its meanings which are extracted from that news article

Additional Instructions:
- don't repeat the articles multiple times should be extracted once only 
- tell me all articles from the image no matter they are small or big headings if you see any heading and if it is not ad then consider it 
- Do NOT include any advertisements or image captions.
- Extract **all news articles** from the image (ignore ads).
- All explanations must be **very simple, short, and clear**, as if you are telling a story to someone who doesn’t know anything about the topic.
- You must strictly follow this format. Only return valid JSON.
"""


# === IMAGE ENCODER ===
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# === ARTICLE EXTRACTION FUNCTION ===
def extract_articles_from_images(image_folder: str, csv_file: str) -> List[dict]:
    columns = ["articleId", "title", "involvement", "past", "present", "points", "glossary", "category"]

    if not os.path.exists(csv_file):
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

    article_id_counter = pd.read_csv(csv_file).shape[0] + 1
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

                    if result_text.startswith("```json"):
                        result_text = result_text.strip("`").split("json")[-1].strip()
                    elif result_text.startswith("```"):
                        result_text = result_text.strip("`").split("\n", 1)[-1].strip()

                    articles = json.loads(result_text)
                    logging.info(f"✅ Extracted {len(articles)} articles from {image_file}")

                    rows = []

                    for art in articles:
                        present_info = art.get("present", "")
                        category = predict_category(present_info)

                        row = {
                            "articleId": article_id_counter,
                            "title": art.get("title", ""),
                            "involvement": art.get("involvement", ""),
                            "past": art.get("past", ""),
                            "present": present_info,
                            "points": "\n".join(art.get("points", [])),
                            "glossary": json.dumps(art.get("glossary", {})),
                            "category": category
                        }

                        rows.append(row)
                        extracted_articles.append(row)
                        article_id_counter += 1

                    pd.DataFrame(rows).to_csv(csv_file, mode='a', header=False, index=False)

                except Exception as e:
                    logging.error(f"❌ Failed to parse response for {image_file}: {e}")
            else:
                logging.error(f"❌ API Error {response.status_code} - {response.text}")

    return extracted_articles

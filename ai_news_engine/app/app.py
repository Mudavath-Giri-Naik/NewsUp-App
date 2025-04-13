from flask import Flask, jsonify
from .vision import extract_articles_from_images  # Import your vision.py function

app = Flask(__name__)

@app.route('/extract_articles', methods=['GET'])
def extract_articles():
    try:
        # Call your function to extract articles from images
        articles = extract_articles_from_images(image_folder="images")
        return jsonify({"status": "success", "data": articles}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

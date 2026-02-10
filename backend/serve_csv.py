from flask import Flask, send_file, jsonify
import os

app = Flask(__name__)

CSV_PATH = 'D:/MasterFolders/S3/Analyse_Sentiment_Text_Mining/mainProject/backend/data/cleaned_crypto_hashtags_fixed.csv'

@app.route('/csv')
def get_csv():
    try:
        if os.path.exists(CSV_PATH):
            return send_file(CSV_PATH, mimetype='text/csv', as_attachment=True)
        else:
            return jsonify({"error": "CSV file not found", "path": CSV_PATH}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # CHANGE THIS: Use '0.0.0.0' instead of '127.0.0.1'
    print("Starting server on http://localhost:5001 (accessible from all networks)")
    app.run(host='localhost', port=5001, debug=True)
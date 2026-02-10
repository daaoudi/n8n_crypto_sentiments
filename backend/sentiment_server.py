"""
Sentiment Analysis API Server
Provides VADER and BERT-based sentiment analysis
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================
# VADER SENTIMENT ANALYZER
# ============================================
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
    logger.info("✅ VADER sentiment analyzer loaded")
except ImportError:
    logger.warning("❌ VADER not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment"])
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
    logger.info("✅ VADER installed and loaded")

# ============================================
# BERT SENTIMENT ANALYZER (Optional)
# ============================================
BERT_AVAILABLE = False
bert_model = None
bert_tokenizer = None

def load_bert_model():
    """Lazy loading of BERT model"""
    global BERT_AVAILABLE, bert_model, bert_tokenizer
    
    try:
        logger.info("🔄 Loading BERT model...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        # Use a lightweight BERT model for sentiment
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Set to evaluation mode
        bert_model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            bert_model = bert_model.cuda()
            logger.info("✅ BERT loaded on GPU")
        else:
            logger.info("✅ BERT loaded on CPU")
        
        BERT_AVAILABLE = True
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load BERT: {e}")
        logger.warning("⚠️ BERT model not available. Using VADER only.")
        return False

# ============================================
# HELPER FUNCTIONS
# ============================================
def get_sentiment_label(score):
    """Convert score to sentiment label"""
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

def analyze_vader(text):
    """Analyze sentiment using VADER"""
    try:
        if not text or not isinstance(text, str) or text.strip() == "":
            return {
                "score": 0.0,
                "label": "neutral",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0
            }
        
        # Get sentiment scores
        scores = vader_analyzer.polarity_scores(text)
        
        # Calculate compound score (normalized between -1 and 1)
        compound = scores['compound']
        
        # Map to our desired format
        return {
            "score": round(float(compound), 4),
            "label": get_sentiment_label(compound),
            "positive": round(float(scores['pos']), 4),
            "negative": round(float(scores['neg']), 4),
            "neutral": round(float(scores['neu']), 4),
            "method": "vader"
        }
        
    except Exception as e:
        logger.error(f"VADER analysis error: {e}")
        return {
            "score": 0.0,
            "label": "neutral",
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "error": str(e)
        }

def analyze_bert(text):
    """Analyze sentiment using BERT"""
    try:
        # Lazy load BERT model
        if not BERT_AVAILABLE:
            if not load_bert_model():
                return {"error": "BERT model not available"}
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        if not text or not isinstance(text, str) or text.strip() == "":
            return {
                "score": 0.0,
                "label": "neutral",
                "method": "bert"
            }
        
        # Tokenize
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = bert_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get scores
        scores = predictions.cpu().numpy()[0]
        
        # For star ratings (1-5 stars), convert to sentiment
        # 1-2 stars: negative, 3 stars: neutral, 4-5 stars: positive
        star_rating = np.argmax(scores) + 1
        
        if star_rating <= 2:
            sentiment_score = -0.5 - (star_rating * 0.25)
            label = "negative"
        elif star_rating == 3:
            sentiment_score = 0.0
            label = "neutral"
        else:  # 4-5 stars
            sentiment_score = 0.5 + ((star_rating - 3) * 0.25)
            label = "positive"
        
        return {
            "score": round(float(sentiment_score), 4),
            "label": label,
            "confidence": round(float(np.max(scores)), 4),
            "star_rating": int(star_rating),
            "method": "bert"
        }
        
    except Exception as e:
        logger.error(f"BERT analysis error: {e}")
        return {
            "score": 0.0,
            "label": "neutral",
            "error": str(e),
            "method": "bert"
        }

# ============================================
# API ENDPOINTS
# ============================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "vader_available": VADER_AVAILABLE,
        "bert_available": BERT_AVAILABLE,
        "endpoints": {
            "/analyze": "POST - Analyze sentiment (text or batch)",
            "/health": "GET - Health check",
            "/models": "GET - Available models"
        }
    })

@app.route('/models', methods=['GET'])
def get_models():
    """Get available sentiment models"""
    return jsonify({
        "vader": {
            "available": VADER_AVAILABLE,
            "description": "Rule-based sentiment analysis optimized for social media",
            "language": "English"
        },
        "bert": {
            "available": BERT_AVAILABLE,
            "description": "BERT-based multilingual sentiment analysis",
            "language": "Multilingual"
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """
    Analyze sentiment for text or batch of texts
    Request body:
    {
        "text": "single text to analyze",  # OR
        "texts": ["text1", "text2", ...],  # for batch
        "method": "vader" or "bert" or "auto",
        "return_scores": true/false  # return detailed scores
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get method preference
        method = data.get('method', 'vader').lower()
        
        # Validate method
        if method not in ['vader', 'bert', 'auto']:
            return jsonify({"error": "Invalid method. Use 'vader', 'bert', or 'auto'"}), 400
        
        # Choose analyzer
        if method == 'bert' and not BERT_AVAILABLE:
            logger.warning("BERT requested but not available. Using VADER.")
            method = 'vader'
        elif method == 'auto':
            method = 'bert' if BERT_AVAILABLE else 'vader'
        
        # Single text analysis
        if 'text' in data:
            text = data['text']
            
            if method == 'vader':
                result = analyze_vader(text)
            else:  # bert
                result = analyze_bert(text)
            
            return jsonify({
                "text": text,
                "sentiment": result,
                "method_used": method
            })
        
        # Batch analysis
        elif 'texts' in data:
            texts = data['texts']
            
            if not isinstance(texts, list):
                return jsonify({"error": "'texts' must be a list"}), 400
            
            results = []
            for text in texts:
                if method == 'vader':
                    result = analyze_vader(text)
                else:  # bert
                    result = analyze_bert(text)
                
                results.append({
                    "text": text,
                    "sentiment": result
                })
            
            # Calculate batch statistics
            sentiments = [r['sentiment']['label'] for r in results]
            scores = [r['sentiment']['score'] for r in results if 'score' in r['sentiment']]
            
            return jsonify({
                "batch_size": len(texts),
                "results": results,
                "statistics": {
                    "positive_count": sentiments.count("positive"),
                    "negative_count": sentiments.count("negative"),
                    "neutral_count": sentiments.count("neutral"),
                    "average_score": round(float(np.mean(scores)), 4) if scores else 0.0,
                    "method_used": method
                }
            })
        
        else:
            return jsonify({"error": "Provide either 'text' or 'texts' in request body"}), 400
        
    except Exception as e:
        logger.error(f"API error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Batch analysis endpoint (alias for /analyze with texts)
    Optimized for n8n batch processing
    """
    return analyze_sentiment()

@app.route('/analyze/tweet', methods=['POST'])
def analyze_tweet():
    """
    Specialized endpoint for tweet analysis
    Includes crypto context awareness
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request"}), 400
        
        text = data['text']
        method = data.get('method', 'vader')
        
        # Get base sentiment
        if method == 'vader':
            sentiment_result = analyze_vader(text)
        else:
            sentiment_result = analyze_bert(text)
        
        # Add crypto-specific adjustments
        crypto_adjustment = 0.0
        
        # Positive crypto indicators
        positive_keywords = [
            'bullish', 'moon', 'lambo', 'to the moon', 'pump', 
            'buy', 'accumulate', 'hodl', 'diamond hands'
        ]
        
        # Negative crypto indicators
        negative_keywords = [
            'bearish', 'dump', 'sell', 'fud', 'scam', 'rug pull',
            'crash', 'rekt', 'panic sell'
        ]
        
        text_lower = text.lower()
        
        # Adjust based on crypto context
        for keyword in positive_keywords:
            if keyword in text_lower:
                crypto_adjustment += 0.1
        
        for keyword in negative_keywords:
            if keyword in text_lower:
                crypto_adjustment -= 0.1
        
        # Apply adjustment (capped)
        adjusted_score = max(-1.0, min(1.0, sentiment_result['score'] + crypto_adjustment))
        
        return jsonify({
            "text": text,
            "original_sentiment": sentiment_result,
            "crypto_adjustment": round(crypto_adjustment, 4),
            "adjusted_score": round(adjusted_score, 4),
            "adjusted_label": get_sentiment_label(adjusted_score),
            "method_used": method
        })
        
    except Exception as e:
        logger.error(f"Tweet analysis error: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# START SERVER
# ============================================
if __name__ == '__main__':
    # Load BERT model in background
    import threading
    bert_loader = threading.Thread(target=load_bert_model)
    bert_loader.daemon = True
    bert_loader.start()
    
    logger.info("=" * 50)
    logger.info("🚀 Sentiment Analysis API Server")
    logger.info("📝 Endpoints:")
    logger.info("  • POST /analyze     - Single or batch analysis")
    logger.info("  • POST /analyze/tweet - Tweet-specific analysis")
    logger.info("  • GET  /health      - Health check")
    logger.info("  • GET  /models      - Available models")
    logger.info("=" * 50)
    logger.info(f"✅ VADER available: {VADER_AVAILABLE}")
    logger.info(f"🔄 Loading BERT in background...")
    
    # Run server
    app.run(host='localhost', port=5002, debug=False, threaded=True)
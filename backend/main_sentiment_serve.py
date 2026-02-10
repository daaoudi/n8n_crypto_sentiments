"""
Lightweight Sentiment Analysis API for n8n
VADER only - Fast and optimized for crypto tweets
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
import json
from datetime import datetime
import io  # ← ADD THIS IMPORT
import csv  # ← MAKE SURE THIS IS IMPORTED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Allow JSON body in GET requests
@app.before_request
def handle_json_body_in_get():
    """Allow JSON body in GET requests"""
    if request.method == 'GET' and request.content_type == 'application/json':
        try:
            request.get_json(force=True)
        except Exception:
            pass  # Ignore if JSON parsing fails

# ============================================
# VADER SENTIMENT ANALYZER
# ============================================
def install_vader():
    """Install VADER if not available"""
    import subprocess
    import sys
    
    logger.info("📦 Installing VADER sentiment analyzer...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment==3.3.2"])
        logger.info("✅ VADER installed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to install VADER: {e}")
        return False

# Try to import VADER, install if not available
VADER_AVAILABLE = False
vader_analyzer = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
    logger.info("✅ VADER sentiment analyzer loaded")
except ImportError:
    logger.warning("VADER not found, attempting to install...")
    if install_vader():
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            vader_analyzer = SentimentIntensityAnalyzer()
            VADER_AVAILABLE = True
            logger.info("✅ VADER loaded after installation")
        except ImportError as e:
            logger.error(f"❌ Still cannot load VADER: {e}")
    else:
        logger.error("❌ Cannot proceed without VADER")

# ============================================
# SENTIMENT ANALYSIS FUNCTIONS
# ============================================
def get_sentiment_label(score):
    """Convert score to sentiment label"""
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

def analyze_vader_simple(text):
    """Fast VADER analysis optimized for tweets"""
    try:
        if not text or not isinstance(text, str) or text.strip() == "":
            return {
                "score": 0.0,
                "label": "neutral",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0
            }
        
        scores = vader_analyzer.polarity_scores(text)
        
        # Crypto-specific adjustments
        crypto_adjustment = 0.0
        text_lower = text.lower()
        
        # Strong positive indicators in crypto tweets
        strong_positive = ['🚀', 'moon', 'bullish', 'lambo', 'to the moon', 'pump', 'buy']
        for keyword in strong_positive:
            if keyword in text_lower or ('🚀' in text):
                crypto_adjustment += 0.15
        
        # Strong negative indicators
        strong_negative = ['dump', 'bearish', 'rekt', 'scam', 'rug pull', 'crash']
        for keyword in strong_negative:
            if keyword in text_lower:
                crypto_adjustment -= 0.15
        
        # Moderate indicators
        moderate_positive = ['hodl', 'diamond hands', '💎', 'accumulate', 'long']
        for keyword in moderate_positive:
            if keyword in text_lower:
                crypto_adjustment += 0.08
        
        moderate_negative = ['sell', 'fud', 'panic', 'short']
        for keyword in moderate_negative:
            if keyword in text_lower:
                crypto_adjustment -= 0.08
        
        # Apply adjustment (capped between -0.3 and 0.3)
        crypto_adjustment = max(-0.3, min(0.3, crypto_adjustment))
        adjusted_compound = scores['compound'] + crypto_adjustment
        
        return {
            "score": round(float(adjusted_compound), 4),
            "label": get_sentiment_label(adjusted_compound),
            "positive": round(float(scores['pos']), 4),
            "negative": round(float(scores['neg']), 4),
            "neutral": round(float(scores['neu']), 4),
            "crypto_adjustment": round(crypto_adjustment, 4),
            "original_score": round(float(scores['compound']), 4)
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

# ============================================
# API ENDPOINTS
# ============================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if VADER_AVAILABLE else "degraded",
        "service": "crypto-sentiment-api",
        "vader_available": VADER_AVAILABLE,
        "endpoints": {
            "/analyze": "POST - Analyze sentiment (single or batch)",
            "/analyze/batch": "POST - Batch analysis (optimized)",
            "/health": "GET - Health check"
        }
    })

@app.route('/csv', methods=['GET', 'POST'])
def csv_endpoint():
    """
    CSV endpoint - handles both GET and POST requests
    POST: Accepts CSV file upload with specific columns
    GET: Returns analysis or sample data
    """
    try:
        if request.method == 'POST':
            logger.info(f"📥 CSV POST request received from {request.remote_addr}")
            
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({
                    "success": False,
                    "error": "No file uploaded",
                    "message": "Please upload a CSV file with 'file' field"
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    "success": False,
                    "error": "No file selected"
                }), 400
            
            if not file.filename.lower().endswith('.csv'):
                return jsonify({
                    "success": False,
                    "error": "File must be CSV format",
                    "received_file": file.filename
                }), 400
            
            logger.info(f"📄 Processing uploaded CSV file: {file.filename}")
            
            # Read and parse CSV file
            try:
                content = file.read().decode('utf-8')
                csv_reader = csv.DictReader(io.StringIO(content))
                
                # Check for required columns
                fieldnames = csv_reader.fieldnames or []
                logger.info(f"📋 CSV columns: {fieldnames}")
                
                # Normalize column names (case-insensitive, strip whitespace)
                normalized_columns = {col.strip().lower(): col for col in fieldnames}
                
                # Check for content column with various possible names
                content_column = None
                possible_content_cols = ['content', 'text', 'tweet', 'message', 'post']
                
                for col in possible_content_cols:
                    if col in normalized_columns:
                        content_column = normalized_columns[col]
                        logger.info(f"✅ Found content column: '{content_column}'")
                        break
                
                if not content_column:
                    return jsonify({
                        "success": False,
                        "error": "CSV must contain a content/text column",
                        "available_columns": fieldnames,
                        "expected_columns": possible_content_cols
                    }), 400
                
                # Process rows
                rows = []
                texts = []
                
                for i, row in enumerate(csv_reader):
                    try:
                        # Extract content for sentiment analysis
                        content_text = row.get(content_column, '').strip()
                        
                        if content_text:
                            # Analyze sentiment
                            sentiment = analyze_vader_simple(content_text)
                            
                            # Prepare row with all original data plus sentiment
                            processed_row = {
                                'row_number': i + 1,
                                'content': content_text[:200] + '...' if len(content_text) > 200 else content_text,
                                'sentiment_score': sentiment.get('score', 0),
                                'sentiment_label': sentiment.get('label', 'neutral'),
                                'positive_score': sentiment.get('positive', 0),
                                'negative_score': sentiment.get('negative', 0),
                                'neutral_score': sentiment.get('neutral', 0),
                                'crypto_adjustment': sentiment.get('crypto_adjustment', 0),
                                'original_score': sentiment.get('original_score', 0)
                            }
                            
                            # Add all original columns
                            for col in fieldnames:
                                if col not in processed_row:  # Avoid overwriting sentiment fields
                                    processed_row[col] = row.get(col, '')
                            
                            rows.append(processed_row)
                            texts.append(content_text)
                            
                    except Exception as row_error:
                        logger.warning(f"⚠️ Error processing row {i}: {row_error}")
                        continue
                
                logger.info(f"✅ Processed {len(rows)} rows with content from CSV")
                
                if len(rows) == 0:
                    return jsonify({
                        "success": False,
                        "error": "No valid content found in CSV",
                        "message": "CSV file contained no analyzable text content"
                    }), 400
                
                # Create CSV response with sentiment analysis
                output = io.StringIO()
                
                # Define output columns - original columns + sentiment columns
                output_columns = []
                
                # Start with original columns (excluding the content column we'll rename)
                for col in fieldnames:
                    if col.lower() != content_column.lower():
                        output_columns.append(col)
                
                # Add sentiment analysis columns
                sentiment_columns = [
                    'sentiment_score', 'sentiment_label', 
                    'positive_score', 'negative_score', 'neutral_score',
                    'crypto_adjustment', 'original_score'
                ]
                
                # Insert content and sentiment columns in logical order
                final_columns = output_columns[:]
                final_columns.insert(0, 'Content_Analyzed')
                for sentiment_col in sentiment_columns:
                    final_columns.append(sentiment_col)
                
                writer = csv.DictWriter(output, fieldnames=final_columns)
                writer.writeheader()
                
                # Write processed rows
                for row in rows:
                    output_row = {}
                    
                    # Copy original columns
                    for col in fieldnames:
                        if col.lower() != content_column.lower():
                            output_row[col] = row.get(col, '')
                    
                    # Add analyzed content (truncated)
                    output_row['Content_Analyzed'] = row['content']
                    
                    # Add sentiment columns
                    for sentiment_col in sentiment_columns:
                        output_row[sentiment_col] = row.get(sentiment_col, '')
                    
                    writer.writerow(output_row)
                
                csv_data = output.getvalue()
                
                # Calculate statistics
                labels = [row['sentiment_label'] for row in rows]
                scores = [row['sentiment_score'] for row in rows]
                
                statistics = {
                    "total_rows": len(rows),
                    "rows_with_content": len(texts),
                    "sentiment_distribution": {
                        "positive": labels.count("positive"),
                        "negative": labels.count("negative"),
                        "neutral": labels.count("neutral")
                    },
                    "score_statistics": {
                        "average": round(float(sum(scores) / len(scores)) if scores else 0.0, 4),
                        "maximum": round(float(max(scores)) if scores else 0.0, 4),
                        "minimum": round(float(min(scores)) if scores else 0.0, 4)
                    }
                }
                
                return jsonify({
                    "success": True,
                    "message": f"Successfully analyzed {len(rows)} CSV rows",
                    "csv_data": csv_data,
                    "statistics": statistics,
                    "file_info": {
                        "original_filename": file.filename,
                        "original_columns": fieldnames,
                        "content_column_used": content_column,
                        "processed_rows": len(rows)
                    },
                    "sample_results": rows[:3] if len(rows) > 3 else rows
                })
                
            except Exception as parse_error:
                logger.error(f"❌ CSV parsing error: {parse_error}")
                return jsonify({
                    "success": False,
                    "error": f"Error parsing CSV: {str(parse_error)}",
                    "message": "Could not parse the uploaded CSV file"
                }), 400
        
        else:  # GET request
            logger.info(f"📥 CSV GET request received from {request.remote_addr}")
            
            # Check for JSON body in GET request
            json_data = None
            if request.data:
                try:
                    json_data = request.get_json()
                    logger.info(f"📦 Received JSON body in GET request")
                except Exception:
                    pass  # Not JSON or empty
            
            # If JSON body with CSV data structure
            if json_data and isinstance(json_data, list) and len(json_data) > 0:
                # Check if it has CSV-like structure
                first_item = json_data[0]
                if isinstance(first_item, dict):
                    # Look for CSV columns
                    csv_columns = ['Unnamed', 'Date', 'Username', 'Content', 'URL', 'Hashtags_Cleaned']
                    found_columns = [col for col in csv_columns if col in first_item]
                    
                    if found_columns:
                        logger.info(f"📋 Processing CSV-like JSON data with columns: {found_columns}")
                        
                        rows = []
                        texts = []
                        
                        for item in json_data:
                            if isinstance(item, dict):
                                content = item.get('Content', '') or item.get('content', '')
                                if content and isinstance(content, str) and content.strip():
                                    # Analyze sentiment
                                    sentiment = analyze_vader_simple(content)
                                    
                                    # Create processed row
                                    processed_row = item.copy()
                                    processed_row.update({
                                        'sentiment_score': sentiment.get('score', 0),
                                        'sentiment_label': sentiment.get('label', 'neutral'),
                                        'positive_score': sentiment.get('positive', 0),
                                        'negative_score': sentiment.get('negative', 0),
                                        'neutral_score': sentiment.get('neutral', 0),
                                        'crypto_adjustment': sentiment.get('crypto_adjustment', 0),
                                        'original_score': sentiment.get('original_score', 0),
                                        'Content_Analyzed': content[:200] + '...' if len(content) > 200 else content
                                    })
                                    
                                    rows.append(processed_row)
                                    texts.append(content)
                        
                        if rows:
                            # Create CSV output
                            output = io.StringIO()
                            
                            # Get all unique keys from all rows
                            all_keys = set()
                            for row in rows:
                                all_keys.update(row.keys())
                            
                            # Define column order
                            preferred_order = [
                                'Unnamed', 'Date', 'Username', 'Content_Analyzed', 
                                'URL', 'Hashtags_Cleaned',
                                'sentiment_score', 'sentiment_label',
                                'positive_score', 'negative_score', 'neutral_score',
                                'crypto_adjustment', 'original_score'
                            ]
                            
                            # Create final column list
                            final_columns = []
                            for col in preferred_order:
                                if col in all_keys:
                                    final_columns.append(col)
                                    all_keys.remove(col)
                            
                            # Add any remaining columns
                            final_columns.extend(sorted(all_keys))
                            
                            writer = csv.DictWriter(output, fieldnames=final_columns)
                            writer.writeheader()
                            
                            for row in rows:
                                writer.writerow(row)
                            
                            csv_data = output.getvalue()
                            
                            # Calculate statistics
                            labels = [row['sentiment_label'] for row in rows]
                            scores = [row['sentiment_score'] for row in rows]
                            
                            return jsonify({
                                "success": True,
                                "message": f"Analyzed {len(rows)} CSV rows from JSON",
                                "csv_data": csv_data,
                                "statistics": {
                                    "total_rows": len(rows),
                                    "positive": labels.count("positive"),
                                    "negative": labels.count("negative"),
                                    "neutral": labels.count("neutral"),
                                    "avg_score": round(float(sum(scores) / len(scores)) if scores else 0.0, 4)
                                }
                            })
            
            # Check for direct text analysis via query parameters
            text_param = request.args.get('text', '').strip()
            
            if text_param:
                # Analyze single text
                sentiment = analyze_vader_simple(text_param)
                
                # Create CSV with standard columns
                output = io.StringIO()
                writer = csv.writer(output)
                
                writer.writerow([
                    'Unnamed', 'Date', 'Username', 'Content', 'URL', 'Hashtags_Cleaned',
                    'sentiment_score', 'sentiment_label', 'positive_score', 'negative_score',
                    'neutral_score', 'crypto_adjustment', 'original_score'
                ])
                
                writer.writerow([
                    1,  # Unnamed
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Date
                    'API User',  # Username
                    text_param[:500],  # Content (truncated)
                    '',  # URL
                    '',  # Hashtags_Cleaned
                    sentiment.get('score', 0),
                    sentiment.get('label', 'neutral'),
                    sentiment.get('positive', 0),
                    sentiment.get('negative', 0),
                    sentiment.get('neutral', 0),
                    sentiment.get('crypto_adjustment', 0),
                    sentiment.get('original_score', 0)
                ])
                
                csv_data = output.getvalue()
                
                return jsonify({
                    "success": True,
                    "message": "Text analyzed successfully",
                    "csv_data": csv_data,
                    "sentiment": sentiment
                })
            
            # Return sample CSV structure if no data provided
            logger.info("📋 Returning CSV endpoint info")
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers with requested columns
            writer.writerow([
                'Unnamed', 'Date', 'Username', 'Content', 'URL', 'Hashtags_Cleaned',
                'sentiment_score', 'sentiment_label', 'positive_score', 'negative_score',
                'neutral_score', 'crypto_adjustment', 'original_score'
            ])
            
            # Write sample data
            sample_data = [
                [
                    1,
                    '2024-01-15 14:30:00',
                    'crypto_trader_99',
                    'Bitcoin is going to the moon! 🚀 #BTC #crypto',
                    'https://twitter.com/user/status/123456',
                    'BTC,crypto',
                    0.85, 'positive', 0.75, 0.05, 0.20, 0.15, 0.70
                ],
                [
                    2,
                    '2024-01-15 14:35:00',
                    'investor_pro',
                    'Market looks bearish today, might be a good time to sell. #trading',
                    'https://twitter.com/user/status/123457',
                    'trading',
                    -0.45, 'negative', 0.10, 0.60, 0.30, -0.10, -0.35
                ],
                [
                    3,
                    '2024-01-15 14:40:00',
                    'neutral_observer',
                    'Ethereum network upgrade completed as scheduled.',
                    'https://twitter.com/user/status/123458',
                    '',
                    0.05, 'neutral', 0.25, 0.20, 0.55, 0.00, 0.05
                ]
            ]
            
            for row in sample_data:
                writer.writerow(row)
            
            csv_data = output.getvalue()
            
            return jsonify({
                "success": True,
                "message": "CSV endpoint ready. Upload CSV file with POST or use query parameters.",
                "usage": {
                    "POST_upload": "POST /csv with multipart/form-data containing CSV file",
                    "GET_json": "GET /csv with JSON body containing CSV-like data",
                    "GET_query": "GET /csv?text=Your text here",
                    "sample_csv": csv_data
                },
                "expected_columns": [
                    "Unnamed", "Date", "Username", "Content", "URL", "Hashtags_Cleaned"
                ],
                "note": "Sentiment analysis columns will be added to the output"
            })
    
    except Exception as e:
        logger.error(f"CSV endpoint error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Error processing CSV request"
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """
    Main sentiment analysis endpoint
    Optimized for n8n integration
    """
    if not VADER_AVAILABLE:
        return jsonify({"error": "VADER analyzer not available"}), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Handle single text
        if 'text' in data:
            text = data['text']
            result = analyze_vader_simple(text)
            
            return jsonify({
                "text": text,
                "sentiment": result,
                "success": True
            })
        
        # Handle batch (optimized for n8n arrays)
        elif 'texts' in data:
            texts = data['texts']
            
            if not isinstance(texts, list):
                return jsonify({"error": "'texts' must be a list"}), 400
            
            # Limit batch size for performance
            if len(texts) > 1000:
                texts = texts[:1000]
                logger.warning(f"Batch truncated to 1000 items")
            
            results = []
            for text in texts:
                result = analyze_vader_simple(text)
                results.append({
                    "text": text,
                    "sentiment": result
                })
            
            # Calculate statistics
            labels = [r['sentiment']['label'] for r in results]
            scores = [r['sentiment']['score'] for r in results]
            
            return jsonify({
                "success": True,
                "batch_size": len(texts),
                "results": results,
                "statistics": {
                    "positive": labels.count("positive"),
                    "negative": labels.count("negative"),
                    "neutral": labels.count("neutral"),
                    "avg_score": round(float(sum(scores) / len(scores)) if scores else 0.0, 4),
                    "max_score": round(float(max(scores)) if scores else 0.0, 4),
                    "min_score": round(float(min(scores)) if scores else 0.0, 4)
                }
            })
        
        # Handle n8n item format
        elif 'items' in data:
            items = data['items']
            results = []
            
            for item in items:
                if 'json' in item and 'text' in item['json']:
                    text = item['json']['text']
                    result = analyze_vader_simple(text)
                    
                    # Preserve original item structure
                    new_item = item.copy()
                    if 'json' in new_item:
                        new_item['json']['sentiment'] = result
                    else:
                        new_item['json'] = {'sentiment': result}
                    
                    results.append(new_item)
            
            return jsonify({
                "success": True,
                "items": results,
                "processed_count": len(results)
            })
        
        else:
            return jsonify({"error": "Provide 'text', 'texts', or 'items' in request"}), 400
        
    except Exception as e:
        logger.error(f"API error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "success": False
        }), 500

@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Optimized batch endpoint for n8n
    Accepts array of texts directly
    """
    if not VADER_AVAILABLE:
        return jsonify({"error": "VADER analyzer not available"}), 503
    
    try:
        # DEBUG: Log incoming request
        print("\n" + "="*60)
        print("📥 INCOMING REQUEST TO /analyze/batch")
        print(f"Time: {datetime.now()}")
        print(f"Content-Type: {request.headers.get('Content-Type')}")
        
        # Get raw data
        raw_data = request.get_data(as_text=True)
        print(f"Raw data length: {len(raw_data)} chars")
        print(f"Raw data (first 500 chars): {raw_data[:500]}")
        
        # Parse JSON
        data = None
        if raw_data and raw_data.strip():
            try:
                data = json.loads(raw_data)
                print(f"✅ Parsed JSON type: {type(data)}")
                print(f"Data sample: {str(data)[:200]}...")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                print(f"Problematic data: {raw_data[:200]}")
                return jsonify({
                    "error": f"Invalid JSON: {str(e)}",
                    "success": False
                }), 400
        else:
            print("❌ No data received")
            return jsonify({"error": "No data received", "success": False}), 400
        
        # Accept array directly or wrapped in 'texts'
        if isinstance(data, list):
            texts = data
            print(f"✅ Received direct array with {len(texts)} items")
        elif isinstance(data, dict) and 'texts' in data:
            texts = data['texts']
            print(f"✅ Received object with texts array: {len(texts)} items")
        else:
            print(f"❌ Invalid format: {type(data)} - {data}")
            return jsonify({
                "error": f"Provide array of texts or {{'texts': []}}. Got: {type(data)}",
                "success": False
            }), 400
        
        if not isinstance(texts, list):
            print(f"❌ 'texts' is not a list: {type(texts)}")
            return jsonify({"error": "Input must be a list", "success": False}), 400
        
        print(f"📊 Processing {len(texts)} texts...")
        
        # Process in batches for performance
        batch_size = 100
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                sentiment = analyze_vader_simple(text)
                batch_results.append({
                    "text": text,
                    "sentiment": sentiment
                })
            
            all_results.extend(batch_results)
        
        print(f"✅ Successfully processed {len(all_results)} texts")
        
        # Return results
        return jsonify({
            "success": True,
            "total_processed": len(all_results),
            "results": all_results
        })
        
    except Exception as e:
        print(f"❌ Server error: {traceback.format_exc()}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False,
            "traceback": traceback.format_exc()[:500]
        }), 500

# ============================================
# START SERVER
# ============================================
if __name__ == '__main__':
    if not VADER_AVAILABLE:
        logger.error("❌ Cannot start server: VADER not available")
        exit(1)
    
    logger.info("=" * 50)
    logger.info("🚀 Crypto Sentiment Analysis API")
    logger.info("📡 Optimized for n8n integration")
    logger.info("=" * 50)
    logger.info("📝 Available endpoints:")
    logger.info("  • POST /analyze      - Single or batch analysis")
    logger.info("  • POST /analyze/batch - Optimized batch processing")
    logger.info("  • GET  /health       - Health check")
    logger.info("=" * 50)
    logger.info(f"✅ Server starting on port 5002")
    
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
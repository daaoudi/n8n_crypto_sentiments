import pandas as pd
import re
import unicodedata
from collections import Counter

def clean_csv_specific(input_file, output_file):
    """Clean CSV with specific focus on Hashtags column Unicode issues"""
    try:
        # Read CSV
        df = pd.read_csv(input_file, encoding='utf-8')
        
        print(f"📊 Data loaded: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show problematic hashtags before cleaning
        if 'Hashtags' in df.columns:
            print(f"\n🔍 Analyzing Hashtags column before cleaning...")
            
            # Get all hashtag entries
            all_hashtag_entries = df['Hashtags'].dropna().tolist()
            
            # Show problematic examples
            problematic_count = 0
            for i, entry in enumerate(all_hashtag_entries[:10]):  # Check first 10
                if has_problematic_unicode(entry):
                    problematic_count += 1
                    print(f"  Row {i}: {entry}")
            
            if problematic_count > 0:
                print(f"\n⚠️ Found {problematic_count} entries with problematic Unicode in first 10 rows")
        
        # Clean Hashtags column specifically
        if 'Hashtags' in df.columns:
            print(f"\n🧹 Cleaning Hashtags column...")
            
            # Convert to string and clean
            df['Hashtags'] = df['Hashtags'].astype(str).apply(clean_hashtags_aggressive)
            
            # Create a new cleaned version
            df['Hashtags_Cleaned'] = df['Hashtags'].apply(extract_english_hashtags)
            
            # Also clean Content column if it exists
            if 'Content' in df.columns:
                df['Content'] = df['Content'].astype(str).apply(clean_text_complete)
        
        # Clean other text columns
        text_cols = ['Username', 'Source', 'URL']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(clean_simple_text)
        
        # Save cleaned CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n✅ Saved cleaned CSV: {output_file}")
        
        # Show after cleaning results
        if 'Hashtags' in df.columns and 'Hashtags_Cleaned' in df.columns:
            print(f"\n🔍 Results after cleaning:")
            print("-" * 50)
            
            # Show before/after comparison
            for i in range(min(5, len(df))):
                original = df['Hashtags'].iloc[i]
                cleaned = df['Hashtags_Cleaned'].iloc[i]
                
                if original != cleaned:
                    print(f"Row {i}:")
                    print(f"  Original: {original}")
                    print(f"  Cleaned: {cleaned}")
                    print()
            
            # Show most common cleaned hashtags
            all_cleaned = ' '.join(df['Hashtags_Cleaned'].dropna().tolist()).split(', ')
            clean_list = [h.strip() for h in all_cleaned if h.strip()]
            
            if clean_list:
                print(f"\n📊 Most common cleaned hashtags:")
                tag_counts = Counter(clean_list)
                for tag, count in tag_counts.most_common(15):
                    if tag:  # Skip empty
                        print(f"  #{tag}: {count}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

def has_problematic_unicode(text):
    """Check if text contains problematic Unicode characters"""
    if pd.isna(text):
        return False
    
    # Check for CJK and other problematic Unicode ranges
    pattern = r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u00c0-\u00ff]+'
    return bool(re.search(pattern, str(text)))

def clean_hashtags_aggressive(hashtag_text):
    """Aggressively clean hashtags to remove problematic Unicode"""
    if pd.isna(hashtag_text) or str(hashtag_text).lower() in ['nan', 'null', 'none', '[]']:
        return ""
    
    text = str(hashtag_text)
    
    # If it's a list-like string (with brackets), parse it
    if text.startswith('[') and text.endswith(']'):
        # Remove brackets and split
        text = text[1:-1]  # Remove brackets
        items = [item.strip().strip("'\"") for item in text.split(',')]
        
        # Clean each item
        cleaned_items = []
        for item in items:
            cleaned = clean_single_hashtag(item)
            if cleaned:  # Only add non-empty
                cleaned_items.append(cleaned)
        
        return ', '.join(cleaned_items)
    
    # If it's already comma-separated
    items = [item.strip() for item in text.split(',')]
    cleaned_items = [clean_single_hashtag(item) for item in items if clean_single_hashtag(item)]
    
    return ', '.join(cleaned_items)

def clean_single_hashtag(hashtag):
    """Clean a single hashtag"""
    if not hashtag or str(hashtag).lower() in ['nan', 'null', 'none', '']:
        return ""
    
    hashtag = str(hashtag).strip()
    
    # Remove # symbol if present
    if hashtag.startswith('#'):
        hashtag = hashtag[1:]
    
    # Remove quotes
    hashtag = hashtag.strip("'\"")
    
    # Aggressive Unicode removal
    # Keep only ASCII letters, numbers, and common crypto symbols
    hashtag = re.sub(r'[^\x00-\x7F]+', '', hashtag)  # Remove all non-ASCII
    
    # Keep only alphanumeric and underscore
    hashtag = re.sub(r'[^\w]', '', hashtag)
    
    # Check if it looks like a crypto term
    crypto_terms = {
        'btc', 'bitcoin', 'eth', 'ethereum', 'ada', 'cardano', 
        'sol', 'solana', 'dot', 'polkadot', 'link', 'chainlink',
        'avax', 'avalanche', 'matic', 'polygon', 'bnb', 'usdc',
        'usdt', 'doge', 'xrp', 'ltc', 'shib', 'uni'
    }
    
    # If cleaned hashtag is in crypto terms, keep it lowercase
    if hashtag.lower() in crypto_terms:
        return hashtag.lower()
    
    # Otherwise, only keep if it's reasonable length and not empty
    if 2 <= len(hashtag) <= 30:
        return hashtag.lower()
    
    return ""

def extract_english_hashtags(hashtag_text):
    """Extract only English/ASCII hashtags"""
    if pd.isna(hashtag_text) or not hashtag_text:
        return ""
    
    # Split by comma
    items = [item.strip() for item in str(hashtag_text).split(',')]
    
    # Filter for English/ASCII only
    english_items = []
    for item in items:
        # Check if item contains only ASCII characters
        if item and all(ord(char) < 128 for char in item):
            # Check if it looks like a valid hashtag
            if re.match(r'^[a-zA-Z0-9_]{2,30}$', item):
                english_items.append(item.lower())
    
    return ', '.join(english_items)

def clean_text_complete(text):
    """Complete text cleaning"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove problematic Unicode
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove # symbol but keep word for hashtags in content
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Clean up
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    return text

def clean_simple_text(text):
    """Simple text cleaning"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_specific_cases():
    """Test specific problematic cases"""
    print("\n🧪 Testing specific problematic hashtag cases:")
    print("=" * 60)
    
    test_cases = [
        "['ä¸­å¸', 'ä¸–ç•Œæ¯', 'USDC', 'ä¸–ç•Œæ¯ä¹°çƒ']",
        "['å½©ç¥¨ç½‘', 'ä¸–ç•Œæ¯', 'cryptocurrency', 'ä¸–ç•Œæ¯æŠ•æ³¨']",
        "['BNB', 'è¶³å½©', 'å¼€äº‘ä½“è‚²']",
        "['bitcoin', 'btc', 'crypto']",
        "['ETH', 'ethereum', 'ä¸–ç•Œæ¯']",
        "[]",
        "nan"
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"  Input:  {test_case}")
        cleaned = clean_hashtags_aggressive(test_case)
        print(f"  Cleaned: {cleaned}")
        
        # Also show English-only extraction
        english_only = extract_english_hashtags(cleaned)
        if english_only != cleaned:
            print(f"  English Only: {english_only}")

def create_crypto_hashtag_summary(df):
    """Create a summary of crypto-related hashtags"""
    if 'Hashtags_Cleaned' not in df.columns:
        return
    
    print(f"\n📈 Crypto Hashtag Analysis:")
    print("=" * 50)
    
    # Get all cleaned hashtags
    all_hashtags = []
    for tags in df['Hashtags_Cleaned'].dropna():
        if tags:
            all_hashtags.extend([tag.strip() for tag in tags.split(',')])
    
    # Crypto-related hashtags to track
    crypto_keywords = {
        'btc': ['btc', 'bitcoin'],
        'eth': ['eth', 'ethereum'],
        'ada': ['ada', 'cardano'],
        'sol': ['sol', 'solana'],
        'dot': ['dot', 'polkadot'],
        'link': ['link', 'chainlink'],
        'avax': ['avax', 'avalanche'],
        'matic': ['matic', 'polygon'],
        'bnb': ['bnb'],
        'usdc': ['usdc'],
        'usdt': ['usdt'],
        'doge': ['doge', 'dogecoin'],
        'xrp': ['xrp', 'ripple'],
        'shib': ['shib', 'shiba'],
        'crypto': ['crypto', 'cryptocurrency']
    }
    
    # Count occurrences
    counts = {coin: 0 for coin in crypto_keywords.keys()}
    
    for hashtag in all_hashtags:
        hashtag_lower = hashtag.lower()
        for coin, keywords in crypto_keywords.items():
            if any(keyword == hashtag_lower for keyword in keywords):
                counts[coin] += 1
    
    # Display results
    for coin, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  #{coin.upper()}: {count:,} mentions")

if __name__ == "__main__":
    input_file = "data/crypto_10k_tweets.csv"
    output_file = "data/cleaned_crypto_hashtags_fixed.csv"
    
    print("🔧 FIXED: Hashtag Cleaning Tool for Unicode Issues")
    print("=" * 60)
    
    # Test specific cases first
    analyze_specific_cases()
    
    print("\n" + "=" * 60)
    print("🔄 Starting CSV cleaning...")
    print("=" * 60)
    
    # Clean the CSV
    cleaned_df = clean_csv_specific(input_file, output_file)
    
    # Create crypto summary
    if cleaned_df is not None:
        create_crypto_hashtag_summary(cleaned_df)
    
    print(f"\n✅ Your Hashtags column is now clean!")
    print("   Problematic Unicode characters have been removed.")
    print("   Ready for n8n sentiment analysis.")
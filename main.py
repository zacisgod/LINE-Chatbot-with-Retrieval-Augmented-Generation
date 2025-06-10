# 加上註解
# 引入所需的庫
import os
import logging
import re
import requests
import pandas as pd
import numpy as np
# 引入 Flask 和 LineBot 相關的類別
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
import openai
from googlesearch import search
# 引入 ChromaDB 相關的類別
import chromadb
from chromadb.utils import embedding_functions
# 載入環境變數
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

# 創建 Flask 應用
app = Flask(__name__)

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 檢查環境變數是否設定
def check_environment_variables():
    required_vars = ['CHANNEL_ACCESS_TOKEN', 'CHANNEL_SECRET', 'OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"缺少以下環境變數: {', '.join(missing_vars)}")
        logger.error("請設定環境變數或創建 .env 檔案")
        logger.error("範例:")
        logger.error("CHANNEL_ACCESS_TOKEN=your_channel_access_token")
        logger.error("CHANNEL_SECRET=your_channel_secret")
        logger.error("OPENAI_API_KEY=your_openai_api_key")
        return False
    return True

# 檢查環境變數
if not check_environment_variables():
    exit(1)

# 設定你的 Channel Access Token 和 Channel Secret
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 設定 OpenAI API 金鑰
openai.api_key = os.getenv('OPENAI_API_KEY')

# 初始化 ChromaDB 客戶端
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 使用 OpenAI 的嵌入函數
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name="text-embedding-ada-002"
)

# 創建或獲取集合
collection = chroma_client.get_or_create_collection(name="amazon_products", embedding_function=openai_ef)

# 資料預處理函數
def preprocess_data(df):
    df_cleaned = df.dropna(axis=1, how='all')
    df_cleaned = df_cleaned[['product_id', 'product_name', 'category', 'discounted_price', 'actual_price', 'about_product', 'img_link', 'product_link']]
    df_cleaned = df_cleaned.dropna(subset=['discounted_price', 'actual_price'], how='all')

    # 處理重複的 product_id
    df_cleaned = df_cleaned.drop_duplicates(subset='product_id', keep='first')

    return df_cleaned

# 將數據加載到 ChromaDB（分批處理）
def load_data_to_chroma(df, batch_size=100):
    existing_ids = set(collection.get()['ids'])
    new_ids = []
    new_metadata = []
    new_documents = []

    for _, row in df.iterrows():
        product_id = str(row['product_id'])
        if product_id not in existing_ids:
            new_ids.append(product_id)
            new_metadata.append({
                "product_name": row['product_name'],
                "category": row['category'],
                "discounted_price": str(row['discounted_price']),
                "actual_price": str(row['actual_price']),
                "about_product": row['about_product'],
                "img_link": row['img_link'],
                "product_link": row['product_link']
            })
            # 限制文本長度以避免 token 過多
            about_product = str(row['about_product'])[:500] if pd.notna(row['about_product']) else ""
            document_text = f"{row['product_name']} {row['category']} {about_product}"
            new_documents.append(document_text)

    if new_ids:
        # 分批處理以避免 token 限制
        total_batches = (len(new_ids) + batch_size - 1) // batch_size
        logging.info(f"Processing {len(new_ids)} new products in {total_batches} batches")
        
        for i in range(0, len(new_ids), batch_size):
            batch_ids = new_ids[i:i+batch_size]
            batch_metadata = new_metadata[i:i+batch_size]
            batch_documents = new_documents[i:i+batch_size]
            
            try:
                collection.add(
                    ids=batch_ids,
                    metadatas=batch_metadata,
                    documents=batch_documents
                )
                logging.info(f"Successfully added batch {(i//batch_size)+1}/{total_batches} ({len(batch_ids)} products)")
            except Exception as e:
                logging.error(f"Error adding batch {(i//batch_size)+1}: {e}")
                # 如果批次太大，嘗試更小的批次
                if "max_tokens_per_request" in str(e) and batch_size > 10:
                    logging.info(f"Reducing batch size from {batch_size} to {batch_size//2}")
                    return load_data_to_chroma(df, batch_size//2)
                else:
                    raise e
    
    logging.info(f"Finished loading data to ChromaDB")

# 載入和預處理數據
def load_initial_data():
    try:
        df = pd.read_csv('./datasets/amazon.csv')
        df_final = preprocess_data(df)
        load_data_to_chroma(df_final)
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {e}")
        raise

# 初始化數據
if collection.count() == 0:
    load_initial_data()
else:
    logging.info("Data already loaded in ChromaDB, skipping initial load")

def clean_amazon_image_url(url):
    pattern = r'(https://m\.media-amazon\.com/images/)W/[^/]+/images/'
    cleaned_url = re.sub(pattern, r'\1', url)
    return cleaned_url

def is_valid_image_url(url):
    try:
        response = requests.head(url)
        return response.status_code == 200 and response.headers.get('content-type', '').startswith('image/')
    except requests.RequestException:
        return False
        
def search_products(query):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=1
        )
        return results
    except Exception as e:
        logger.error(f"Error searching products: {e}")
        return None

def format_price(price_str, include_symbol=False):
    if not price_str:
        return "無資料"
    # 移除所有非數字字符
    price_num = re.sub(r'[^\d.]', '', price_str)
    try:
        if include_symbol:
            return f"₹{float(price_num):,.0f}"
        else:
            return f"{float(price_num):.0f}"
    except ValueError:
        return price_str

# 使用 OpenAI 的 GPT-4 模型生成回應
def get_gpt4_response(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides information about products."},
                {"role": "user", "content": f"I'm looking for information about {query}. Can you help me?"}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

# 使用 Google 搜尋獲取結果
def get_google_search_results(query):
    try:
        search_results = list(search(query, num_results=3))
        if search_results:
            return f"I couldn't find the product in our database, but here are some relevant links:\n\n" + "\n".join(search_results)
        else:
            return "I'm sorry, I couldn't find any relevant information."
    except Exception as e:
        print(f"Google search error: {e}")
        return None

@app.route("/", methods=['GET'])
def hello():
    return "Hello! The server is running."

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logging.error("Invalid signature")
        abort(400)
    except Exception as e:
        logging.error(f"Error handling webhook: {e}")
        return 'Error', 500

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    try:
        text = event.message.text
        user_id = event.source.user_id
        logger.info(f"Received message from user {user_id}: {text}")

        # 使用 ChromaDB 搜尋產品
        results = search_products(text)

        # 新增詳細日誌
        logger.info(f"Raw search results: {results}")
        logger.info(f"Type of results: {type(results)}")
        logger.info(f"Type of results['metadatas']: {type(results['metadatas'])}")

        if results is None or len(results['ids']) == 0:
            logger.info(f"No results found for query: {text}")
            gpt4_response = get_gpt4_response(text)
            if gpt4_response:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=gpt4_response)
                )
            else:
                google_results = get_google_search_results(text)
                if google_results:
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text=google_results)
                    )
                else:
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text="抱歉，我無法找到相關的信息。請嘗試其他關鍵字。")
                    )
        else:
            # 確保 metadatas 是一個列表，並且至少有一個元素
            if isinstance(results['metadatas'], list) and len(results['metadatas']) > 0:
                product = results['metadatas'][0]
                if isinstance(product, list) and len(product) > 0:
                    product = product[0]  # 如果 product 是列表，取第一個元素
            else:
                logger.error("Unexpected results structure")
                raise ValueError("Unexpected results structure")

            logger.info(f"Found product: {product}")

            # 使用 dict.get() 方法來安全地訪問字典鍵
            product_info = f"產品: {product.get('product_name', '無資料')}\n"
            product_info += f"類別: {product.get('category', '無資料')}\n"

            discounted_price = format_price(product.get('discounted_price', '無資料'))
            actual_price = format_price(product.get('actual_price', '無資料'))

            product_info += f"折扣價: {discounted_price}\n"
            product_info += f"原價: {actual_price}\n"

            about_product = product.get('about_product', '無資料')
            product_info += f"產品介紹: {about_product[:200]}...\n" if len(about_product) > 200 else f"產品介紹: {about_product}\n"
            product_info += f"產品連結: {product.get('product_link', '無資料')}"

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=product_info)
            )

            img_link = product.get('img_link')
            if img_link:
                cleaned_img_url = clean_amazon_image_url(img_link)
                if is_valid_image_url(cleaned_img_url):
                    line_bot_api.push_message(
                        user_id,
                        ImageSendMessage(
                            original_content_url=cleaned_img_url,
                            preview_image_url=cleaned_img_url
                        )
                    )
                else:
                    logger.warning(f"Invalid image URL: {cleaned_img_url}")
            else:
                logger.warning("No image link found for the product")

    except Exception as e:
        logger.error(f"Error in handle_message: {e}", exc_info=True)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="抱歉，處理您的請求時發生錯誤。請稍後再試。")
        )

if __name__ == "__main__":
    app.run()
from flask import Flask, request, jsonify, render_template
import whisper
from pydub import AudioSegment
import os
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import math
from yt_dlp import YoutubeDL
import logging
from functools import lru_cache
from dotenv import load_dotenv
import time
import pymysql.cursors
from collections import OrderedDict
from pymongo import MongoClient
import youtube_dl
import re

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Model setup
MODEL_NAME = "google/pegasus-xsum"

# MySQL database configuration with pymysql
db = pymysql.connect(
    host=os.getenv('DB_HOST'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME'),
    cursorclass=pymysql.cursors.DictCursor  # Use DictCursor for easier dictionary access
)

# MongoDB database configuration
mongo_host = os.getenv('MONGO_HOST')
mongo_db_name = os.getenv('MONGO_DB_NAME')
mongo_collection_name = os.getenv('MONGO_COLLECTION_NAME')

mongo_client = MongoClient(mongo_host)
mongo_db = mongo_client[mongo_db_name]
mongo_collection = mongo_db[mongo_collection_name]

# Function to create table if not exists
def create_table_if_not_exists():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS data (
        sno INT AUTO_INCREMENT PRIMARY KEY,
        file VARCHAR(255) NOT NULL,
        transcription TEXT,
        summary TEXT,
        upload_time DATETIME NOT NULL
    )
    """
    with db.cursor() as cursor:
        cursor.execute(create_table_query)
        db.commit()

# Function to convert audio to WAV format
def convert_to_wav(audio_file_path):
    try:
        logging.info(f"Converting {audio_file_path} to WAV...")
        file_name, file_extension = os.path.splitext(audio_file_path)
        if file_extension.lower() != '.wav':
            audio = AudioSegment.from_file(audio_file_path)
            wav_file_path = f"{file_name}.wav"  # Ensure correct extension
            audio.export(wav_file_path, format="wav")
            logging.info(f"Conversion successful. WAV file saved at {wav_file_path}")
            return wav_file_path
        return audio_file_path
    except Exception as e:
        logging.error(f"Error converting audio to WAV: {e}")
        raise ValueError(f"Error converting audio to WAV: {e}")

# Function to load Whisper model
@lru_cache(maxsize=1)
def load_whisper_model():
    return whisper.load_model("base")

# Function to load Pegasus model
@lru_cache(maxsize=1)
def load_pegasus_model():
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

# Function to transcribe audio using Whisper
def transcribe_audio_with_whisper(audio_file_path):
    try:
        logging.info(f"Transcribing audio file: {audio_file_path}")
        model = load_whisper_model()
        if not os.path.exists(audio_file_path):
            raise ValueError(f"File {audio_file_path} not found.")
        result = model.transcribe(audio_file_path)
        return result["text"]
    except Exception as e:
        logging.error(f"Error in audio transcription: {e}")
        raise ValueError(f"Error in audio transcription: {e}")

# Function to summarize text using Pegasus
def summarize_text_with_pegasus(text, tokenizer, model):
    try:
        inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
        total_tokens = len(inputs["input_ids"][0])
        min_summary_length = max(math.ceil(total_tokens / 3), 30)
        max_summary_length = max(math.ceil(total_tokens / 2), 120)

        if min_summary_length >= max_summary_length:
            min_summary_length = max_summary_length - 1

        summary_ids = model.generate(
            inputs.input_ids,
            num_beams=5,
            min_length=min_summary_length,
            max_length=max_summary_length,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logging.error(f"Error in text summarization: {e}")
        raise ValueError(f"Error in text summarization: {e}")

# Function to download audio from YouTube using yt_dlp
def download_audio_from_youtube(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded_audio.%(ext)s'  # Generic name for downloaded audio
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_file_path = ydl.prepare_filename(info)
            audio_file_path = audio_file_path.replace('.webm', '.wav')
            return audio_file_path
    except youtube_dl.utils.DownloadError as e:
        logging.error(f"Error downloading audio from YouTube: {e}")
        raise ValueError(f"Error downloading audio from YouTube: {e}")
    except Exception as e:
        logging.error(f"Unexpected error downloading audio: {e}")
        raise ValueError(f"Unexpected error downloading audio: {e}")

# Function to check allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'aac', 'flac', 'm4a'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to remove repeated sentences
def remove_repeated_sentences(text):
    sentences = re.split(r'[,.]\s*', text)

    unique_sentences = list(OrderedDict.fromkeys(sentences))
    return '. '.join(unique_sentences)

# Function to check MongoDB connection
def check_mongodb_connection():
    try:
        mongo_client.server_info()  # This will trigger an exception if not connected
        return True
    except Exception as e:
        logging.error(f"MongoDB connection failed: {e}")
        return False

# Route to render index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle transcription and summarization
@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        create_table_if_not_exists()  # Ensure table exists
        
        # Check MongoDB connection before proceeding
        if os.path.exists("downloaded_audio.wav"):
            os.remove("downloaded_audio.wav")
        if os.path.exists("uploaded_audio.wav"):
            os.remove("uploaded_audio.wav")

        if not check_mongodb_connection():
            return jsonify({"error": "Failed to connect to MongoDB."}), 500
        
        if 'url' in request.form and request.form['url']:
            youtube_url = request.form['url']
            audio_file_path = download_audio_from_youtube(youtube_url)
            file_data = youtube_url  # Use YouTube URL as file data
        elif 'file' in request.files:
            audio_file = request.files['file']
            if not audio_file.filename:
                return jsonify({"error": "No file selected."}), 400
            if not allowed_file(audio_file.filename):
                return jsonify({"error": "Invalid file type. Please upload an audio file."}), 400
            audio_file_path = "uploaded_audio.wav"  # Fixed name for uploaded audio
            audio_file.save(audio_file_path)
            file_data = audio_file.filename  # Use filename as file data
        else:
            return jsonify({"error": "No audio file or URL provided."}), 400
        
        audio_file_path = convert_to_wav(audio_file_path)  # Ensure the file is in WAV format

        transcription = transcribe_audio_with_whisper(audio_file_path)
        if transcription:
            transcription = remove_repeated_sentences(transcription)
            tokenizer, model = load_pegasus_model()
            summary = summarize_text_with_pegasus(transcription, tokenizer, model)
            summary = remove_repeated_sentences(summary)
            
            # Save transcription and summary to MySQL database
            insert_query = "INSERT INTO data (file, transcription, summary, upload_time) VALUES (%s, %s, %s, %s)"
            upload_time = time.strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp
            
            with db.cursor() as cursor:
                cursor.execute(insert_query, (file_data, transcription, summary, upload_time))
                db.commit()
            
            # Save transcription and summary to MongoDB
            mongo_document = {
                "file": file_data,
                "transcription": transcription,
                "summary": summary,
                "upload_time": upload_time
            }
            mongo_collection.insert_one(mongo_document)
            if os.path.exists("downloaded_audio.wav"):
                os.remove("downloaded_audio.wav")
            if os.path.exists("uploaded_audio.wav"):
                os.remove("uploaded_audio.wav")
            return jsonify({"transcription": transcription, "summary": summary})
        else:
            return jsonify({"error": "Transcription failed."}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True)

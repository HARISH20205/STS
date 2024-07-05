from flask import Flask, request, jsonify, render_template
import whisper
from pydub import AudioSegment
import os
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import math
from pytube import YouTube
import validators
import logging
from functools import lru_cache
from dotenv import load_dotenv
import time
import mysql.connector
from collections import OrderedDict
from pymongo import MongoClient

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Model setup
MODEL_NAME = "google/pegasus-multi_news"

# MySQL database configuration
db = mysql.connector.connect(
    host=os.getenv('DB_HOST'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME')
)
cursor = db.cursor()

# MongoDB database configuration
mongo_client = MongoClient(os.getenv('MONGO_URI'))
mongo_db = mongo_client[os.getenv('MONGO_DB_NAME')]
mongo_collection = mongo_db[os.getenv('MONGO_COLLECTION_NAME')]

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
    cursor.execute(create_table_query)
    db.commit()

# Function to convert audio to WAV format
def convert_to_wav(audio_file_path):
    try:
        logging.info(f"Converting {audio_file_path} to WAV...")
        file_name, file_extension = os.path.splitext(audio_file_path)
        if file_extension.lower() != '.wav':
            audio = AudioSegment.from_file(audio_file_path)
            wav_file_path = f"{file_name}.wav"
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
        wav_file_path = convert_to_wav(audio_file_path)
        if wav_file_path:
            result = model.transcribe(wav_file_path)
            return result["text"]
        return None
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
            num_beams=6,
            min_length=min_summary_length,
            max_length=max_summary_length,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logging.error(f"Error in text summarization: {e}")
        raise ValueError(f"Error in text summarization: {e}")

# Function to download audio from YouTube
def download_audio_from_youtube(url, retries=3, backoff_factor=0.3):
    attempt = 0
    while attempt < retries:
        try:
            if validators.url(url):
                yt = YouTube(url)
                audio_stream = yt.streams.filter(only_audio=True).first()
                if (audio_stream):
                    output_file_path = audio_stream.download(filename="downloaded_audio")
                    return output_file_path
                else:
                    raise ValueError("No audio streams found for the YouTube video.")
            else:
                raise ValueError("Invalid YouTube URL provided.")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            time.sleep(backoff_factor * (2 ** attempt))
            if attempt == retries:
                raise ValueError(f"Error downloading audio from YouTube: {e}")

# Function to check allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'aac', 'flac', 'm4a'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to remove repeated sentences
def remove_repeated_sentences(text):
    sentences = text.split('. ')
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
            audio_file_path = f"uploaded_audio.{audio_file.filename.split('.')[-1]}"
            audio_file.save(audio_file_path)
            file_data = audio_file.filename  # Use filename as file data
        else:
            return jsonify({"error": "No audio file or URL provided."}), 400
        
        transcription = transcribe_audio_with_whisper(audio_file_path)
        if transcription:
            transcription = remove_repeated_sentences(transcription)
            tokenizer, model = load_pegasus_model()
            summary = summarize_text_with_pegasus(transcription, tokenizer, model)
            summary = remove_repeated_sentences(summary)
            
            # Save transcription and summary to MySQL database
            insert_query = "INSERT INTO data (file, transcription, summary, upload_time) VALUES (%s, %s, %s, %s)"
            upload_time = time.strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp
            
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

from flask import Flask, request, jsonify, render_template
import whisper
from pydub import AudioSegment
import os
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import math
from yt_dlp import YoutubeDL
import logging
from functools import lru_cache
from dotenv import load_dotenv
import time
import re

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Model setup
MODEL_NAME = "google/pegasus-xsum"

# Function to convert audio to MP3 format
def convert_to_mp3(audio_file_path):
    try:
        logging.info(f"Converting {audio_file_path} to MP3...")
        file_name, file_extension = os.path.splitext(audio_file_path)
        if file_extension.lower() != '.mp3':
            audio = AudioSegment.from_file(audio_file_path)
            mp3_file_path = f"{file_name}.mp3"  # Ensure correct extension
            audio.export(mp3_file_path, format="mp3")
            logging.info(f"Conversion successful. MP3 file saved at {mp3_file_path}")
            return mp3_file_path
        return audio_file_path
    except Exception as e:
        logging.error(f"Error converting audio to MP3: {e}")
        raise ValueError(f"Error converting audio to MP3: {e}")

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
        min_summary_length = max(math.ceil(total_tokens / 4), 75)  
        max_summary_length = max(math.ceil(total_tokens / 3), 200) 

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
        summary = remove_repeated_sentences(summary)  # Remove repeated sentences from summary
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
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded_audio.%(ext)s'  # Generic name for downloaded audio
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_file_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            return audio_file_path
    except Exception as e:
        logging.error(f"Unexpected error downloading audio: {e}")
        raise ValueError(f"Error downloading audio from YouTube: {e}")

# Function to check allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp3', 'aac', 'flac', 'm4a'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to remove repeated sentences
def remove_repeated_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split by sentence-ending punctuation
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        normalized_sentence = sentence.lower().strip()
        if normalized_sentence not in seen_sentences:
            unique_sentences.append(sentence)
            seen_sentences.add(normalized_sentence)
    
    return ' '.join(unique_sentences)


# Route to render index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle transcription and summarization
@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Clean up any existing files
        if os.path.exists("downloaded_audio.mp3"):
            os.remove("downloaded_audio.mp3")
        if os.path.exists("uploaded_audio.mp3"):
            os.remove("uploaded_audio.mp3")

        if 'url' in request.form and request.form['url']:
            youtube_url = request.form['url']
            audio_file_path = download_audio_from_youtube(youtube_url)
        elif 'file' in request.files:
            audio_file = request.files['file']
            if not audio_file.filename:
                return jsonify({"error": "No file selected."}), 400
            if not allowed_file(audio_file.filename):
                return jsonify({"error": "Invalid file type. Please upload an audio file."}), 400
            audio_file_path = "uploaded_audio.mp3"  # Fixed name for uploaded audio
            audio_file.save(audio_file_path)
        else:
            return jsonify({"error": "No audio file or URL provided."}), 400
        
        audio_file_path = convert_to_mp3(audio_file_path)  # Ensure the file is in MP3 format

        transcription = transcribe_audio_with_whisper(audio_file_path)
        if transcription:
            tokenizer, model = load_pegasus_model()
            summary = summarize_text_with_pegasus(transcription, tokenizer, model)
            
            # Clean up files
            if os.path.exists("downloaded_audio.mp3"):
                os.remove("downloaded_audio.mp3")
            if os.path.exists("uploaded_audio.mp3"):
                os.remove("uploaded_audio.mp3")
                
            return jsonify({"transcription": transcription, "summary": summary})
        else:
            return jsonify({"error": "Transcription failed."}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True,port=7860)

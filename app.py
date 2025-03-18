from flask import Flask, request, jsonify, render_template
import whisper
from pydub import AudioSegment
import os
import io
import numpy as np
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import math
from yt_dlp import YoutubeDL
import logging
from functools import lru_cache
from dotenv import load_dotenv
import time
import re
import tempfile

load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "google/pegasus-xsum"

def convert_audio_to_mp3(audio_bytes, original_format=None):
    try:
        logging.info(f"Converting audio from {original_format} to MP3 in memory...")
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=original_format)
        buffer = io.BytesIO()
        audio.export(buffer, format="mp3")
        buffer.seek(0)
        logging.info("Conversion successful")
        return buffer
    except Exception as e:
        logging.error(f"Error converting audio to MP3: {e}")
        raise ValueError(f"Error converting audio to MP3: {e}")

@lru_cache(maxsize=1)
def load_whisper_model():
    return whisper.load_model("base")

@lru_cache(maxsize=1)
def load_pegasus_model():
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

def transcribe_audio_with_whisper(audio_data):
    try:
        logging.info("Transcribing audio data")
        model = load_whisper_model()
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_file:
            if isinstance(audio_data, io.BytesIO):
                temp_file.write(audio_data.getvalue())
            else:
                temp_file.write(audio_data)
            temp_file.flush()
            
            result = model.transcribe(temp_file.name)
        
        return result["text"]
    except Exception as e:
        logging.error(f"Error in audio transcription: {e}")
        raise ValueError(f"Error in audio transcription: {e}")

def summarize_text_with_pegasus(text, tokenizer, model):
    try:
        inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
        total_tokens = len(inputs["input_ids"][0])
        min_summary_length = max(math.ceil(total_tokens / 4), 100)  
        max_summary_length = max(math.ceil(total_tokens / 3), 200) 

        if min_summary_length >= max_summary_length:
            min_summary_length = max_summary_length - 1

        summary_ids = model.generate(
            inputs.input_ids,
            num_beams=3,
            min_length=min_summary_length,
            max_length=max_summary_length
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary = remove_repeated_sentences(summary)
        return summary
    except Exception as e:
        logging.error(f"Error in text summarization: {e}")
        raise ValueError(f"Error in text summarization: {e}")

def download_audio_from_youtube(url):
    buffer = io.BytesIO()
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': '-',
        'logtostderr': True,
        'quiet': True,
        'no_warnings': True,
        'extract_audio': True,
    }
    
    try:
        logging.info(f"Downloading audio from YouTube: {url}")
        with tempfile.NamedTemporaryFile(suffix=".%(ext)s") as temp_file:
            ydl_opts['outtmpl'] = temp_file.name

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_file_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
                
                with open(audio_file_path, 'rb') as audio_file:
                    buffer = io.BytesIO(audio_file.read())
                    buffer.seek(0)
        
        return buffer
    except Exception as e:
        logging.error(f"Unexpected error downloading audio: {e}")
        raise ValueError(f"Error downloading audio from YouTube: {e}")

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp3', 'aac', 'flac', 'm4a'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_repeated_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)  
    unique_sen = []
    seen_sen = set()

    for sentence in sentences:
        normalized_sentence = sentence.lower().strip()
        if normalized_sentence not in seen_sen:
            unique_sen.append(sentence)
            seen_sen.add(normalized_sentence)
    
    return ' '.join(unique_sen)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        audio_data = None
        
        if 'url' in request.form and request.form['url']:
            youtube_url = request.form['url']
            audio_data = download_audio_from_youtube(youtube_url)
        elif 'file' in request.files:
            audio_file = request.files['file']
            if not audio_file.filename:
                return jsonify({"error": "No file selected."}), 400
            if not allowed_file(audio_file.filename):
                return jsonify({"error": "Invalid file type. Please upload an audio file."}), 400
            
            audio_bytes = audio_file.read()
            file_format = audio_file.filename.rsplit('.', 1)[1].lower()
            audio_data = convert_audio_to_mp3(audio_bytes, original_format=file_format)
        else:
            return jsonify({"error": "No audio file or URL provided."}), 400
        
        transcription = transcribe_audio_with_whisper(audio_data)
        
        if transcription:
            tokenizer, model = load_pegasus_model()
            summary = summarize_text_with_pegasus(transcription, tokenizer, model)
            return jsonify({"transcription": transcription, "summary": summary})
        else:
            return jsonify({"error": "Transcription failed."}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=7860)
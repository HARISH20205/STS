# Speech-to-Text Summarization

## 📜 Introduction
Welcome to the Speech-to-Text Summarization Project! This nifty tool allows you to transcribe audio files or YouTube videos into text and then summarize that text for quick and easy reading. Perfect for busy bees and multitaskers who want to convert lengthy audio into concise summaries!

## ✨ Features
- Audio Transcription: Converts spoken words from audio files or YouTube videos into text using Whisper.
- Text Summarization: Summarizes the transcribed text into a shorter, more digestible format using Pegasus.
- Web Interface: Sleek and easy-to-use web interface for uploading audio files or entering YouTube URLs.
- Database Storage: Stores transcriptions and summaries in both MySQL and MongoDB databases for future reference.

## 🎯 Use Cases
- Students: Quickly summarize lectures and study materials.
- Journalists: Transcribe and summarize interviews and speeches.
- Podcasters: Create show notes and summaries for episodes.

## 🚀 Installation Guide

### 🛠 Prerequisites
Before you dive in, make sure you have the following:
- Python 3.8 or higher
- MySQL database setup
- MongoDB setup
- FFmpeg installed (required for audio processing)
- Basic knowledge of Flask and web development

### 📦 Setup Instructions
1. Clone the Repository:
   ```sh
   git clone https://github.com/HARISH20205/STS.git
   cd STS
   ```

2. Install Dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Install FFmpeg:
   Follow the instructions for your operating system from the [FFmpeg website](https://ffmpeg.org/download.html).

4. Configure Environment Variables:
   Create a `.env` file in the root directory with your MySQL and MongoDB database credentials:
   ```sh
   DB_HOST=your_db_host
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_NAME=your_db_name
   MONGO_URI=your_mongo_uri
   MONGO_DB_NAME=your_mongo_db_name
   MONGO_COLLECTION_NAME=your_mongo_collection_name
   ```

5. Run the Application:
   ```sh
   python app.py
   ```
   The application should now be running at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## 📋 Usage Instructions

### 🌐 Running the Application
Fire up your web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000). You'll be greeted with the main interface where you can either upload an audio file or enter a YouTube URL.

### 🖱 User Interface Guide
- Upload Audio File: Click the file input field to select an audio file from your device.
- Enter YouTube URL: Paste a YouTube URL in the provided text field.
- Submit: Click the "Transcribe and Summarize" button to start the magic.
- Results: The transcription and summary will appear on the right side once processing is complete.

## 🔍 Detailed Functionality

### 🎤 Audio Transcription
The project uses Whisper to convert audio into text. Whisper is a state-of-the-art speech recognition model that ensures accurate transcriptions, capturing even the smallest nuances of speech.

### 📝 Text Summarization
Pegasus, a transformer model designed for summarization, is used to create concise summaries of the transcribed text. The model smartly balances between truncating too little and summarizing too aggressively, ensuring you get the essence without missing the details.

### 🗄 Database Integration
Transcriptions and summaries are stored in both MySQL and MongoDB databases. The MySQL table holds the following columns:
- `sno`: Serial number (auto-incremented)
- `file`: Name of the audio file or YouTube URL
- `transcription`: The transcribed text
- `summary`: The summarized text
- `upload_time`: Timestamp of when the data was uploaded

The MongoDB collection stores the documents with similar fields for easy retrieval and scalability.

## 🗂 Code Structure

### Project Structure
```vbnet
speech-to-text-summarization/
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── scripts.js
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── .env
```

### Main Files and Directories
- app.py: The main Flask application file containing all the logic.
- templates/: HTML templates for the web interface.
- static/: Static files like CSS and JavaScript for styling and functionality.

## 🌐 API Documentation

### Endpoints
- `GET /`: Renders the main page.
- `POST /transcribe`: Handles the transcription and summarization process.

### Example Request
```sh
curl -X POST http://127.0.0.1:5000/transcribe -F "file=@path/to/your/audiofile.mp3"
```

### Example Response
```json
{
  "transcription": "Your transcribed text here...",
  "summary": "Your summarized text here..."
}
```

## ⚙️ Configuration Details

### Environment Variables
- `DB_HOST`: MySQL database host.
- `DB_USER`: MySQL database username.
- `DB_PASSWORD`: MySQL database password.
- `DB_NAME`: MySQL database name.
- `MONGO_URI`: MongoDB connection URI.
- `MONGO_DB_NAME`: MongoDB database name.
- `MONGO_COLLECTION_NAME`: MongoDB collection name.

### Configuration Files
- `.env`: Stores environment variables for database configuration.

## 🚑 Error Handling

### Common Issues
- Invalid File Type: Ensure you're uploading a supported audio file format (wav, mp3, aac, flac, m4a).
- Transcription Errors: Check the logs for detailed error messages.

### Logs
Logs can be found in the console output, providing detailed information on the application's status and errors.

## 🏎 Performance Optimization
- Model Loading Times: Whisper and Pegasus Models are cached using `functools.lru_cache` to minimize loading times.
- Efficient Data Handling: Only necessary parts of the audio are processed, reducing computational load.

## 🔧 Customization and Extensibility

### How to Train Models
To train Whisper and Pegasus on new data, refer to their respective documentation and follow the training procedures. Customize the models to suit your specific needs.

### Adding Features
1. Clone the Repository.
2. Create a New Branch for your feature.
3. Implement the Feature in the appropriate files.
4. Test and Commit your changes.
5. Submit a Pull Request with a detailed description of your feature.

### Credits
- Whisper: OpenAI
- Pegasus: Google Research
- PyDub: Audio processing
- Pytube: YouTube audio download
- Flask: Web framework

### Maintainers
- Harish KB : harishkb20205@gmail.com

## 📚 Additional Resources

### References
- Whisper Documentation
- Pegasus Documentation

### Further Reading
- Flask Documentation
- PyDub Documentation
- Pytube Documentation

Enjoy transcribing and summarizing like never before! If you have any questions or need help, don't hesitate to reach out. Happy coding! 🎉

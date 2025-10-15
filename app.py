import os
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file for local testing
# On Render, this key is set directly in the dashboard
load_dotenv()
client = OpenAI()

app = Flask(__name__)
# Render's free tier has an ephemeral file system; using /tmp is a good practice.
# MAX_CONTENT_LENGTH is 100MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

SUMMARY_MODEL = "gpt-4-turbo-preview"
SUMMARY_PROMPT = (
    "You are an expert meeting minutes generator. "
    "Analyze the following meeting transcript. "
    "Output structured: Summary, Key Decisions, Action Items."
)

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        # Note: The free tier may have resource limits for very long files.
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
    return transcript

def generate_summary(transcript):
    full_prompt = SUMMARY_PROMPT + "\n\n" + transcript
    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[{"role":"system","content":"You are a professional meeting summarizer."},
                  {"role":"user","content":full_prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

@app.route('/status', methods=['GET'])
def status():
    # Basic health check route
    return jsonify({"status":"ok","message":"Meeting Summarizer API running"})

@app.route('/summarize', methods=['POST'])
def summarize_meeting():
    if 'audio_file' not in request.files:
        return jsonify({"error":"No 'audio_file' part"}), 400
    
    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({"error":"No selected file"}), 400
    
    # Use /tmp/ for temporary file storage on deployment platforms
    # The uploaded file name is used to ensure a unique name
    temp_path = os.path.join("/tmp", audio_file.filename)
    audio_file.save(temp_path)
    
    try:
        transcript = transcribe_audio(temp_path)
        summary = generate_summary(transcript)
        # Delete the temporary file immediately
        os.remove(temp_path)
    except Exception as e:
        # Ensure the file is deleted even if the transcription/summary fails
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # Log the error on the server and return a generic client error
        print(f"Error during processing: {e}") 
        return jsonify({"error": "An error occurred during processing. Check server logs."}), 500
    
    return jsonify({"transcript": transcript, "meeting_summary": summary})

# Remove the __main__ block for production deployment with Gunicorn
# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, redirect, url_for, render_template, request, jsonify
import openai

# from jinja2 import Environment, PackageLoader, select_autoescape
import re
import html
import json

# Database
# from flask_sqlalchemy import SQLAlchemy
# from sqlalchemy.sql import func

#OCR --> pip install pytesseract Pillow
import pytesseract
from PIL import Image
#speaker --> pip install gtts, python -m pip install playsound
import gtts
from playsound import playsound
#speech transcriber -- mic --> pip install SpeechRecognition
import speech_recognition as sr
import math
#speech transcriber -- audiofile --> pip install pydub
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

# encryption
# from Crypto.Cipher import AES
# from Crypto.Protocol.KDF import PBKDF2
# from Crypto.Random import get_random_bytes
# import base64

#pip install python-dotenv
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment
openai.api_key = os.getenv("API_KEY")

def update_api_key(new_api_key):
    # Update the API key environment variable
    os.environ["API_KEY"] = new_api_key

#app start
app = Flask(__name__)

'''def set_api_key(new_api_key):
    os.environ['OPENAI_API_KEY'] = new_api_key
    openai.api_key = new_api_key

default_key = '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'
set_api_key(default_key)'''

# openai.api_key = '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'

@app.route("/")
def main():
    return render_template("home.html")

prompt_custom = ''
prompt_default = 'You are a helpful tutor named DILL who always asks follow-up questions and suggests example questions for the user to ask. When given information, you remember it and become an expert on it. Ask the user to upload their textbook or external text for you to learn. Ask questions like "would you like me to summarize this information" and "would you like me to take notes on this topic".'
prompt = 'Act as an AI writing tutor named Shreyas. I will provide you with a student who needs help improving their writing. Your task is to use artificial intelligence tools, such as natural language processing, to give the student feedback on improving their composition. You should also use your rhetorical knowledge and experience with effective writing techniques to suggest ways the student can better express their thoughts and ideas in written form. My first request is I need somebody to help me edit my masters thesis. Be entertaining and use emojis.'
prompt2 = 'Act as an AI history tutor named Chloe. I will provide you with a student who needs help improving their history knowledge and skills. Your task is to use artificial intelligence tools, such as natural language processing, to help students understand historical concepts and facts. Help the student work on their historical analytic abilities (such as analyzing the long-term effect, short-term effects, author, audience, point of view, and turning points) about events they have questions about. Additionally, help them compare events, including both similarities and differences. Be entertaining while giving clear and accurate answers. It is very important that you always ask follow-up questions. Use emojis.'
prompt3 = 'Act as an AI elementary school tutor named Sohum. Do not refer to yourself as Mr. Sohum or in other formal ways. I will provide you with a student who needs help improving their general learning skills and a variety of subjects. Your task is to use artificial intelligence tools, such as natural language processing, to help students understand the concepts they have trouble with. Provide clear and accurate answers. Make sure your answers are short, not overwhelming, use examples, and can be understood by elementary school students. It is very important that you always ask follow-up questions. Be entertaining and use emojis.'
prompt4 = 'Act as an AI science tutor named Steven. I will provide you with a student who needs help improving their science knowledge and skills. Your task is to use artificial intelligence tools, such as natural language processing, to help students understand complex scientific concepts and improve their knowledge in subjects like physics, chemistry, biology, and environmental science. Be patient and give clear explanations, accurate answers, and practical examples. If you are unable to provide an accurate answer, such as with math problems, say so. It is very important that you always ask follow-up questions. Be entertaining and use emojis. '
messages = []
conversation_history = []

def generate_response(message, prompt1):
    global messages
    try:
        # Add user message to the conversation history
        conversation_history.append({'role': 'user', 'content': message})
        if prompt1 == '':
            prompt1 = prompt_default
        print(prompt1)
        # Create a list of messages for the OpenAI API, including the conversation history
        messages = [
            {
                'role': 'system',
                'content': prompt1
            }
        ] + conversation_history

        # Call the OpenAI API to generate a response
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k',
            messages=messages,
            max_tokens=2049,
            temperature=0.7,
            n=1,
            stop=None
        )

        # Extract the assistant's reply from the API response and format it
        reply = response.choices[0].message['content']
        # reply = format_bot_response(reply)

        # Add AI reply to the conversation history
        conversation_history.append({'role': 'assistant', 'content': reply})

        if prompt1 == 'Act as a content writer, use the given parameters to write a professional essay in full sentences.':
            prompt1 = ''
        return reply

    except(openai.error.AuthenticationError):
        api_error = "ERROR"
        return api_error

@app.route('/send-message', methods=['POST'])
def send_message():
    print(prompt_custom)
    message = request.json['message']
    reply = generate_response(message, prompt_custom)
    return jsonify({'message': reply})

@app.route('/customize-tutor', methods=['POST'])
def customize_tutor():
    global prompt_custom
    prompt_custom = request.json['message']
    print(prompt_custom)

    if prompt_custom == '0':
        prompt_custom = ''
        return jsonify({'message': 'DILL'})
    elif prompt_custom == '1':
        prompt_custom = prompt
        return jsonify({'message': 'Shreyas'})
    elif prompt_custom == '2':
        prompt_custom = prompt2
        return jsonify({'message': 'Chloe'})
    elif prompt_custom == '3':
        prompt_custom = prompt3
        return jsonify({'message': 'Sohum'})
    elif prompt_custom == '4':
        prompt_custom = prompt4
        return jsonify({'message': 'Steven'})
    else:
        return jsonify({'message': 'Kate'})
    print(prompt_custom)

@app.route('/read-message', methods=['POST'])
def read_message():
    data = request.get_json()
    newest_message = data.get('message', '')
    # Process the newest message as needed
    print("Received newest message from the frontend:", newest_message)
    tts = gtts.gTTS(newest_message)
    tts.save("sample.mp3")
    playsound("sample.mp3")
    return jsonify({'message': 'done'})

@app.route('/upload', methods=['POST'])
def upload_file():
    global messages, prompt_custom
    print('hi')
    # Get the uploaded file content from the POST request
    uploaded_file = request.files['filename']
    print(uploaded_file)
    allowed_image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
    allowed_audio_extensions = {'wav', 'mp3'}
    allowed_history_restore = {'txt'}
    extension = uploaded_file.filename.split('.')[-1].lower()
    
    if extension in allowed_image_extensions:
        try:
            print("hello")
            image = Image.open(uploaded_file)
            response = pytesseract.image_to_string(image)
            print(response)
            if response == '':
                response = 'No text detected.'
        except Image.UnidentifiedImageError:
            response = "Error: Unidentified image format."
    elif extension in allowed_audio_extensions:
        response = get_large_audio_transcription_on_silence(uploaded_file)
    elif extension in allowed_history_restore:
        response = json.loads(uploaded_file.read())
        messages = response
        prompt_custom = response[0]["content"]
    else:
        response = "Error: Unsupported file format."
    
    return response

# Function to transcribe speech in the audio file
def transcribe_audio(path):
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio_listened = r.record(source)
        text = r.recognize_google(audio_listened)  # convert to text
    return text

# Function to split the audio file into chunks on silence and apply speech recognition
def get_large_audio_transcription_on_silence(uploaded_file):
    # Save the audio file temporarily to process it
    audio_path = "temp_audio.wav"
    uploaded_file.save(audio_path)
    
    # Open the audio file using pydub
    sound = AudioSegment.from_file(audio_path)
    
    # Split audio sound where silence is >= 500 milliseconds and get chunks
    chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=sound.dBFS - 14, keep_silence=500)
    
    # Process each chunk
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        # Export audio chunk and save it
        chunk_file = f"audio-chunk{i}.wav"
        audio_chunk.export(chunk_file, format="wav")
        
        # Recognize the chunk
        try:
            text = transcribe_audio(chunk_file)
        except sr.UnknownValueError as e:
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            print(chunk_file, ":", text)
            whole_text += text

        # Delete the temporary chunk file
        os.remove(chunk_file)

    # Delete the temporary audio file
    os.remove(audio_path)
    
    # Return the whole transcribed text
    return whole_text
    
@app.route('/download', methods=['GET'])
def download():
    global messages
    return jsonify(messages)

@app.route('/api', methods=['POST'])
def api():
    new_key = request.json['message']
    update_api_key(new_key)
    openai.api_key = os.getenv("API_KEY")
    key_status = generate_response('hello, what is your name?', prompt_custom)
    if key_status == "ERROR":
        default_key = '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'
        update_api_key(default_key)
        openai.api_key = os.getenv("API_KEY")
        return 'Default key restored'
    else:
    # encrypted_secret_key = request.json['message']
    # password = "P@ssw0rd$-EnCrypT1on"
    # # Derive a key from the password
    # salt = get_random_bytes(16)
    # key = PBKDF2(password, salt, dkLen=32, count=1000000)

    # # Decode the encrypted secret key from base64
    # encrypted_bytes = base64.b64decode(encrypted_secret_key)

    # # Decrypt the secret key
    # cipher = AES.new(key, AES.MODE_EAX, nonce=encrypted_bytes[:16])
    # decrypted_secret_key_bytes = cipher.decrypt(encrypted_bytes[16:])
    # decoded_string = decrypted_secret_key_bytes.decode('utf-8')

    # print(decoded_string)
        return jsonify(key_status)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

    # ssl_context='adhoc'

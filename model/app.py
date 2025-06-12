from flask import Flask, request, jsonify, render_template
import os
import uuid
import requests
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import torch
from collections import Counter
from kobert_tokenizer import KoBERTTokenizer

from model import ImageEmotionClassifier, TextEmotionClassifier, VideoModel
from predict import VideoPredictor, predict_text, predict_folder_emotion, predict_video, predict_final_emotion, load_image_model, load_text_model, adjust_label_order
from preprocessing import preprocess_image, preprocess_text, VideoPreprocessor
from emotion_ensemble import EmotionEnsemble
from flask_cors import CORS
import base64
import io
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

device = "cuda" if torch.cuda.is_available() else "cpu"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_dotenv()

SCOPES = 'streaming user-read-email user-read-private user-modify-playback-state user-read-playback-state'

SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

spotify = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
)

EMOTION_TRACKS = {
    "anger": [
        "39gaUtq2z4ejJbno7tWHbL",  
        "4m9Badp5xjHRXcPjsSPqHk",  
        "49L0nTFWXcQzXvhHJ4a5VW",
        "1Vu0QfGrvtG2DooEgi6XYW",   
        "1j9Y2na5KDqntYChmIRUGf",
    ],
    "joy": [
        "39gaUtq2z4ejJbno7tWHbL", 
        "4m9Badp5xjHRXcPjsSPqHk",  
        "49L0nTFWXcQzXvhHJ4a5VW",
        "1Vu0QfGrvtG2DooEgi6XYW",
        "1j9Y2na5KDqntYChmIRUGf",    
    ],
    "sadness": [
        "39gaUtq2z4ejJbno7tWHbL",  
        "4m9Badp5xjHRXcPjsSPqHk", 
        "49L0nTFWXcQzXvhHJ4a5VW",
        "1Vu0QfGrvtG2DooEgi6XYW",
        "1j9Y2na5KDqntYChmIRUGf", 
    ]
}

SPOTIFY_ACCESS_TOKEN = None

def get_access_token():
    global SPOTIFY_ACCESS_TOKEN
    auth_url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(
        f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()
    ).decode("ascii")
    
    headers = {"Authorization": f"Basic {auth_header}"}
    payload = {"grant_type": "client_credentials"}
    
    response = requests.post(auth_url, data=payload, headers=headers)
    if response.status_code == 200:
        SPOTIFY_ACCESS_TOKEN = response.json()['access_token']
        print("New access token:", SPOTIFY_ACCESS_TOKEN)
        return SPOTIFY_ACCESS_TOKEN
    else:
        print(f"Error getting access token: {response.status_code}")
        return None

def get_track_by_emotion(emotion):
    try:
        track_id = EMOTION_TRACKS.get(emotion, EMOTION_TRACKS["joy"])[0]
        track = spotify.track(track_id)
        return {
            "bgm_name": track["name"],
            "artist": track["artists"][0]["name"],
            "album_cover": track["album"]["images"][0]["url"] if track["album"]["images"] else "",
            "preview_url": track["preview_url"],
            "spotify_url": track["external_urls"]["spotify"],
            "track_id": track_id
        }
    except Exception as e:
        print(f"Error getting track: {str(e)}")
        return None

@app.route('/')
def get_music():
    emotion = request.args.get('emotion', 'joy')
    track_info = get_track_by_emotion(emotion)
    if not track_info:
        return jsonify({"error": "No tracks available"}), 404
    return jsonify(track_info)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_upload_folder(file_type):
    if file_type == 'text':
        return os.path.join(app.config['UPLOAD_FOLDER'], 'text')
    elif file_type == 'image':
        return os.path.join(app.config['UPLOAD_FOLDER'], 'image')
    elif file_type == 'video':
        return os.path.join(app.config['UPLOAD_FOLDER'], 'video')
    else:
        return app.config['UPLOAD_FOLDER']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/all', methods=['POST'])
def predict_all():
    text_emotion = None
    if 'text' in request.form:
        text = request.form['text']
        if text:
            text_folder = get_upload_folder('text')
            os.makedirs(text_folder, exist_ok=True)
            text_filename = str(uuid.uuid4()) + '.txt'
            text_filepath = os.path.join(text_folder, text_filename)
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            text_emotion = str(predict_text(text_model, tokenizer, text))

    image_emotion = None
    if 'image' in request.files:
        file = request.files['image']
        if file and allowed_file(file.filename):
            image_folder = get_upload_folder('image')
            os.makedirs(image_folder, exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(image_folder, filename)
            file.save(filepath)
            image_emotion, _ = predict_folder_emotion(image_model, image_folder)

    video_emotion = None
    if 'video' in request.files:
        file = request.files['video']
        if file and allowed_file(file.filename):
            video_folder = get_upload_folder('video')
            os.makedirs(video_folder, exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(video_folder, filename)
            file.save(filepath)
            frame_dir = os.path.join(video_folder, str(uuid.uuid4()))
            os.makedirs(frame_dir, exist_ok=True)
            _, video_emotion = predict_video(filepath, frame_dir)

    final_emotion = predict_final_emotion(text_emotion, image_emotion, video_emotion)
    return jsonify({
        "text_emotion":  text_emotion,
        "image_emotion": image_emotion,
        "video_emotion": video_emotion,
        "final_emotion": final_emotion
    })

@app.route('/api/analyze', methods=['GET', 'POST', 'OPTIONS'])
def analyze_media():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return response

    try:
        if request.method == 'POST':
            data = request.json
            media_files = data.get('mediaFiles', [])
            post_text = data.get('postText', '')
        elif request.method == 'GET':
            media_files = []
            post_text = request.args.get('postText', '')

        print(f"Received {len(media_files)} media files")

        temp_dir = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)

        image_emotions = []
        video_emotions = []

        for i, media in enumerate(media_files):
            print(f"Processing media {i+1}/{len(media_files)}")
            print(f"Media type: {media['type']}")

            try:
                file_data = media['data'].split(',')[1]
                file_bytes = base64.b64decode(file_data)
                
                ext = 'jpg' if media['type'] == 'image' else 'mp4'
                filename = secure_filename(f"{uuid.uuid4()}.{ext}")
                filepath = os.path.join(temp_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(file_bytes)
                print(f"Saved file to {filepath}")

                if media['type'] == 'image':
                    print("Analyzing image...")
                    emotion, _ = predict_folder_emotion(image_model, temp_dir)
                    print(f"Image emotion result: {emotion}")
                    if emotion:
                        image_emotions.append(emotion)
                elif media['type'] == 'video':
                    print("Analyzing video...")
                    frame_dir = os.path.join(temp_dir, 'frames')
                    os.makedirs(frame_dir, exist_ok=True)
                    _, emotion = predict_video(filepath, frame_dir)
                    print(f"Video emotion result: {emotion}")
                    if emotion:
                        video_emotions.append(emotion)

            except Exception as e:
                print(f"Error processing media file: {str(e)}")
                continue

        text_emotion = None
        if post_text:
            print("Analyzing text...")
            text_emotion = str(predict_text(text_model, tokenizer, post_text))
            print(f"Text emotion result: {text_emotion}")

        final_image_emotion = image_emotions[0] if image_emotions else None
        final_video_emotion = video_emotions[0] if video_emotions else None
        
        final_emotion = predict_final_emotion(
            text_emotion,
            final_image_emotion,
            final_video_emotion
        )

        print("Analysis complete")
        print(f"Final results - Text: {text_emotion}, Image: {final_image_emotion}, Video: {final_video_emotion}, Final: {final_emotion}")

        import shutil
        shutil.rmtree(temp_dir)

        return jsonify({
            'success': True,
            'result': {
                'textEmotion': text_emotion,
                'imageEmotion': final_image_emotion,
                'videoEmotion': final_video_emotion,
                'finalEmotion': final_emotion
            }
        })

    except Exception as e:
        print(f"Error in analyze_media: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/tracks/<emotion>')
def get_tracks(emotion):
    try:
        track_ids = EMOTION_TRACKS.get(emotion, EMOTION_TRACKS["joy"])
        selected_ids = random.sample(track_ids, min(3, len(track_ids))) if track_ids else []
        
        tracks = []
        for track_id in selected_ids:
            try:
                track = spotify.track(track_id)
                tracks.append({
                    "bgm_name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "album_cover": track["album"]["images"][0]["url"] if track["album"]["images"] else "",
                    "spotify_url": track["external_urls"]["spotify"],
                    "track_id": track_id
                })
            except Exception as e:
                print(f"Error processing track {track_id}: {str(e)}")
                continue
        
        return jsonify(tracks)
        
    except Exception as e:
        print(f"Error in get_tracks: {str(e)}")
        return jsonify([])

@app.route('/api/get-access-token', methods=['GET'])
def api_get_access_token():
    return jsonify({'access_token': None})

port = int(os.environ.get('PORT', 3002))
if __name__ == "__main__":
    image_model_path = os.path.join('models', 'final_resnet50_emotion_classifier_crawling_3.pth')
    text_model_path = os.path.join('models', 'trained_model_full_v2_3emotions_weights_new.pth')
    image_model = load_image_model(image_model_path)
    text_model = load_text_model(text_model_path)
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'text'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'image'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'video'), exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=port)

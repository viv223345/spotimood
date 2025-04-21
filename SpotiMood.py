import cv2
import numpy as np
from fer import FER
import requests
import json
import time
import os
from dotenv import load_dotenv
import webbrowser

load_dotenv()

def detect_mood(frame):
    emotion_detector = FER(mtcnn=True)
    emotions = emotion_detector.detect_emotions(frame)
    if not emotions:
        return "neutral"
    dominant_emotion = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
    emotion_name, confidence = dominant_emotion
    if emotion_name in ['happy', 'surprise']:
        return "happy"
    elif emotion_name in ['sad', 'fear', 'angry', 'disgust']:
        return "sad"
    else:
        return "neutral"
    
def get_song_recommendation(mood):
    prompt = f"Recommend one Spotify song for someone who is feeling {mood}. Only return the song name and artist in this format: 'Song Name by Artist'. No explanation or additional text. Dont repeat the same songs."
    payload = {
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(
        "https://ai.hackclub.com/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )
    if response.status_code == 200:
        try:
            result = response.json()
            song_recommendation = result.get('choices', [{}])[0].get('message', {}).get('content', "")
            return song_recommendation
        except Exception as e:
            print(f"Error parsing response: {e}")
            return "Error getting recommendation"
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "Error contacting recommendation service"
    
def open_spotify_search(song_recommendation):
    search_query = song_recommendation.replace(" ", "%20")
    spotify_url = f"https://open.spotify.com/search/{search_query}"
    webbrowser.open(spotify_url)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Welcome to SpotiMood.")
    print("Press 's' to scan your mood and get a song recommendation")
    print("Press 'q' to quit")
    last_mood = None
    last_recommendation = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        frame = cv2.flip(frame, 1)
        if last_mood:
            cv2.putText(frame, f"Mood: {last_mood}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if last_recommendation:
            words = last_recommendation.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 30:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            y_pos = 70
            for line in lines:
                cv2.putText(frame, line, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
        cv2.imshow('SpotiMood', frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            mood = detect_mood(frame)
            last_mood = mood
            print(f"Detected mood: {mood}")
            print("Getting song recommendation...")
            recommendation = get_song_recommendation(mood)
            last_recommendation = recommendation
            print(f"Recommendation: {recommendation}")
            open_spotify = input("Open in Spotify? (y/n): ")
            if open_spotify.lower() == 'y':
                open_spotify_search(recommendation)
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

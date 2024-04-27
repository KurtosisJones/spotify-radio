import time
import requests
from base64 import b64encode

def get_spotify_token(client_id, client_secret):
    # Encode as Base64
    credentials = b64encode(f"{client_id}:{client_secret}".encode()).decode('utf-8')

    headers = {
        "Authorization": f"Basic {credentials}"
    }

    body = {
        "grant_type": "client_credentials"
    }

    token_url = "https://accounts.spotify.com/api/token"
    response = requests.post(token_url, headers=headers, data=body)
    response_data = response.json()
    return response_data['access_token']

def safe_request(url:str, headers:dict, params:dict=None, max_retries:int=20):
    retry_wait = 1
    for i in range(max_retries):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', retry_wait))
            print(retry_after)
            print(f"Rate limit exceeded, retrying after {retry_after} seconds...")
            time.sleep(retry_after)
            retry_wait *= 2 
        else:
            response.raise_for_status()
    raise Exception("Max retries exceeded")

def search_tracks_by_genre(access_token:str, genre:str, limit:int=50, records:int=500):
    url = "https://api.spotify.com/v1/search"
    track_ids = []
    offset = 0

    while len(track_ids) < records:
        query_params = {
            "q": f"genre:{genre}",
            "type": "track",
            "limit": limit,
            "offset": offset
        }
        headers = {"Authorization": f"Bearer {access_token}"}
        
        response = safe_request(url, headers=headers, params=query_params)
        data = response.json()
        tracks = data.get('tracks', {}).get('items', [])
        
        if not tracks:
            break

        for track in tracks:
            track_ids.append(track['id'])
        
        offset += limit
        if len(tracks) < limit:
            break

    return track_ids

def get_audio_features_by_id(access_token, track_ids):
    audio_features = []
    batch_size = 100

    for start in range(0, len(track_ids), batch_size):
        end = start + batch_size
        batch_ids = track_ids[start:end]
        url = f"https://api.spotify.com/v1/audio-analysis?ids={','.join(batch_ids)}"
        headers = {"Authorization": f"Bearer {access_token}"}
        
        print(f"{url=}")
        response = safe_request(url, headers=headers)
        data = response.json()

        if 'error' in data:
            print("Error:", data['error']['message'])
            return None
        
        batch_features = data.get('audio_features', [])
        if batch_features:
            audio_features.extend(batch_features)
        else:
            print("No data received for batch:", batch_ids)

    return audio_features
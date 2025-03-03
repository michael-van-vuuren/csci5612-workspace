import requests
from utils.logger import logger
from api.spotify.authenticator import get_token

SEARCH_ENDPOINT = 'https://api.spotify.com/v1/search'
ARTIST_ENDPOINT = 'https://api.spotify.com/v1/artists/{}'

def extract_track_info(results, isrc):
    """Extract track name, album name, artist name, and artist ID from response."""
    items = results.get('tracks', {}).get('items', [])
    if items:
        top_result = items[0]
        return {
            'Track': top_result['name'],
            'Album Name': top_result['album']['name'],
            'Artist': top_result['artists'][0]['name'],
            'Artist ID': top_result['artists'][0]['id'],
            'ISRC': isrc
        }
    else:
        logger.warning(f'No results found for ISRC {isrc}')
        return None

def get_artist_details(artist_id):
    """Retrieve genres and number of followers for an artist."""
    if not artist_id:
        logger.warning("No artist ID provided for lookup.")
        return {"Genres": [], "Followers": 0}

    token = get_token()
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(ARTIST_ENDPOINT.format(artist_id), headers=headers)

    if response.status_code == 200:
        artist_data = response.json()
        return {
            "Genres": artist_data.get('genres', []),
            "Followers": artist_data.get('followers', {}).get('total', 0)
        }
    else:
        logger.error(f'Error {response.status_code} fetching details for artist {artist_id}: {response.text}')
        return {"Genres": [], "Followers": 0}

def search_isrc(isrc, category='track', limit=1, extractor=extract_track_info, include_details=False):
    """Search Spotify for media using ISRC and optionally retrieve artist details (genres & followers)."""
    token = get_token()
    headers = {'Authorization': f'Bearer {token}'}
    
    query = f'isrc:{isrc}'
    params = {'q': query, 'type': category, 'limit': limit}
    
    response = requests.get(SEARCH_ENDPOINT, headers=headers, params=params)
    
    if response.status_code == 200:
        track_info = extractor(response.json(), isrc)
        if track_info and include_details:
            artist_details = get_artist_details(track_info.get('Artist ID'))
            track_info.update(artist_details)
        return track_info
    else:
        logger.error(f'Error {response.status_code} for ISRC {isrc}: {response.text}')
        return None

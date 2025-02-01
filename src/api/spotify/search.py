import requests
from utils.logger import logger
from api.spotify.authenticator import get_token

SEARCH_ENDPOINT = 'https://api.spotify.com/v1/search'

def search_isrc(isrc, category='track', limit=1):
    """Search Spotify for media using ISRC (International Standard Recording Code)."""
    token = get_token()
    headers = {'Authorization': f'Bearer {token}'}
    
    query = f'isrc:{isrc}'
    params = {'q': query, 'type': category, 'limit': limit}
    
    response = requests.get(SEARCH_ENDPOINT, headers=headers, params=params)
    
    if response.status_code == 200:
        return extract_track_info(response.json(), isrc)
    else:
        logger.error(f'Error {response.status_code} for ISRC {isrc}: {response.text}')
        return None
    
def extract_track_info(results, isrc):
    """Exctract track name, album name, and artist name from response."""
    items = results.get('tracks', {}).get('items', [])
    if items:
        top_result = items[0]
        return {
            'Track': top_result['name'],
            'Album Name': top_result['album']['name'],
            'Artist': top_result['artists'][0]['name']
        }
    else:
        logger.warning(f'No results found for ISRC {isrc}')
        return None
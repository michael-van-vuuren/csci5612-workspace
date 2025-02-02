import requests
from utils.logger import logger

SEARCH_ENDPOINT = 'https://musicbrainz.org/ws/2/recording/'

def extract_recording_info(results, isrc):
    """Extracts and aggregates length, release count, and tags from the API response."""
    recordings = results.get('recordings', [])
    
    # not found -> Length = None
    # found, but no valid lengths -> Length = 0
    # found, valid lengths -> Length = length_avg
    if not recordings:
        logger.warning(f'No recordings found for ISRC {isrc}')
        return {'Length': None, 'Releases': 0, 'Genres': [], 'ISRC': isrc}
    
    total_length = 0
    total_releases = 0
    tags_set = set()
    valid_lengths = 0

    for recording in recordings:
        # extract length
        length = recording.get('length')
        if length:
            total_length += length
            valid_lengths += 1

        # count releases
        releases = recording.get('releases', [])
        total_releases += len(releases)

        # collect tags
        tags = recording.get('tags', [])
        tags_set.update(tag['name'] for tag in tags if 'name' in tag)

    # average length in seconds
    length_avg = (total_length / valid_lengths / 1000) if valid_lengths > 0 else None

    return {
        'Length': length_avg,
        'Releases': total_releases,
        'Genres': list(tags_set),
        'ISRC': isrc
    }

def search_isrc(isrc, extractor=extract_recording_info):
    """Search MusicBrainz for track information using ISRC."""
    headers = {'User-Agent': 'DataScienceProject/1.0 ( miva6981@colorado.edu )'}
    
    query = f'isrc:{isrc}'
    params = {'query': query, 'fmt': 'json'}

    response = requests.get(SEARCH_ENDPOINT, headers=headers, params=params)

    if response.status_code == 200:
        return extractor(response.json(), isrc)
    elif response.status_code == 503:
        raise RuntimeError(f'Rate limit reached at {isrc}')
    else:
        logger.error(f'Error {response.status_code} for ISRC {isrc}: {response.text}')
        return None
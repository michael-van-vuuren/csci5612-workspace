import os
import logging

"""
Set up logging. Writes to logs/spotify_api.log.
Additionally outputs logs to console.
"""

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
log_file = os.path.join(log_dir, 'api_spotify.log')

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger('SpotifyAPI')
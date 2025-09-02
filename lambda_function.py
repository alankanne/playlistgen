
# Lambda function for detecting sentiment and building playlist


import json
import os
import boto3
import requests
import re
import logging
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bedrock_client = boto3.client("bedrock-runtime")  # Bedrock runtime client

# ---- Robust parsing helpers ----
DASH_SPLIT = re.compile(r"\s*[-–—]\s*")  # hyphen, en dash, em dash
LEADING_NUM = re.compile(r"^\s*\d+[\)\.\:\-]\s*")  # "1) ", "1. ", "1- ", "1: "
FEATURE_PAT = re.compile(r"\s*[\(\[]\s*(feat\.?|ft\.?|with)\s+[^\)\]]+[\)\]]", re.I)
REMIX_PAT = re.compile(r"\s*[\(\[]\s*(remix|edit|version|live|radio|acoustic)[^\)\]]*[\)\]]", re.I)
QUOTE_CHARS = "\"'“”‘’"

def _clean_text(s: str) -> str:
    s = s.strip().strip(QUOTE_CHARS)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _normalize_title_artist(title: str, artist: str) -> Tuple[str, str]:
    title = _clean_text(title)
    artist = _clean_text(artist)
    # drop common noise in title
    title = FEATURE_PAT.sub("", title)
    title = REMIX_PAT.sub("", title)
    title = re.sub(r"\s+", " ", title).strip()
    artist = re.sub(r"\s+", " ", artist).strip()
    return title, artist

def _split_title_artist(line: str) -> Optional[Tuple[str, str]]:
    line = LEADING_NUM.sub("", line.strip())
    parts = DASH_SPLIT.split(line, maxsplit=1)
    if len(parts) != 2:
        return None
    title, artist = _normalize_title_artist(parts[0], parts[1])
    if not title or not artist:
        return None
    return title, artist

# ---- Lambda entrypoint ----
def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        user_text = body["text"]
        spotify_token = body["spotify_access_token"]

        # 1) sentiment
        sentiment = get_sentiment_from_llama(user_text)
        logger.info(f"Sentiment: {sentiment}")

        # 2) user's top artists
        top_artists = get_top_artists(spotify_token)
        logger.info(f"Top artists: {[a.get('name') for a in top_artists]}")

        # 3) suggestions from LLM
        suggestions = get_song_suggestions(top_artists, sentiment)
        logger.info(f"Suggestions ({len(suggestions)}): {suggestions}")

        # 4) resolve to Spotify URIs
        uris: List[str] = []
        for s in suggestions:
            uri = search_spotify_track(spotify_token, s)
            if uri:
                uris.append(uri)

        # de-dupe, preserve order
        seen = set()
        track_uris = []
        for u in uris:
            if u not in seen:
                seen.add(u)
                track_uris.append(u)

        logger.info(f"Resolved {len(track_uris)} / {len(suggestions)} suggestions to Spotify URIs")

        # 5) user id
        user_id = get_user_id(spotify_token)
        if not user_id:
            raise Exception("Failed to retrieve Spotify user ID")

        # 6) playlist title
        playlist_title = generate_playlist_title(suggestions, sentiment)

        # 7) create playlist
        playlist_id = create_spotify_playlist(spotify_token, user_id, playlist_title)
        if not playlist_id:
            raise Exception("Failed to create Spotify playlist")

        # 8) add tracks (chunked)
        if track_uris:
            add_tracks_to_playlist(spotify_token, playlist_id, track_uris)
        else:
            logger.warning("No tracks resolved; playlist will be empty.")

        playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"playlist_url": playlist_url, "added": len(track_uris)})
        }

    except KeyError as e:
        logger.error(f"Missing required field: {str(e)}")
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Missing required field: {str(e)}"})
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }

# ---- Bedrock sentiment ----
def get_sentiment_from_llama(text: str) -> str:
    prompt = (
        "You are a music mood detection assistant.\n"
        "Analyze the following text and return EXACTLY one single English word that best describes the music mood.\n"
        "Examples: excited, slow, tired, happy, sad, energetic, calm, etc.\n"
        "No punctuation or explanation. If you are unsure, guess a close positive or negative word.\n\n"
        f"User text: '{text}'\n"
        "Your one-word sentiment label:"
    )
    payload = {
        "prompt": prompt,
        "max_gen_len": 3,
        "temperature": 0.3,
        "top_p": 0.9
    }
    try:
        response = bedrock_client.invoke_model(
            modelId="meta.llama3-3-70b-instruct-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        result = json.loads(response["body"].read().decode("utf-8"))
        raw = (result.get("generation") or "").strip().lower()
        logger.info(f"LLM raw sentiment output: '{raw}'")
        if not raw or raw == "neutral":
            return "happy"
        # strip any quotes/periods
        raw = raw.strip(QUOTE_CHARS + ".")
        # just first token to be safe
        return raw.split()[0]
    except Exception as e:
        logger.error(f"Error detecting sentiment: {e}")
        return "happy"

# ---- Spotify helpers ----
def get_top_artists(token: str) -> List[dict]:
    url = "https://api.spotify.com/v1/me/top/artists?limit=10"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json().get("items", [])
    except requests.RequestException as e:
        logger.error(f"Error fetching top artists: {e}")
        return []

def get_song_suggestions(top_artists: List[dict], sentiment: str) -> List[str]:
    artist_names = [a.get("name", "") for a in top_artists if a.get("name")]
    prompt = (
        "You are a music recommendation engine with broad knowledge of mainstream songs from 1970 to 2023.\n"
        "Output EXACTLY 20 lines, each in the format:\n"
        "Song Title - Artist Name\n"
        "No extra commentary.\n\n"
        f"The user is feeling '{sentiment}'. Their favorite artists: {', '.join(artist_names)}.\n"
        f"Suggest 20 widely known songs for a/an '{sentiment}' vibe."
    )
    payload = {"prompt": prompt, "max_gen_len": 512, "temperature": 0.3, "top_p": 0.9}
    try:
        resp = bedrock_client.invoke_model(
            modelId="meta.llama3-3-70b-instruct-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        result = json.loads(resp["body"].read().decode("utf-8"))
        text = (result.get("generation") or "").strip()
        raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        suggestions: List[str] = []
        for ln in raw_lines:
            pair = _split_title_artist(ln)
            if not pair:
                continue
            title, artist = pair
            suggestions.append(f"{title} - {artist}")
            if len(suggestions) == 20:
                break

        if len(suggestions) < 20:
            logger.warning(f"Collected {len(suggestions)} valid suggestions (LLM returned {len(raw_lines)} lines).")

        return suggestions
    except Exception as e:
        logger.error(f"Error generating song suggestions: {e}")
        return []

def search_spotify_track(token: str, suggestion: str) -> Optional[str]:
    """Resolve 'Title - Artist' to a Spotify track URI with market fallbacks."""
    try:
        title, artist = suggestion.split(" - ", 1)
        title, artist = _normalize_title_artist(title, artist)

        headers = {"Authorization": f"Bearer {token}"}
        url = "https://api.spotify.com/v1/search"

        def do_search(q: str, market: Optional[str]) -> Optional[str]:
            params = {
                "q": q,
                "type": "track",
                "limit": 1
            }
            if market:
                params["market"] = market
            r = requests.get(url, headers=headers, params=params, timeout=8)
            # If the token can't use market=from_token, Spotify often returns 403.
            if r.status_code in (400, 403):
                try:
                    j = r.json()
                except Exception:
                    j = {"_raw": r.text[:200]}
                logger.warning(f"Search status {r.status_code} for q='{q}' market='{market}': {j}")
                return None
            r.raise_for_status()
            items = r.json().get("tracks", {}).get("items", [])
            return items[0]["uri"] if items else None

        # 1) strict phrase match with from_token
        q1 = f'track:"{title}" artist:"{artist}"'
        uri = do_search(q1, "from_token")
        if uri:
            return uri

        # 2) strict phrase match with NO market
        uri = do_search(q1, None)
        if uri:
            return uri

        # 3) strict phrase match with US market
        uri = do_search(q1, "US")
        if uri:
            return uri

        # 4) relax artist (drop features / split on delimiters)
        lead_artist = re.split(r"[,&/]| feat\.?| ft\.?| with ", artist, flags=re.I)[0].strip()
        if lead_artist and lead_artist != artist:
            q2 = f'track:"{title}" artist:"{lead_artist}"'
            for m in ("from_token", None, "US"):
                uri = do_search(q2, m)
                if uri:
                    return uri

        # 5) title only (last resort)
        q3 = f'track:"{title}"'
        for m in ("from_token", None, "US"):
            uri = do_search(q3, m)
            if uri:
                return uri

        logger.info(f"No Spotify match for: {title} — {artist}")
        return None

    except requests.RequestException as e:
        logger.error(f"Spotify search HTTP error for '{suggestion}': {e}")
        return None
    except Exception as e:
        logger.error(f"Error searching for track '{suggestion}': {e}")
        return None
    try:
        title, artist = suggestion.split(" - ", 1)
        title, artist = _normalize_title_artist(title, artist)

        headers = {"Authorization": f"Bearer {token}"}
        url = "https://api.spotify.com/v1/search"

        def do_search(q: str) -> Optional[str]:
            params = {
                "q": q,
                "type": "track",
                "market": "from_token",  # key: match the user’s region
                "limit": 1
            }
            r = requests.get(url, headers=headers, params=params, timeout=8)
            r.raise_for_status()
            items = r.json().get("tracks", {}).get("items", [])
            return items[0]["uri"] if items else None

        # Strict phrase match
        q1 = f'track:"{title}" artist:"{artist}"'
        uri = do_search(q1)
        if uri:
            return uri

        # Relax artist (drop features / split on delimiters)
        lead_artist = re.split(r"[,&/]| feat\.?| ft\.?| with ", artist, flags=re.I)[0].strip()
        if lead_artist and lead_artist != artist:
            q2 = f'track:"{title}" artist:"{lead_artist}"'
            uri = do_search(q2)
            if uri:
                return uri

        # Title only as last resort
        q3 = f'track:"{title}"'
        uri = do_search(q3)
        if uri:
            return uri

        logger.info(f"No Spotify match for: {title} — {artist}")
        return None

    except requests.RequestException as e:
        logger.error(f"Spotify search HTTP error for '{suggestion}': {e}")
        return None
    except Exception as e:
        logger.error(f"Error searching for track '{suggestion}': {e}")
        return None

def get_user_id(token: str) -> Optional[str]:
    url = "https://api.spotify.com/v1/me"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        user_data = r.json()
        logger.info(f"User data: {user_data.get('id')}")
        return user_data.get("id")
    except requests.RequestException as e:
        logger.error(f"Error fetching user ID: {e}")
        return None

def create_spotify_playlist(token: str, user_id: str, playlist_title: str) -> Optional[str]:
    if not user_id:
        logger.error("No valid user_id provided")
        return None
    url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "name": playlist_title,
        "public": False,
        "description": "Playlist generated using LLaMA-based suggestions"
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        playlist_data = r.json()
        logger.info(f"Playlist created: {playlist_data.get('id')}")
        return playlist_data.get("id")
    except requests.RequestException as e:
        logger.error(f"Error creating playlist: {e}")
        return None

def add_tracks_to_playlist(token: str, playlist_id: str, track_uris: List[str]) -> None:
    if not track_uris:
        logger.warning("No tracks to add to playlist")
        return
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Spotify limit: 100 URIs per request
    i = 0
    total = 0
    while i < len(track_uris):
        chunk = track_uris[i:i+100]
        payload = {"uris": chunk}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=15)
            r.raise_for_status()
            total += len(chunk)
            i += len(chunk)
        except requests.RequestException as e:
            logger.error(f"Error adding tracks chunk to playlist: {e}")
            # best effort: break rather than raising, so the playlist still exists
            break
    logger.info(f"Added {total} tracks to playlist {playlist_id}")

def generate_playlist_title(suggestions: List[str], sentiment: str) -> str:
    tokens = (sentiment or "").strip().split()
    mood = tokens[0].capitalize() if tokens else "Default"
    return f"{mood} Vibes"


import streamlit as st
import pandas as pd
import requests
import html
import base64
import mimetypes
import re

from pathlib import Path
from urllib.parse import quote
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="VENOM | Movie Intelligence",
    page_icon="🖤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PROJECT = Path(__file__).resolve().parent
BASE_URL = "https://v3-cinemeta.strem.io"
PAGE_SIZE = 100
DISPLAY_COUNT = 8
POSTER_BASE = "https://images.metahub.space/poster/medium"


# =====================================================
# 2. SESSION STATE
# =====================================================

DEFAULTS = {
    "v2_search_results": [],
    "v2_search_query": "",
    "v2_search_skip": 0,
    "v2_search_more": False,
    "v2_search_visible": DISPLAY_COUNT,

    "v2_genre_results": [],
    "v2_genre_name": "",
    "v2_genre_skip": 0,
    "v2_genre_more": False,
    "v2_genre_visible": DISPLAY_COUNT,
    "v2_genre_loaded": False,

    "v2_popular_results": [],
    "v2_popular_skip": 0,
    "v2_popular_more": False,
    "v2_popular_visible": DISPLAY_COUNT,

    "v2_collection": [],
    "v2_collection_loaded": False,

    "v2_recommendations": [],
    "v2_recommendation_title": ""
}

for key, default in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = (
            default.copy()
            if isinstance(default, list)
            else default
        )


# =====================================================
# 3. FIND VENOM WALLPAPER
# =====================================================

venom_image = None

for file in PROJECT.rglob("*"):

    if (
        file.is_file()
        and "venom" in file.stem.lower()
        and file.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".webp"
        ]
    ):
        venom_image = file
        break

wallpaper = ""

if venom_image:

    mime = mimetypes.guess_type(
        venom_image.name
    )[0] or "image/jpeg"

    encoded = base64.b64encode(
        venom_image.read_bytes()
    ).decode("utf-8")

    wallpaper = f"data:{mime};base64,{encoded}"


# =====================================================
# 4. LOAD YOUR LOCAL MOVIES
# =====================================================

CSV_PATH = PROJECT / "movies.csv"

if not CSV_PATH.exists():
    st.error("movies.csv is missing from your project folder.")
    st.stop()


@st.cache_data
def load_local_movies(path):

    data = pd.read_csv(path)

    required = ["title", "genre", "description"]

    if not all(col in data.columns for col in required):
        raise ValueError(
            "CSV must contain title, genre and description."
        )

    data = data.dropna(subset=required).copy()

    for col in required:
        data[col] = data[col].astype(str).str.strip()

    data = data[
        (data["title"] != "") &
        (data["genre"] != "") &
        (data["description"] != "")
    ]

    return data.reset_index(drop=True)


try:
    local_movies = load_local_movies(str(CSV_PATH))
except Exception as error:
    st.error(f"Movie data error: {error}")
    st.stop()


# =====================================================
# 5. GENRE HANDLING
# =====================================================

def split_genres(value):

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if not value:
        return []

    return [
        item.strip()
        for item in re.split(r"[,;/|]", str(value))
        if item.strip()
    ]


def clean_title(title):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(title).lower()
    )


# =====================================================
# 6. CINEMETA API
# =====================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_catalog(path):

    response = requests.get(
        BASE_URL + path,
        headers={
            "User-Agent": "VenomMovieRecommendationSystem/5.0"
        },
        timeout=(4, 10)
    )

    response.raise_for_status()

    data = response.json()

    return data.get("metas", [])


def search_url(query, skip=0):

    path = (
        "/catalog/movie/top/search="
        + quote(query.strip(), safe="")
    )

    if skip:
        path += f"/skip={skip}"

    return path + ".json"


def genre_url(genre, skip=0):

    path = (
        "/catalog/movie/top/genre="
        + quote(genre, safe="")
    )

    if skip:
        path += f"/skip={skip}"

    return path + ".json"


def popular_url(skip=0):

    path = "/catalog/movie/top"

    if skip:
        path += f"/skip={skip}"

    return path + ".json"


# =====================================================
# 7. FULL MOVIE DETAILS
# =====================================================

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_movie_details_api(imdb_id):

    if not re.fullmatch(r"tt\d+", str(imdb_id)):
        return {}

    url = (
        BASE_URL
        + "/meta/movie/"
        + quote(str(imdb_id), safe="")
        + ".json"
    )

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent": "VenomMovieRecommendationSystem/5.0"
            },
            timeout=(4, 10)
        )

        response.raise_for_status()

        return response.json().get("meta") or {}

    except (requests.RequestException, ValueError):

        return {}


# =====================================================
# 8. POSTER URL HANDLING
# =====================================================

def poster_fallback(movie_id):

    movie_id = str(movie_id or "").strip()

    if re.fullmatch(r"tt\d+", movie_id):

        return f"{POSTER_BASE}/{movie_id}/img"

    return ""


def normalize_poster(poster, movie_id):

    poster = str(poster or "").strip()

    if poster.startswith("http://"):
        poster = "https://" + poster[7:]

    if not poster:
        poster = poster_fallback(movie_id)

    return poster


# =====================================================
# 9. NORMALIZE ONLINE MOVIES
# =====================================================

def normalize_movie(raw):

    title = str(
        raw.get("name") or raw.get("title") or ""
    ).strip()

    if not title:
        return None

    movie_id = str(
        raw.get("id")
        or raw.get("imdb_id")
        or raw.get("imdbId")
        or title
    )

    genres = split_genres(
        raw.get("genres") or raw.get("genre") or []
    )

    poster = normalize_poster(
        raw.get("poster"),
        movie_id
    )

    return {
        "id": movie_id,
        "title": title,
        "genre_list": genres,
        "genre": ", ".join(genres),
        "description": str(raw.get("description") or ""),
        "year": str(
            raw.get("year") or raw.get("releaseInfo") or ""
        ),
        "poster": poster,
        "rating": str(raw.get("imdbRating") or ""),
        "runtime": str(raw.get("runtime") or ""),
        "director": raw.get("director", []),
        "cast": raw.get("cast", []),
        "writer": raw.get("writer", []),
        "source": "Cinemeta"
    }


# =====================================================
# 10. REMOVE DUPLICATE MOVIES
# =====================================================

def unique_movies(items):

    output = []
    seen = set()

    for movie in items:

        if not movie:
            continue

        movie_id = str(movie.get("id") or "")

        if re.fullmatch(r"tt\d+", movie_id):

            key = ("imdb", movie_id)

        else:

            key = (
                "title",
                clean_title(movie.get("title", "")),
                str(movie.get("year", ""))
            )

        if key not in seen:

            seen.add(key)
            output.append(movie.copy())

        else:

            for existing in output:

                existing_id = str(existing.get("id") or "")

                if re.fullmatch(r"tt\d+", existing_id):
                    existing_key = ("imdb", existing_id)
                else:
                    existing_key = (
                        "title",
                        clean_title(existing.get("title", "")),
                        str(existing.get("year", ""))
                    )

                if existing_key == key:

                    for field in [
                        "poster", "description",
                        "genre", "rating", "year"
                    ]:

                        if not existing.get(field) and movie.get(field):
                            existing[field] = movie[field]

                    break

    return output


# =====================================================
# 11. LOCAL MOVIE COLLECTION
# =====================================================

def get_local_movies():

    output = []

    for _, row in local_movies.iterrows():

        imdb_id = ""

        if "imdb_id" in local_movies.columns:
            imdb_id = str(row.get("imdb_id", "")).strip()

        poster = ""

        if "poster_url" in local_movies.columns:
            poster = str(row.get("poster_url", "")).strip()

        genres = split_genres(row["genre"])

        output.append({
            "id": imdb_id or "local:" + str(row["title"]),
            "title": str(row["title"]),
            "genre_list": genres,
            "genre": str(row["genre"]),
            "description": str(row["description"]),
            "year": "",
            "poster": normalize_poster(poster, imdb_id),
            "rating": "",
            "runtime": "",
            "director": [],
            "cast": [],
            "writer": [],
            "source": "Your collection"
        })

    return output


# =====================================================
# 12. MATCH LOCAL MOVIES TO ONLINE MOVIES
# =====================================================

@st.cache_data(ttl=86400, show_spinner=False)
def lookup_movie_by_title(title):

    try:

        raw = fetch_catalog(search_url(title, 0))

    except requests.RequestException:

        return {}

    candidates = [
        normalize_movie(item)
        for item in raw
    ]

    candidates = [
        item for item in candidates if item
    ]

    best = None
    best_score = 0

    target = clean_title(title)

    for movie in candidates:

        score = SequenceMatcher(
            None,
            target,
            clean_title(movie["title"])
        ).ratio()

        if score > best_score:
            best_score = score
            best = movie

    if best and best_score >= 0.85:
        return best

    return {}


def enrich_movie(movie):

    result = movie.copy()

    if result.get("poster"):
        return result

    match = lookup_movie_by_title(result["title"])

    if match:

        for field in [
            "id", "poster", "year", "rating",
            "description", "genre", "genre_list"
        ]:

            if not result.get(field):
                result[field] = match.get(field, "")

    return result


# =====================================================
# 13. ACTUAL POSTER IMAGE DOWNLOAD
# =====================================================

@st.cache_data(ttl=86400, show_spinner=False)
def download_poster(url):

    if not url:
        return None

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 VenomMovieApp/5.0"
            },
            timeout=(3, 5)
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", ""
        )

        if not content_type.lower().startswith("image/"):
            return None

        return response.content

    except requests.RequestException:

        return None


def get_poster_bytes(movie):

    urls = []

    direct = movie.get("poster", "")

    if direct:
        urls.append(direct)

    imdb_id = str(movie.get("id", ""))

    fallback = poster_fallback(imdb_id)

    if fallback and fallback not in urls:
        urls.append(fallback)

    for url in urls:

        image_data = download_poster(url)

        if image_data:
            return image_data

    return None


# =====================================================
# 14. FULL MOVIE INFORMATION
# =====================================================

def get_full_movie_details(movie):

    result = movie.copy()

    movie_id = str(result.get("id") or "")

    # Identify local movies through catalogue search.

    if not re.fullmatch(r"tt\d+", movie_id):

        matched = lookup_movie_by_title(
            result["title"]
        )

        if matched:

            movie_id = str(matched.get("id") or "")

            for field in [
                "poster", "year", "rating",
                "description", "genre", "genre_list"
            ]:

                if not result.get(field):
                    result[field] = matched.get(field, "")

    # Fetch complete metadata only when an IMDb ID is available.

    if re.fullmatch(r"tt\d+", movie_id):

        details = fetch_movie_details_api(movie_id)

        if details:

            full_movie = normalize_movie(details)

            if full_movie:

                for key, value in full_movie.items():

                    if value:
                        result[key] = value

            for field in [
                "runtime", "director", "cast",
                "writer", "country", "language",
                "awards", "released"
            ]:

                if details.get(field):
                    result[field] = details[field]

    result["id"] = movie_id or result.get("id", "")

    if not result.get("poster"):
        result["poster"] = poster_fallback(movie_id)

    return result


def people_text(value):

    if isinstance(value, list):
        return ", ".join(str(item) for item in value)

    if isinstance(value, str) and value:
        return value

    return "Not available"


# =====================================================
# 15. MOVIE DETAILS DIALOG
# =====================================================

@st.dialog("VENOM CINEMA | MOVIE DETAILS", width="large")
def show_movie_details(movie):

    with st.spinner("Loading available movie details..."):
        details = get_full_movie_details(movie)

    left, right = st.columns([0.9, 1.5], gap="large")

    with left:

        poster_bytes = get_poster_bytes(details)

        if poster_bytes:
            st.image(poster_bytes, use_container_width=True)
        else:
            st.info("Poster unavailable for this movie.")

    with right:

        title = html.escape(
            str(details.get("title", "Unknown"))
        )

        st.html(f"""
        <div class="detail-title">{title}</div>
        """)

        year = (
            details.get("year")
            or details.get("released")
            or "Not available"
        )

        if str(year).startswith(("19", "20")):
            year = str(year)[:4]

        st.markdown(f"**Release year:** {year}")
        st.markdown(
            f"**Genre:** {details.get('genre') or 'Not available'}"
        )
        st.markdown(
            f"**IMDb rating:** {details.get('rating') or 'Not available'}"
        )
        st.markdown(
            f"**Runtime:** {details.get('runtime') or 'Not available'}"
        )
        st.markdown(
            f"**Director:** {people_text(details.get('director'))}"
        )
        st.markdown(
            f"**Cast:** {people_text(details.get('cast'))}"
        )

        st.divider()

        st.markdown("### MOVIE SYNOPSIS")

        synopsis = details.get("description")

        if synopsis:
            st.write(synopsis)
        else:
            st.info("Full synopsis is not available in the catalogue.")

        if details.get("writer"):
            st.markdown(
                "**Writer:** " + people_text(details["writer"])
            )

        if details.get("country"):
            st.markdown(
                "**Country:** " + people_text(details["country"])
            )

        if details.get("awards"):
            st.markdown(
                "**Awards:** " + str(details["awards"])
            )

        st.caption("VENOM MOVIE INTELLIGENCE")


# =====================================================
# 16. ORIGINAL TF-IDF + COSINE SIMILARITY
# =====================================================

def recommend_similar_movies(selected_movie):

    candidates = [selected_movie]

    genres = selected_movie.get("genre_list", [])

    # Use one genre catalogue to keep requests manageable.

    if genres:

        try:

            raw = fetch_catalog(genre_url(genres[0], 0))

            candidates.extend([
                normalize_movie(item)
                for item in raw
                if normalize_movie(item)
            ])

        except requests.RequestException:
            pass

    candidates.extend(get_local_movies())

    candidates = unique_movies(candidates)

    selected_id = str(selected_movie.get("id") or "")

    if len(candidates) < 2:
        return []

    documents = [
        movie["genre"] + " " + movie["description"]
        for movie in candidates
    ]

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(documents)

        selected_index = 0

        scores = cosine_similarity(
            matrix[selected_index:selected_index + 1],
            matrix
        ).flatten()

    except ValueError:
        return []

    ranking = scores.argsort()[::-1]

    recommendations = []

    for index in ranking:

        candidate = candidates[index]

        if index == 0:
            continue

        if (
            selected_id
            and str(candidate.get("id")) == selected_id
        ):
            continue

        recommendations.append(candidate)

        if len(recommendations) == 5:
            break

    return recommendations


# =====================================================
# 17. MOVIE CARD DISPLAY
# =====================================================

def display_movie_cards(
    items,
    label="MOVIE",
    allow_recommend=True
):

    if not items:
        st.info("No movies found.")
        return

    # Download visible posters concurrently and cache the bytes.

    with ThreadPoolExecutor(
        max_workers=min(8, len(items))
    ) as executor:

        poster_images = list(
            executor.map(get_poster_bytes, items)
        )

    for start in range(0, len(items), 2):

        columns = st.columns(2, gap="large")

        for offset, (column, movie) in enumerate(
            zip(columns, items[start:start + 2])
        ):

            index = start + offset
            poster_image = poster_images[index]

            with column:

                poster_col, detail_col = st.columns(
                    [0.85, 1.6],
                    gap="small"
                )

                with poster_col:

                    if poster_image:

                        st.image(
                            poster_image,
                            use_container_width=True
                        )

                    else:

                        st.html(f"""
                        <div class="poster-placeholder">
                            <span>VENOM CINEMA</span>
                            <strong>{html.escape(movie["title"])}</strong>
                            <small>Poster unavailable</small>
                        </div>
                        """)

                with detail_col:

                    title = html.escape(str(movie.get("title", "")))
                    genre = html.escape(str(movie.get("genre", "")))
                    description = html.escape(
                        str(movie.get("description") or "Description unavailable.")
                    )

                    year = html.escape(str(movie.get("year") or ""))
                    rating = html.escape(str(movie.get("rating") or ""))

                    st.html(f"""
                    <div class="movie-card">

                        <div class="movie-number">
                            {html.escape(label)}
                        </div>

                        <div class="movie-title">
                            {title}
                        </div>

                        <div class="movie-number">
                            {year} {rating}
                        </div>

                        <div class="movie-genre">
                            {genre or "GENRE UNAVAILABLE"}
                        </div>

                        <div class="movie-description">
                            {description[:250]}
                        </div>

                    </div>
                    """)

                # No broken second search dropdown:
                # recommendations are triggered directly from the card.

                safe_id = re.sub(
                    r"[^a-zA-Z0-9]",
                    "_",
                    str(movie.get("id") or movie["title"])
                )

                detail_key = (
                    f"details_{label}_{index}_{safe_id}"
                )

                if st.button(
                    "VIEW FULL DETAILS ↗",
                    key=detail_key,
                    use_container_width=True
                ):

                    show_movie_details(movie)

                if allow_recommend:

                    recommend_key = (
                        f"recommend_{label}_{index}_{safe_id}"
                    )

                    if st.button(
                        "RECOMMEND SIMILAR",
                        key=recommend_key,
                        use_container_width=True
                    ):

                        with st.spinner("Finding similar movies..."):

                            st.session_state.v2_recommendations = (
                                recommend_similar_movies(movie)
                            )

                            st.session_state.v2_recommendation_title = (
                                movie["title"]
                            )

                        st.rerun()


# =====================================================
# 18. VENOM LIQUID GLASS DESIGN
# =====================================================

CSS = """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap'
);

.stApp {

    background-color:#030303 !important;

    background-image:
        linear-gradient(
            90deg,
            rgba(0,0,0,.78),
            rgba(0,0,0,.48),
            rgba(0,0,0,.76)
        ),
        linear-gradient(
            0deg,
            rgba(0,0,0,.82),
            rgba(0,0,0,.22)
        ),
        url("__WALLPAPER__");

    background-position:center;
    background-size:cover;
    background-repeat:no-repeat;
    background-attachment:fixed;

    color:white;
    font-family:'Inter',sans-serif;
}

header, footer {
    visibility:hidden;
}

.stAppViewContainer,
[data-testid="stMain"] {
    background:transparent !important;
}

.block-container {
    max-width:1200px;
    padding-top:30px;
    padding-bottom:60px;
}

/* HERO */

.hero-panel {

    padding:48px 42px;
    border-radius:27px;

    background:linear-gradient(
        120deg,
        rgba(0,0,0,.65),
        rgba(25,25,25,.32)
    );

    border:1px solid rgba(255,255,255,.22);

    backdrop-filter:blur(12px);

    box-shadow:
        0 15px 50px rgba(0,0,0,.4),
        inset 0 1px 0 rgba(255,255,255,.15);
}

.hero-label {
    color:#ccc;
    font-size:11px;
    letter-spacing:5px;
}

.hero-title {

    font-family:'Orbitron',sans-serif;
    font-size:clamp(43px,7vw,78px);
    font-weight:900;
    letter-spacing:6px;
    line-height:1.15;

    color:white;

    text-shadow:
        0 0 9px white,
        0 0 23px rgba(255,255,255,.8),
        0 0 45px rgba(255,255,255,.4);

    margin:18px 0;
}

.hero-description {
    max-width:540px;
    color:#c3c3c3;
    font-size:14px;
    line-height:1.9;
}

.hero-status {

    display:inline-block;
    margin-top:22px;
    padding:10px 16px;

    color:white;
    font-size:10px;
    letter-spacing:2px;

    border:1px solid rgba(255,255,255,.3);
    border-radius:30px;

    background:rgba(255,255,255,.07);
}

/* HEADINGS */

.section-heading {

    font-family:'Orbitron',sans-serif;
    font-size:17px;
    font-weight:700;
    letter-spacing:3px;

    color:white;

    text-shadow:0 0 12px rgba(255,255,255,.6);

    margin-bottom:9px;
}

.section-description {
    color:#aaa;
    font-size:12px;
    line-height:1.8;
    margin-bottom:22px;
}

/* GLASS PANEL */

.glass-panel {

    padding:24px;
    border-radius:22px;

    background:linear-gradient(
        135deg,
        rgba(255,255,255,.14),
        rgba(255,255,255,.045)
    );

    border:1px solid rgba(255,255,255,.22);

    backdrop-filter:blur(25px);

    box-shadow:
        0 12px 35px rgba(0,0,0,.35),
        inset 0 1px 0 rgba(255,255,255,.13);
}

/* MOVIE CARDS */

@keyframes venomReveal {

    from {
        opacity:0;
        transform:translateY(12px);
    }

    to {
        opacity:1;
        transform:translateY(0);
    }

}

.movie-card {

    min-height:190px;

    padding:17px;

    border-radius:18px;

    background:linear-gradient(
        145deg,
        rgba(255,255,255,.14),
        rgba(255,255,255,.045)
    );

    border:1px solid rgba(255,255,255,.19);

    backdrop-filter:blur(20px);

    animation:venomReveal .4s ease both;

    transition:all .3s ease;

}

.movie-card:hover {

    border-color:white;

    box-shadow:
        0 0 16px rgba(255,255,255,.2),
        0 0 30px rgba(255,255,255,.1);

}

.movie-number {

    font-family:'Orbitron',sans-serif;
    color:#bdbdbd;
    font-size:9px;
    letter-spacing:1.5px;

}

.movie-title {

    font-family:'Orbitron',sans-serif;
    font-size:14px;
    font-weight:700;
    color:white;

    margin:13px 0;

}

.movie-genre {

    display:inline-block;

    border:1px solid rgba(255,255,255,.35);
    border-radius:20px;

    padding:5px 10px;

    color:white;
    font-size:9px;

    margin:10px 0;

}

.movie-description {

    color:#bdbdbd;
    font-size:11px;
    line-height:1.65;

}

/* POSTER PLACEHOLDER */

.poster-placeholder {

    min-height:220px;

    background:linear-gradient(
        145deg,
        #454545,
        #101010
    );

    border:1px solid rgba(255,255,255,.25);
    border-radius:13px;

    color:white;

    display:flex;
    flex-direction:column;

    align-items:center;
    justify-content:center;

    text-align:center;
    gap:12px;
    padding:10px;

}

.poster-placeholder span {

    color:#aaa;
    font-family:'Orbitron',sans-serif;
    font-size:8px;
    letter-spacing:2px;

}

.poster-placeholder small {
    color:#888;
    font-size:9px;
}

/* DETAILS WINDOW */

.detail-title {

    font-family:'Orbitron',sans-serif;
    font-size:clamp(22px,3vw,32px);
    font-weight:800;
    color:white;

    text-shadow:0 0 12px rgba(255,255,255,.6);

    margin-bottom:18px;

}

/* INPUTS */

div[data-testid="stTextInput"] input,
div[data-baseweb="select"] > div {

    background:rgba(0,0,0,.42) !important;
    border:1px solid rgba(255,255,255,.34) !important;
    border-radius:12px !important;
    color:white !important;

}

div[data-baseweb="select"] * {
    color:white !important;
}

div[data-baseweb="popover"] {
    background:#171717 !important;
}

/* BUTTONS */

.stButton > button {

    width:100%;
    min-height:43px;

    background:linear-gradient(
        110deg,
        white,
        #c8c8c8
    );

    color:#050505;

    border:1px solid white;
    border-radius:13px;

    font-family:'Orbitron',sans-serif;
    font-size:9px;
    font-weight:800;
    letter-spacing:1px;

    transition:all .25s ease;

}

.stButton > button:hover {

    background:white;
    color:black;

    box-shadow:
        0 0 15px white,
        0 0 35px rgba(255,255,255,.35);

}

/* TABS */

button[data-baseweb="tab"] {

    color:#aaa !important;

    font-family:'Orbitron',sans-serif;
    font-size:10px;
    letter-spacing:1px;

}

button[data-baseweb="tab"][aria-selected="true"] {
    color:white !important;
}

/* DIVIDER */

.glow-line {

    height:1px;
    margin:32px 0;

    background:linear-gradient(
        90deg,
        transparent,
        white,
        transparent
    );

    opacity:.5;

}

/* FOOTER */

.footer {

    text-align:center;
    color:#999;

    font-size:10px;
    letter-spacing:3px;
    line-height:2.2;

    padding:25px;

}

/* MOBILE */

@media(max-width:700px) {

    .block-container {
        padding:16px 12px 40px;
    }

    .hero-panel {
        padding:27px 21px;
    }

    .hero-title {
        font-size:42px;
        letter-spacing:3px;
    }

    .movie-card {
        padding:12px;
    }

}

</style>
"""

st.html(CSS.replace("__WALLPAPER__", wallpaper))


# =====================================================
# 19. HERO SECTION
# =====================================================

st.html("""
<div class="hero-panel">

    <div class="hero-label">
        THE SYMBIOTE IS ONLINE
    </div>

    <div class="hero-title">
        WE ARE VENOM
    </div>

    <div class="hero-description">
        Enter a cinematic universe.
        Discover movies, explore genres
        and find your next favourite film.
    </div>

    <div class="hero-status">
        ● LIVE MOVIE INTELLIGENCE | WHITE GLOW EDITION
    </div>

</div>
""")

if not venom_image:
    st.warning("Venom wallpaper not found.")


# =====================================================
# 20. DISCOVERY TABS
# =====================================================

st.html("""
<div class="glow-line"></div>

<div class="section-heading">
    MOVIE DISCOVERY
</div>

<div class="section-description">
    Search, browse popular movies, discover genres
    or explore your local collection.
</div>
""")

movie_tab, genre_tab, popular_tab, collection_tab = st.tabs([
    "🎬 MOVIE SEARCH",
    "🎭 GENRE DISCOVERY",
    "🔥 POPULAR MOVIES",
    "📁 MY COLLECTION"
])


# =====================================================
# 21. MOVIE SEARCH
# =====================================================

with movie_tab:

    st.html("""
    <div class="glass-panel">

        <div class="section-heading">
            SEARCH THE MOVIE UNIVERSE
        </div>

        <div class="section-description">
            Enter a title and find matching movies.
        </div>

    </div>
    """)

    query = st.text_input(
        "MOVIE SEARCH",
        placeholder="Try Inception, Oppenheimer, Spider-Man...",
        key="v2_movie_search"
    )

    if st.button(
        "⚡ SEARCH MOVIES",
        key="v2_search_button"
    ):

        if not query.strip():

            st.warning("Enter a movie title first.")

        else:

            with st.spinner("Searching movies..."):

                try:

                    raw = fetch_catalog(
                        search_url(query, 0)
                    )

                    remote = [
                        normalize_movie(item)
                        for item in raw
                    ]

                    remote = [
                        item for item in remote if item
                    ]

                    local_matches = [
                        item
                        for item in get_local_movies()
                        if query.casefold() in item["title"].casefold()
                    ]

                    st.session_state.v2_search_results = unique_movies(
                        remote + local_matches
                    )

                    st.session_state.v2_search_query = query
                    st.session_state.v2_search_skip = PAGE_SIZE

                    st.session_state.v2_search_more = (
                        len(raw) >= PAGE_SIZE
                    )

                    st.session_state.v2_search_visible = DISPLAY_COUNT
                    st.session_state.v2_recommendations = []

                except requests.RequestException as error:

                    st.error(f"Search connection error: {error}")

    search_results = st.session_state.v2_search_results

    if search_results:

        st.markdown(
            f"**{len(search_results)} MOVIES LOADED**"
        )

        display_movie_cards(
            search_results[:st.session_state.v2_search_visible],
            "SEARCH RESULT"
        )

        if (
            st.session_state.v2_search_visible < len(search_results)
            or st.session_state.v2_search_more
        ):

            if st.button(
                "LOAD MORE SEARCH RESULTS",
                key="v2_load_search"
            ):

                if (
                    st.session_state.v2_search_visible
                    < len(search_results)
                ):

                    st.session_state.v2_search_visible = min(
                        st.session_state.v2_search_visible + DISPLAY_COUNT,
                        len(search_results)
                    )

                elif st.session_state.v2_search_more:

                    try:

                        raw = fetch_catalog(
                            search_url(
                                st.session_state.v2_search_query,
                                st.session_state.v2_search_skip
                            )
                        )

                        more = [
                            normalize_movie(item)
                            for item in raw
                        ]

                        more = [
                            item for item in more if item
                        ]

                        old_count = len(search_results)

                        st.session_state.v2_search_results = unique_movies(
                            search_results + more
                        )

                        st.session_state.v2_search_skip += PAGE_SIZE

                        added = (
                            len(st.session_state.v2_search_results)
                            - old_count
                        )

                        st.session_state.v2_search_more = (
                            len(raw) >= PAGE_SIZE and added > 0
                        )

                        st.session_state.v2_search_visible = min(
                            len(st.session_state.v2_search_results),
                            st.session_state.v2_search_visible + DISPLAY_COUNT
                        )

                    except requests.RequestException as error:

                        st.error(f"Unable to load more: {error}")

    else:

        st.info("Search for a movie to see results.")


# =====================================================
# 22. GENRE DISCOVERY
# =====================================================

with genre_tab:

    st.html("""
    <div class="glass-panel">

        <div class="section-heading">
            EXPLORE BY GENRE
        </div>

        <div class="section-description">
            Discover movies from your favourite categories.
        </div>

    </div>
    """)

    genres = [
        "Action", "Adventure", "Animation", "Biography",
        "Comedy", "Crime", "Documentary", "Drama",
        "Family", "Fantasy", "History", "Horror",
        "Mystery", "Romance", "Sci-Fi", "Sport",
        "Thriller", "War", "Western"
    ]

    selected_genre = st.selectbox(
        "SELECT GENRE",
        genres,
        key="v2_genre_choice"
    )

    if selected_genre != st.session_state.v2_genre_name:

        st.session_state.v2_genre_name = selected_genre
        st.session_state.v2_genre_results = []
        st.session_state.v2_genre_skip = 0
        st.session_state.v2_genre_more = False
        st.session_state.v2_genre_visible = DISPLAY_COUNT
        st.session_state.v2_genre_loaded = False

    if st.button(
        "⚡ EXPLORE GENRE",
        key="v2_explore_genre"
    ):

        with st.spinner("Loading genre catalogue..."):

            try:

                raw = fetch_catalog(
                    genre_url(selected_genre, 0)
                )

                remote = [
                    normalize_movie(item)
                    for item in raw
                ]

                remote = [
                    item for item in remote if item
                ]

                local_matches = [
                    item for item in get_local_movies()
                    if selected_genre.casefold() in [
                        g.casefold()
                        for g in item["genre_list"]
                    ]
                ]

                st.session_state.v2_genre_results = unique_movies(
                    remote + local_matches
                )

                st.session_state.v2_genre_skip = PAGE_SIZE
                st.session_state.v2_genre_more = (
                    len(raw) >= PAGE_SIZE
                )

                st.session_state.v2_genre_visible = DISPLAY_COUNT
                st.session_state.v2_genre_loaded = True

            except requests.RequestException as error:

                st.error(f"Genre loading error: {error}")

    if st.session_state.v2_genre_loaded:

        genre_results = st.session_state.v2_genre_results

        st.markdown(
            f"**{selected_genre.upper()} MOVIES: "
            f"{len(genre_results)} LOADED**"
        )

        display_movie_cards(
            genre_results[:st.session_state.v2_genre_visible],
            "GENRE DISCOVERY"
        )

        if (
            st.session_state.v2_genre_visible < len(genre_results)
            or st.session_state.v2_genre_more
        ):

            if st.button(
                "LOAD MORE GENRE MOVIES",
                key="v2_load_genre"
            ):

                if (
                    st.session_state.v2_genre_visible
                    < len(genre_results)
                ):

                    st.session_state.v2_genre_visible = min(
                        st.session_state.v2_genre_visible + DISPLAY_COUNT,
                        len(genre_results)
                    )

                elif st.session_state.v2_genre_more:

                    try:

                        raw = fetch_catalog(
                            genre_url(
                                selected_genre,
                                st.session_state.v2_genre_skip
                            )
                        )

                        more = [
                            normalize_movie(item)
                            for item in raw
                        ]

                        more = [
                            item for item in more if item
                        ]

                        old_count = len(genre_results)

                        st.session_state.v2_genre_results = unique_movies(
                            genre_results + more
                        )

                        added = (
                            len(st.session_state.v2_genre_results)
                            - old_count
                        )

                        st.session_state.v2_genre_skip += PAGE_SIZE

                        st.session_state.v2_genre_more = (
                            len(raw) >= PAGE_SIZE and added > 0
                        )

                        st.session_state.v2_genre_visible = min(
                            len(st.session_state.v2_genre_results),
                            st.session_state.v2_genre_visible + DISPLAY_COUNT
                        )

                    except requests.RequestException as error:

                        st.error(f"Unable to load more genre movies: {error}")


# =====================================================
# 23. POPULAR MOVIE BROWSING
# =====================================================

with popular_tab:

    st.html("""
    <div class="glass-panel">

        <div class="section-heading">
            POPULAR MOVIES
        </div>

        <div class="section-description">
            Browse the general movie catalogue without searching
            for a particular title or genre.
        </div>

    </div>
    """)

    if not st.session_state.v2_popular_results:

        if st.button(
            "⚡ LOAD POPULAR MOVIES",
            key="v2_popular_button"
        ):

            with st.spinner("Loading popular movies..."):

                try:

                    raw = fetch_catalog(
                        popular_url(0)
                    )

                    results = [
                        normalize_movie(item)
                        for item in raw
                    ]

                    results = [
                        item for item in results if item
                    ]

                    st.session_state.v2_popular_results = unique_movies(
                        results
                    )

                    st.session_state.v2_popular_skip = PAGE_SIZE
                    st.session_state.v2_popular_more = (
                        len(raw) >= PAGE_SIZE
                    )

                    st.session_state.v2_popular_visible = DISPLAY_COUNT

                except requests.RequestException as error:

                    st.error(f"Unable to load popular movies: {error}")

    popular_results = st.session_state.v2_popular_results

    if popular_results:

        st.markdown(
            f"**{len(popular_results)} POPULAR MOVIES LOADED**"
        )

        display_movie_cards(
            popular_results[:st.session_state.v2_popular_visible],
            "POPULAR MOVIE"
        )

        if (
            st.session_state.v2_popular_visible < len(popular_results)
            or st.session_state.v2_popular_more
        ):

            if st.button(
                "LOAD MORE POPULAR MOVIES",
                key="v2_load_popular"
            ):

                if (
                    st.session_state.v2_popular_visible
                    < len(popular_results)
                ):

                    st.session_state.v2_popular_visible = min(
                        st.session_state.v2_popular_visible + DISPLAY_COUNT,
                        len(popular_results)
                    )

                elif st.session_state.v2_popular_more:

                    try:

                        raw = fetch_catalog(
                            popular_url(
                                st.session_state.v2_popular_skip
                            )
                        )

                        more = [
                            normalize_movie(item)
                            for item in raw
                        ]

                        more = [
                            item for item in more if item
                        ]

                        old_count = len(popular_results)

                        st.session_state.v2_popular_results = unique_movies(
                            popular_results + more
                        )

                        added = (
                            len(st.session_state.v2_popular_results)
                            - old_count
                        )

                        st.session_state.v2_popular_skip += PAGE_SIZE

                        st.session_state.v2_popular_more = (
                            len(raw) >= PAGE_SIZE and added > 0
                        )

                        st.session_state.v2_popular_visible = min(
                            len(st.session_state.v2_popular_results),
                            st.session_state.v2_popular_visible + DISPLAY_COUNT
                        )

                    except requests.RequestException as error:

                        st.error(f"Unable to load more popular movies: {error}")


# =====================================================
# 24. MY COLLECTION
# =====================================================

with collection_tab:

    st.html("""
    <div class="glass-panel">

        <div class="section-heading">
            YOUR PERSONAL MOVIE COLLECTION
        </div>

        <div class="section-description">
            Your original movies.csv collection with automatic
            poster and metadata matching.
        </div>

    </div>
    """)

    st.markdown(
        f"**TOTAL MOVIES: {len(local_movies)}**"
    )

    if st.button(
        "⚡ LOAD MY COLLECTION",
        key="v2_collection_button"
    ):

        with st.spinner("Finding your movie artwork..."):

            items = get_local_movies()

            with ThreadPoolExecutor(max_workers=8) as executor:

                enriched = list(
                    executor.map(enrich_movie, items)
                )

            st.session_state.v2_collection = enriched
            st.session_state.v2_collection_loaded = True

    if st.session_state.v2_collection_loaded:

        display_movie_cards(
            st.session_state.v2_collection,
            "MY COLLECTION"
        )


# =====================================================
# 25. RECOMMENDATION RESULTS
# =====================================================

if st.session_state.v2_recommendations:

    st.html("""
    <div class="glow-line"></div>

    <div class="section-heading">
        YOUR SYMBIOTE RECOMMENDATIONS
    </div>
    """)

    st.caption(
        "BASED ON: "
        + st.session_state.v2_recommendation_title
    )

    display_movie_cards(
        st.session_state.v2_recommendations,
        "RECOMMENDATION",
        allow_recommend=False
    )

    if st.button(
        "CLEAR RECOMMENDATIONS",
        key="v2_clear_recommendations"
    ):

        st.session_state.v2_recommendations = []
        st.session_state.v2_recommendation_title = ""

        st.rerun()


# =====================================================
# 26. FOOTER
# =====================================================

st.html("""
<div class="glow-line"></div>

<div class="footer">

    VENOM MOVIE INTELLIGENCE

    <br>

    CINEMETA LIVE MOVIE CATALOGUE

    <br>

    TF-IDF · COSINE SIMILARITY

    <br><br>

    WE ARE VENOM

</div>
""")

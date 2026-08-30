
"""
Context-aware sports image fetching for LiveMatch Intelligence.

Player-image strategy:
1. Existing context-specific cached image.
2. Wikimedia Commons search using:
       player name + represented team + football
   This is useful for national-team/historical match context.
3. TheSportsDB player artwork fallback.
4. Return None safely if no usable image is found.

Team badge strategy:
1. Existing validated cached badge.
2. Wikimedia Commons crest/badge/logo search with team-name ranking.
3. TheSportsDB team badge/logo fallback.
4. Preserve transparency in PNG cache where possible.
5. Return None safely.

The module never raises into the Streamlit app for normal lookup failures.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import quote

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "image_cache"
)

PLAYER_CACHE_DIR = (
    CACHE_ROOT
    / "players"
)

TEAM_CACHE_DIR = (
    CACHE_ROOT
    / "teams"
)

METADATA_CACHE_DIR = (
    CACHE_ROOT
    / "metadata"
)

THESPORTSDB_API_KEY = os.getenv(
    "THESPORTSDB_API_KEY",
    "3",
)

THESPORTSDB_BASE = (
    "https://www.thesportsdb.com/api/v1/json"
)

COMMONS_API = (
    "https://commons.wikimedia.org/w/api.php"
)

REQUEST_TIMEOUT = 15

HEADERS = {
    "User-Agent": (
        "LiveMatchIntelligence/1.0 "
        "(football analytics educational project)"
    )
}

NATIONAL_TEAM_HINTS = {
    "argentina",
    "france",
    "germany",
    "england",
    "spain",
    "portugal",
    "brazil",
    "croatia",
    "netherlands",
    "morocco",
    "japan",
    "mexico",
    "belgium",
    "switzerland",
    "uruguay",
    "poland",
    "senegal",
    "ghana",
    "cameroon",
    "canada",
    "australia",
    "denmark",
    "serbia",
    "wales",
    "ecuador",
    "qatar",
    "iran",
    "tunisia",
    "costa rica",
    "saudi arabia",
    "south korea",
    "united states",
    "usa",
}


def _safe_stem(value: str) -> str:
    value = str(value).strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    ).strip("_")

    if value:
        return value

    return hashlib.sha1(
        str(value).encode("utf-8")
    ).hexdigest()[:12]



def _normalise_search_text(value: str) -> str:
    """
    Lowercase and remove accents/diacritics for robust name matching.

    Examples:
        Mbappé -> mbappe
        Fernández -> fernandez
    """
    value = str(value or "")

    normalised = unicodedata.normalize(
        "NFKD",
        value,
    )

    ascii_text = "".join(
        char
        for char in normalised
        if not unicodedata.combining(
            char
        )
    )

    return ascii_text.casefold()




def _image_candidate_penalty_terms() -> tuple[str, ...]:
    return (
        " team ",
        " squad ",
        " group ",
        " lineup ",
        " line-up ",
        " starting eleven ",
        " starting xi ",
        "team photo",
        "squad photo",
        "group photo",
        "trophy",
        "award",
        "celebration",
        "celebrates",
        "celebrating",
        "training session",
        "press conference",
        "poster",
        "signature",
        "autograph",
        "illustration",
        "drawing",
        "logo",
        "crest",
        "badge",
        "icon",
        "svg",
    )


def _image_candidate_portrait_bonus(
    width,
    height,
) -> int:
    """
    Prefer portrait or near-square images suitable for player cards.
    """
    try:
        width = float(width)
        height = float(height)

        if width <= 0 or height <= 0:
            return 0

        aspect = width / height

        if 0.55 <= aspect <= 0.95:
            return 18

        if 0.95 < aspect <= 1.15:
            return 12

        if 1.15 < aspect <= 1.35:
            return 5

        if aspect >= 1.70:
            return -12

    except Exception:
        pass

    return 0


def _score_player_candidate(
    *,
    alias_tokens: list[str],
    title_text: str,
    metadata_text: str,
    represented_team: str = "",
    competition: str = "",
    width=None,
    height=None,
    alias_index: int = 0,
    query_index: int = 0,
) -> tuple[int, int]:
    """
    Shared global scoring for player-image candidates.

    Returns:
        (score, matched_alias_tokens)
    """
    title_norm = _normalise_search_text(
        title_text
    )

    meta_norm = _normalise_search_text(
        metadata_text
    )

    combined = (
        f" {title_norm} {meta_norm} "
    )

    title_hits = sum(
        1
        for token in alias_tokens
        if token in title_norm
    )

    if title_hits == 0:
        return (
            -999,
            0,
        )

    score = (
        title_hits
        * 16
    )

    # Strong bonus when at least two alias tokens identify the player
    # directly in the file title.
    if (
        len(alias_tokens) >= 2
        and title_hits >= 2
    ):
        score += 18

    # Prefer more natural aliases and more specific searches.
    score += max(
        0,
        10 - alias_index,
    )

    score += max(
        0,
        5 - query_index,
    )

    represented_team_norm = _normalise_search_text(
        represented_team
    )

    competition_norm = _normalise_search_text(
        competition
    )

    team_tokens = [
        token
        for token in re.split(
            r"\W+",
            represented_team_norm,
        )
        if len(token) >= 3
    ]

    competition_tokens = [
        token
        for token in re.split(
            r"\W+",
            competition_norm,
        )
        if len(token) >= 3
    ]

    team_hits = sum(
        1
        for token in team_tokens
        if token in combined
    )

    competition_hits = sum(
        1
        for token in competition_tokens
        if token in combined
    )

    score += (
        team_hits
        * 7
    )

    score += (
        competition_hits
        * 3
    )

    # National-team and tournament context are useful but not compulsory.
    if any(
        phrase in combined
        for phrase in (
            "national team",
            "international",
            "world cup",
            "fifa",
        )
    ):
        score += 5

    # Portrait suitability.
    score += _image_candidate_portrait_bonus(
        width,
        height,
    )

    # Prefer likely individual photos.
    if any(
        phrase in combined
        for phrase in (
            "portrait",
            "footballer",
            "player",
            "headshot",
            "profile",
        )
    ):
        score += 8

    # Penalise group / trophy / non-player-card style imagery.
    for bad in _image_candidate_penalty_terms():
        if bad.strip() in combined:
            score -= 14

    # Stronger penalty for clear group/team photographs.
    if any(
        phrase in combined
        for phrase in (
            "team photo",
            "squad photo",
            "group photo",
            "starting eleven",
            "starting xi",
            "team lineup",
            "team line-up",
        )
    ):
        score -= 30

    return (
        score,
        title_hits,
    )


def _football_name_aliases(player_name: str) -> list[str]:
    """
    Generate sensible football-name variants automatically.

    Examples:
        Lionel Andrés Messi Cuccittini
            -> Lionel Andrés Messi Cuccittini
            -> Lionel Messi
            -> Lionel Cuccittini

        Kylian Mbappé Lottin
            -> Kylian Mbappé Lottin
            -> Kylian Mbappé
            -> Kylian Lottin

        Enzo Fernández
            -> Enzo Fernández

    The goal is not to guess one 'correct' short name, but to search several
    plausible football aliases and let Wikimedia ranking choose the best match.
    """
    raw = str(
        player_name
    ).strip()

    if not raw:
        return []

    parts = [
        part
        for part in raw.replace(
            "-",
            " ",
        ).split()
        if part
    ]

    aliases = []

    def add(value: str):
        value = " ".join(
            str(value).split()
        ).strip()

        if (
            value
            and value not in aliases
        ):
            aliases.append(
                value
            )

    add(
        raw
    )

    # Standard two-token football name.
    if len(parts) == 2:
        add(
            f"{parts[0]} {parts[1]}"
        )

    # For longer legal names, pair the first name with each likely
    # football surname/name token after it.
    if len(parts) >= 3:
        for token in parts[1:]:
            add(
                f"{parts[0]} {token}"
            )

        # First + penultimate token is especially useful for:
        # Lionel Andrés Messi Cuccittini -> Lionel Messi
        add(
            f"{parts[0]} {parts[-2]}"
        )

        # First + last token remains a useful secondary fallback.
        add(
            f"{parts[0]} {parts[-1]}"
        )

    # Also include a compact first-two-token alias for names where
    # the common football name appears early.
    if len(parts) >= 2:
        add(
            " ".join(
                parts[:2]
            )
        )

    return aliases


def _ensure_cache_dirs() -> None:
    for folder in (
        PLAYER_CACHE_DIR,
        TEAM_CACHE_DIR,
        METADATA_CACHE_DIR,
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )


def _valid_cached_image(
    path: Path,
) -> bool:
    if not path.exists():
        return False

    try:
        with Image.open(path) as image:
            image.verify()

        return True

    except Exception:
        try:
            path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        return False


def _request_json(
    url: str,
    params: Optional[dict] = None,
) -> Optional[dict]:
    try:
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        payload = response.json()

        if isinstance(
            payload,
            dict,
        ):
            return payload

    except Exception:
        return None

    return None


def _download_image(
    image_url: str,
    destination: Path,
) -> Optional[Path]:
    """
    Download, validate and save an image safely.

    Important for Windows:
    the image is decoded from memory rather than opening the destination
    file and then trying to overwrite that same open file.
    """
    try:
        response = requests.get(
            image_url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Decode from memory first. This avoids Windows file-locking
        # problems caused by reopening and overwriting the same file.
        with Image.open(
            BytesIO(
                response.content
            )
        ) as image:
            image.load()

            processed = image.convert(
                "RGB"
            )

            processed.thumbnail(
                (1200, 1200)
            )

            processed.save(
                destination,
                format="JPEG",
                quality=92,
            )

        if _valid_cached_image(
            destination
        ):
            return destination

    except Exception as exc:
        # Keep the app safe, but make manual diagnostics possible.
        try:
            destination.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        return None

    return None


def _is_national_team(
    represented_team: Optional[str],
) -> bool:
    if not represented_team:
        return False

    return (
        str(
            represented_team
        )
        .strip()
        .lower()
        in NATIONAL_TEAM_HINTS
    )


def _player_cache_path(
    player_name: str,
    represented_team: Optional[str],
) -> Path:
    player_stem = _safe_stem(
        player_name
    )

    team_stem = _safe_stem(
        represented_team
        or "generic"
    )

    return (
        PLAYER_CACHE_DIR
        / f"{player_stem}__{team_stem}.jpg"
    )


def _cached_player_image_is_context_valid(
    player_name: str,
    represented_team: Optional[str],
    destination: Path,
) -> bool:
    """
    For national-team contexts, reject a cached image if its metadata
    says it came from a mismatched current club.
    """
    if not _valid_cached_image(destination):
        return False

    if not _is_national_team(represented_team):
        return True

    metadata_file = _metadata_path(
        player_name,
        represented_team,
    )

    if not metadata_file.exists():
        # Older cached files without metadata are not trusted for
        # national-team context.
        return False

    try:
        metadata = json.loads(
            metadata_file.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    if metadata.get("context_mismatch") is True:
        return False

    provider = str(
        metadata.get(
            "provider",
            "",
        )
    )

    # Context-aware Wikimedia result is acceptable only if it
    # was selected by the stricter individual-player filter.
    if provider == "Wikimedia Commons":
        return int(
            metadata.get(
                "selection_version",
                0,
            )
            or 0
        ) >= 6

    # TheSportsDB is acceptable only when its current team matches
    # the represented team.
    current_team = str(
        metadata.get(
            "current_team",
            "",
        )
        or ""
    ).strip()

    if (
        provider == "TheSportsDB"
        and current_team
        and current_team.casefold()
        == str(
            represented_team
        ).strip().casefold()
    ):
        return True

    return False


def _metadata_path(
    player_name: str,
    represented_team: Optional[str],
) -> Path:
    player_stem = _safe_stem(
        player_name
    )

    team_stem = _safe_stem(
        represented_team
        or "generic"
    )

    return (
        METADATA_CACHE_DIR
        / f"{player_stem}__{team_stem}.json"
    )


def _save_metadata(
    player_name: str,
    represented_team: Optional[str],
    metadata: dict,
) -> None:
    _ensure_cache_dirs()

    try:
        _metadata_path(
            player_name,
            represented_team,
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def _commons_player_image(
    player_name: str,
    represented_team: Optional[str],
    competition: Optional[str] = None,
) -> Optional[Tuple[str, dict]]:
    """
    Find the strongest individual Wikimedia image for the selected match context.
    """
    aliases = _football_name_aliases(
        player_name
    )

    if not aliases:
        return None

    represented_team = str(
        represented_team
        or ""
    ).strip()

    competition = str(
        competition
        or ""
    ).strip()

    ranked = []
    seen_urls = set()

    for alias_index, alias in enumerate(
        aliases
    ):
        alias_tokens = [
            token
            for token in re.split(
                r"\W+",
                _normalise_search_text(
                    alias
                ),
            )
            if len(token) >= 3
        ]

        min_alias_hits = (
            2
            if len(alias_tokens) >= 2
            else 1
        )

        queries = [
            f'"{alias}" "{represented_team}" football',
            f'"{alias}" "{represented_team}"',
            f'"{alias}" footballer',
        ]

        if competition:
            queries.insert(
                0,
                f'"{alias}" "{represented_team}" "{competition}"',
            )

        for query_index, search_query in enumerate(
            queries
        ):
            payload = _request_json(
                COMMONS_API,
                params={
                    "action": "query",
                    "format": "json",
                    "generator": "search",
                    "gsrnamespace": 6,
                    "gsrsearch": search_query,
                    "gsrlimit": 35,
                    "prop": "imageinfo",
                    "iiprop": "url|size|extmetadata",
                    "iiurlwidth": 900,
                    "origin": "*",
                },
            )

            if not payload:
                continue

            pages = (
                payload.get(
                    "query",
                    {}
                ).get(
                    "pages",
                    {}
                )
            )

            for page in pages.values():
                title = str(
                    page.get(
                        "title",
                        "",
                    )
                ).strip()

                infos = (
                    page.get(
                        "imageinfo"
                    )
                    or []
                )

                if not infos:
                    continue

                info = infos[0]

                image_url = (
                    info.get(
                        "thumburl"
                    )
                    or info.get(
                        "url"
                    )
                )

                if (
                    not isinstance(
                        image_url,
                        str,
                    )
                    or image_url in seen_urls
                ):
                    continue

                seen_urls.add(
                    image_url
                )

                ext = info.get(
                    "extmetadata",
                    {}
                )

                metadata_text = " ".join(
                    str(
                        ext.get(
                            key,
                            {},
                        ).get(
                            "value",
                            "",
                        )
                    )
                    for key in (
                        "ImageDescription",
                        "ObjectName",
                        "Categories",
                        "Credit",
                    )
                )

                score, alias_hits = (
                    _score_player_candidate(
                        alias_tokens=alias_tokens,
                        title_text=title,
                        metadata_text=metadata_text,
                        represented_team=represented_team,
                        competition=competition,
                        width=info.get(
                            "width"
                        ),
                        height=info.get(
                            "height"
                        ),
                        alias_index=alias_index,
                        query_index=query_index,
                    )
                )

                if alias_hits < min_alias_hits:
                    continue

                if score < 20:
                    continue

                metadata = {
                    "provider": "Wikimedia Commons",
                    "selection_version": 6,
                    "image_kind": "individual_player",
                    "matched_alias": alias,
                    "search_query": search_query,
                    "file_title": title,
                    "description_url": info.get(
                        "descriptionurl"
                    ),
                    "artist": (
                        ext.get(
                            "Artist",
                            {},
                        ).get(
                            "value"
                        )
                    ),
                    "license": (
                        ext.get(
                            "LicenseShortName",
                            {},
                        ).get(
                            "value"
                        )
                    ),
                    "license_url": (
                        ext.get(
                            "LicenseUrl",
                            {},
                        ).get(
                            "value"
                        )
                    ),
                    "represented_team": represented_team,
                    "competition": competition,
                    "ranking_score": score,
                    "matched_name_tokens": alias_hits,
                }

                ranked.append(
                    (
                        score,
                        image_url,
                        metadata,
                    )
                )

    if not ranked:
        return None

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return (
        ranked[0][1],
        ranked[0][2],
    )


def _commons_generic_individual_portrait(
    player_name: str,
) -> Optional[Tuple[str, dict]]:
    """
    Find the strongest generic individual Wikimedia player image.
    """
    aliases = _football_name_aliases(
        player_name
    )

    if not aliases:
        return None

    ranked = []
    seen_urls = set()

    for alias_index, alias in enumerate(
        aliases
    ):
        alias_tokens = [
            token
            for token in re.split(
                r"\W+",
                _normalise_search_text(
                    alias
                ),
            )
            if len(token) >= 3
        ]

        min_alias_hits = (
            2
            if len(alias_tokens) >= 2
            else 1
        )

        queries = [
            f'"{alias}" footballer',
            f'"{alias}" portrait football',
            f'"{alias}" football',
        ]

        for query_index, search_query in enumerate(
            queries
        ):
            payload = _request_json(
                COMMONS_API,
                params={
                    "action": "query",
                    "format": "json",
                    "generator": "search",
                    "gsrnamespace": 6,
                    "gsrsearch": search_query,
                    "gsrlimit": 35,
                    "prop": "imageinfo",
                    "iiprop": "url|size|extmetadata",
                    "iiurlwidth": 900,
                    "origin": "*",
                },
            )

            if not payload:
                continue

            pages = (
                payload.get(
                    "query",
                    {}
                ).get(
                    "pages",
                    {}
                )
            )

            for page in pages.values():
                title = str(
                    page.get(
                        "title",
                        "",
                    )
                ).strip()

                infos = (
                    page.get(
                        "imageinfo"
                    )
                    or []
                )

                if not infos:
                    continue

                info = infos[0]

                image_url = (
                    info.get(
                        "thumburl"
                    )
                    or info.get(
                        "url"
                    )
                )

                if (
                    not isinstance(
                        image_url,
                        str,
                    )
                    or image_url in seen_urls
                ):
                    continue

                seen_urls.add(
                    image_url
                )

                ext = info.get(
                    "extmetadata",
                    {}
                )

                metadata_text = " ".join(
                    str(
                        ext.get(
                            key,
                            {},
                        ).get(
                            "value",
                            "",
                        )
                    )
                    for key in (
                        "ImageDescription",
                        "ObjectName",
                        "Categories",
                        "Credit",
                    )
                )

                score, alias_hits = (
                    _score_player_candidate(
                        alias_tokens=alias_tokens,
                        title_text=title,
                        metadata_text=metadata_text,
                        represented_team="",
                        competition="",
                        width=info.get(
                            "width"
                        ),
                        height=info.get(
                            "height"
                        ),
                        alias_index=alias_index,
                        query_index=query_index,
                    )
                )

                if alias_hits < min_alias_hits:
                    continue

                if score < 24:
                    continue

                metadata = {
                    "provider": "Wikimedia Commons",
                    "selection_version": 6,
                    "image_kind": "generic_individual_headshot",
                    "matched_alias": alias,
                    "search_query": search_query,
                    "file_title": title,
                    "description_url": info.get(
                        "descriptionurl"
                    ),
                    "artist": (
                        ext.get(
                            "Artist",
                            {},
                        ).get(
                            "value"
                        )
                    ),
                    "license": (
                        ext.get(
                            "LicenseShortName",
                            {},
                        ).get(
                            "value"
                        )
                    ),
                    "license_url": (
                        ext.get(
                            "LicenseUrl",
                            {},
                        ).get(
                            "value"
                        )
                    ),
                    "ranking_score": score,
                    "matched_name_tokens": alias_hits,
                }

                ranked.append(
                    (
                        score,
                        image_url,
                        metadata,
                    )
                )

    if not ranked:
        return None

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return (
        ranked[0][1],
        ranked[0][2],
    )


def _download_cropped_headshot(
    image_url: str,
    destination: Path,
) -> Optional[Path]:
    """
    Download a real player image and crop it to the upper head/shoulders area.

    This reduces visible shirt/club branding when the source photo is not
    national-team-specific.
    """
    try:
        response = requests.get(
            image_url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        from io import BytesIO

        with Image.open(
            BytesIO(
                response.content
            )
        ) as image:
            image.load()

            image = image.convert(
                "RGB"
            )

            width, height = (
                image.size
            )

            # Crop mostly to the face + shoulders.
            left = int(
                width * 0.14
            )
            right = int(
                width * 0.86
            )
            top = 0
            bottom = int(
                height * 0.58
            )

            if (
                right <= left
                or bottom <= top
            ):
                cropped = image
            else:
                cropped = image.crop(
                    (
                        left,
                        top,
                        right,
                        bottom,
                    )
                )

            canvas = Image.new(
                "RGB",
                (700, 700),
                "white",
            )

            cropped.thumbnail(
                (620, 620)
            )

            x = (
                700
                - cropped.width
            ) // 2

            y = (
                700
                - cropped.height
            ) // 2

            canvas.paste(
                cropped,
                (
                    x,
                    y,
                ),
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            canvas.save(
                destination,
                format="JPEG",
                quality=92,
            )

        if _valid_cached_image(
            destination
        ):
            return destination

    except Exception:
        try:
            destination.unlink(
                missing_ok=True
            )
        except Exception:
            pass

    return None


def search_player_artwork_url(
    player_name: str,
) -> Optional[Tuple[str, dict]]:
    """
    Search TheSportsDB for general/current player artwork.
    """
    name = str(
        player_name
    ).strip()

    if not name:
        return None

    url = (
        f"{THESPORTSDB_BASE}/"
        f"{THESPORTSDB_API_KEY}/"
        f"searchplayers.php?"
        f"p={quote(name)}"
    )

    payload = _request_json(
        url
    )

    if not payload:
        return None

    players = payload.get(
        "player"
    )

    if not players:
        return None

    selected = None

    for item in players:
        candidate = str(
            item.get(
                "strPlayer",
                "",
            )
        ).strip()

        if (
            candidate.casefold()
            == name.casefold()
        ):
            selected = item
            break

    if selected is None:
        selected = players[0]

    for key in (
        "strCutout",
        "strThumb",
        "strRender",
        "strFanart1",
    ):
        candidate = selected.get(
            key
        )

        if (
            isinstance(
                candidate,
                str,
            )
            and candidate.startswith(
                (
                    "http://",
                    "https://",
                )
            )
        ):
            return (
                candidate,
                {
                    "provider": "TheSportsDB",
                    "player": selected.get(
                        "strPlayer"
                    ),
                    "current_team": selected.get(
                        "strTeam"
                    ),
                    "nationality": selected.get(
                        "strNationality"
                    ),
                },
            )

    return None


def search_team_badge_url(
    team_name: str,
) -> Optional[str]:
    name = str(
        team_name
    ).strip()

    if not name:
        return None

    url = (
        f"{THESPORTSDB_BASE}/"
        f"{THESPORTSDB_API_KEY}/"
        f"searchteams.php?"
        f"t={quote(name)}"
    )

    payload = _request_json(
        url
    )

    if not payload:
        return None

    teams = payload.get(
        "teams"
    )

    if not teams:
        return None

    selected = None

    for item in teams:
        candidate = str(
            item.get(
                "strTeam",
                "",
            )
        ).strip()

        if (
            candidate.casefold()
            == name.casefold()
        ):
            selected = item
            break

    if selected is None:
        selected = teams[0]

    for key in (
        "strBadge",
        "strLogo",
    ):
        candidate = selected.get(
            key
        )

        if (
            isinstance(
                candidate,
                str,
            )
            and candidate.startswith(
                (
                    "http://",
                    "https://",
                )
            )
        ):
            return candidate

    return None




def _team_name_tokens(team_name: str) -> list[str]:
    """Return meaningful normalised tokens used to validate team badge candidates."""
    norm = _normalise_search_text(team_name)
    stop = {
        "fc", "cf", "afc", "sc", "club", "football", "futbol",
        "soccer", "the", "de", "of", "and", "team", "national",
    }
    return [
        token for token in re.split(r"\W+", norm)
        if len(token) >= 3 and token not in stop
    ]


def _score_team_badge_candidate(
    *,
    team_name: str,
    title_text: str,
    metadata_text: str,
    width=None,
    height=None,
) -> tuple[int, int]:
    """Score a Wikimedia image for use as a professional team crest.

    The filter rewards exact team-name evidence and logo/crest terminology,
    while rejecting flags, kits, photos, maps, tournament graphics and other
    unrelated artwork.
    """
    title_norm = _normalise_search_text(title_text)
    meta_norm = _normalise_search_text(metadata_text)
    combined = f" {title_norm} {meta_norm} "
    tokens = _team_name_tokens(team_name)

    if not tokens:
        return (-999, 0)

    title_hits = sum(1 for token in tokens if token in title_norm)
    combined_hits = sum(1 for token in tokens if token in combined)

    if combined_hits == 0:
        return (-999, 0)

    score = title_hits * 18 + combined_hits * 8

    team_norm = _normalise_search_text(team_name).strip()
    if team_norm and team_norm in title_norm:
        score += 35
    elif team_norm and team_norm in combined:
        score += 20

    if any(term in combined for term in (
        " logo", " crest", " badge", " emblem",
        " association", " federation", " football federation",
        " football association",
    )):
        score += 24

    # Vector originals are especially useful because Commons can provide
    # a sharp raster thumbnail at the requested display size.
    if title_norm.endswith(".svg") or " svg " in combined:
        score += 10

    # Prefer approximately square artwork, typical of badges and crests.
    try:
        w = float(width)
        h = float(height)
        if w > 0 and h > 0:
            aspect = w / h
            if 0.65 <= aspect <= 1.35:
                score += 12
            elif aspect < 0.40 or aspect > 2.20:
                score -= 18
    except Exception:
        pass

    bad_terms = (
        " flag", " jersey", " shirt", " kit", " uniform",
        " squad", " team photo", " lineup", " line-up", " player",
        " stadium", " match", " ticket", " poster", " map",
        " world cup trophy", " tournament logo", " competition logo",
        " wordmark", " text logo", " monochrome",
    )
    for term in bad_terms:
        if term in combined:
            score -= 22

    # A candidate should normally identify at least one meaningful team token
    # in the title itself, not only in broad Commons metadata.
    if title_hits == 0:
        score -= 25

    return score, title_hits


def _commons_team_badge(
    team_name: str,
) -> Optional[Tuple[str, dict]]:
    """Return the best Wikimedia Commons crest candidate for any team.

    Works for both clubs and national teams and intentionally avoids hardcoded
    Argentina/France URLs.
    """
    name = str(team_name or "").strip()
    if not name:
        return None

    queries = [
        f'"{name}" football crest',
        f'"{name}" football logo',
        f'"{name}" badge',
        f'"{name}" football association logo',
    ]

    ranked = []
    seen_urls = set()

    for query_index, search_query in enumerate(queries):
        payload = _request_json(
            COMMONS_API,
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrnamespace": 6,
                "gsrsearch": search_query,
                "gsrlimit": 40,
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata",
                "iiurlwidth": 1000,
                "origin": "*",
            },
        )
        if not payload:
            continue

        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            title = str(page.get("title", "")).strip()
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            image_url = info.get("thumburl") or info.get("url")
            if not isinstance(image_url, str) or image_url in seen_urls:
                continue
            seen_urls.add(image_url)

            ext = info.get("extmetadata", {})
            metadata_text = " ".join(
                str(ext.get(key, {}).get("value", ""))
                for key in (
                    "ImageDescription", "ObjectName", "Categories",
                    "Credit", "Artist",
                )
            )

            score, title_hits = _score_team_badge_candidate(
                team_name=name,
                title_text=title,
                metadata_text=metadata_text,
                width=info.get("width"),
                height=info.get("height"),
            )
            score += max(0, 6 - query_index)

            if title_hits < 1 or score < 30:
                continue

            ranked.append((
                score,
                image_url,
                {
                    "provider": "Wikimedia Commons",
                    "selection_version": 1,
                    "image_kind": "team_badge",
                    "team": name,
                    "search_query": search_query,
                    "file_title": title,
                    "description_url": info.get("descriptionurl"),
                    "license": ext.get("LicenseShortName", {}).get("value"),
                    "license_url": ext.get("LicenseUrl", {}).get("value"),
                    "ranking_score": score,
                },
            ))

    if not ranked:
        return None

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1], ranked[0][2]


def _download_badge_image(
    image_url: str,
    destination: Path,
) -> Optional[Path]:
    """Download a crest while preserving transparent backgrounds."""
    try:
        response = requests.get(
            image_url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(BytesIO(response.content)) as image:
            image.load()
            # RGBA keeps the clean transparent boundary expected for report
            # headers and avoids the white JPEG rectangle around a crest.
            if image.mode not in ("RGBA", "LA"):
                image = image.convert("RGBA")
            else:
                image = image.convert("RGBA")
            image.thumbnail((1200, 1200))
            image.save(destination, format="PNG", optimize=True)

        return destination if _valid_cached_image(destination) else None
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def _team_metadata_path(team_name: str) -> Path:
    return METADATA_CACHE_DIR / f"team__{_safe_stem(team_name)}.json"


def _save_team_metadata(team_name: str, metadata: dict) -> None:
    _ensure_cache_dirs()
    try:
        _team_metadata_path(team_name).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def get_team_badge_metadata(team_name: str) -> Optional[dict]:
    path = _team_metadata_path(team_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _generate_neutral_player_portrait(
    player_name: str,
    represented_team: Optional[str],
    destination: Path,
) -> Optional[Path]:
    """
    Last-resort fallback when no trustworthy real contextual image exists.
    """
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)

        from PIL import ImageDraw, ImageFont

        image = Image.new(
            "RGB",
            (700, 850),
            "white",
        )

        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 700, 150), fill=(28, 28, 32))
        draw.ellipse(
            (245, 210, 455, 420),
            fill=(205, 205, 210),
            outline=(80, 80, 85),
            width=4,
        )
        draw.rounded_rectangle(
            (170, 430, 530, 690),
            radius=90,
            fill=(215, 215, 220),
            outline=(80, 80, 85),
            width=4,
        )

        initials = "".join(
            part[0]
            for part in str(player_name).replace("-", " ").split()
            if part
        )[:3].upper()

        try:
            font_large = ImageFont.truetype("arial.ttf", 72)
            font_name = ImageFont.truetype("arial.ttf", 36)
            font_team = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            font_large = ImageFont.load_default()
            font_name = ImageFont.load_default()
            font_team = ImageFont.load_default()

        draw.text((350, 315), initials, fill=(45, 45, 50), font=font_large, anchor="mm")
        draw.text((350, 735), str(player_name), fill=(25, 25, 30), font=font_name, anchor="mm")
        draw.text((350, 790), str(represented_team or "Football"), fill=(90, 90, 95), font=font_team, anchor="mm")

        image.save(destination, format="JPEG", quality=92)

        _save_metadata(
            player_name,
            represented_team,
            {
                "provider": "LiveMatch Intelligence",
                "type": "neutral_placeholder",
                "player": player_name,
                "represented_team": represented_team,
                "context_mismatch": False,
            },
        )

        return destination if _valid_cached_image(destination) else None

    except Exception:
        return None


def get_player_image(
    player_name: str,
    represented_team: Optional[str] = None,
    competition: Optional[str] = None,
    force_refresh: bool = False,
) -> Optional[Path]:
    """
    Retrieve the best available player image.

    National-team context:
    1. Context-valid cached image.
    2. Individual Wikimedia photo matching player + represented team.
    3. Generic individual Wikimedia photo of the same player, cropped tightly
       to head/shoulders to minimise club-shirt context.
    4. Neutral placeholder.

    Club/generic context:
    1. Valid cache.
    2. TheSportsDB artwork.
    3. Generic Wikimedia individual headshot.
    """
    _ensure_cache_dirs()

    destination = _player_cache_path(
        player_name,
        represented_team,
    )

    if (
        not force_refresh
        and _cached_player_image_is_context_valid(
            player_name,
            represented_team,
            destination,
        )
    ):
        return destination

    if destination.exists():
        try:
            destination.unlink(
                missing_ok=True
            )
        except Exception:
            pass

    # -----------------------------------------------------
    # National-team context
    # -----------------------------------------------------
    if _is_national_team(
        represented_team
    ):
        contextual = _commons_player_image(
            player_name,
            represented_team,
            competition,
        )

        if contextual:
            image_url, metadata = contextual

            downloaded = _download_image(
                image_url,
                destination,
            )

            if downloaded:
                _save_metadata(
                    player_name,
                    represented_team,
                    metadata,
                )

                return downloaded

        # Second-stage fallback: real individual photo of the same player,
        # tightly cropped to reduce club-kit context.
        generic = _commons_generic_individual_portrait(
            player_name
        )

        if generic:
            image_url, metadata = generic

            downloaded = _download_cropped_headshot(
                image_url,
                destination,
            )

            if downloaded:
                metadata[
                    "represented_team"
                ] = represented_team
                metadata[
                    "competition"
                ] = competition
                metadata[
                    "context_mismatch"
                ] = False

                _save_metadata(
                    player_name,
                    represented_team,
                    metadata,
                )

                return downloaded

        return _generate_neutral_player_portrait(
            player_name,
            represented_team,
            destination,
        )

    # -----------------------------------------------------
    # Club / generic context
    # -----------------------------------------------------
    sportsdb = search_player_artwork_url(
        player_name
    )

    if sportsdb:
        image_url, metadata = (
            sportsdb
        )

        downloaded = _download_image(
            image_url,
            destination,
        )

        if downloaded:
            _save_metadata(
                player_name,
                represented_team,
                metadata,
            )

            return downloaded

    generic = _commons_generic_individual_portrait(
        player_name
    )

    if generic:
        image_url, metadata = generic

        downloaded = _download_cropped_headshot(
            image_url,
            destination,
        )

        if downloaded:
            metadata[
                "represented_team"
            ] = represented_team

            _save_metadata(
                player_name,
                represented_team,
                metadata,
            )

            return downloaded

    return None


def get_team_badge(
    team_name: str,
    force_refresh: bool = False,
) -> Optional[Path]:
    """Retrieve a professional team crest for clubs or national teams.

    Resolution order:
    1. Existing transparent PNG cache.
    2. Wikimedia Commons ranked crest/logo candidate.
    3. TheSportsDB badge/logo fallback.

    The resolver is team-agnostic: no Argentina/France hardcoding is used.
    """
    _ensure_cache_dirs()

    name = str(team_name or "").strip()
    if not name:
        return None

    destination = TEAM_CACHE_DIR / f"{_safe_stem(name)}.png"
    legacy_destination = TEAM_CACHE_DIR / f"{_safe_stem(name)}.jpg"

    if not force_refresh:
        if _valid_cached_image(destination):
            return destination
        # Preserve compatibility with existing projects until a refreshed PNG
        # is requested. New downloads are always cached as PNG.
        if _valid_cached_image(legacy_destination):
            return legacy_destination

    if force_refresh:
        for path in (destination, legacy_destination):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    commons = _commons_team_badge(name)
    if commons:
        image_url, metadata = commons
        downloaded = _download_badge_image(image_url, destination)
        if downloaded:
            _save_team_metadata(name, metadata)
            return downloaded

    image_url = search_team_badge_url(name)
    if image_url:
        downloaded = _download_badge_image(image_url, destination)
        if downloaded:
            _save_team_metadata(
                name,
                {
                    "provider": "TheSportsDB",
                    "selection_version": 1,
                    "image_kind": "team_badge",
                    "team": name,
                },
            )
            return downloaded

    return None


def get_player_image_metadata(
    player_name: str,
    represented_team: Optional[str] = None,
) -> Optional[dict]:
    path = _metadata_path(
        player_name,
        represented_team,
    )

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None


if __name__ == "__main__":
    print(
        "Context-aware image fetcher ready."
    )
    print(
        "Player cache:",
        PLAYER_CACHE_DIR,
    )
    print(
        "Team cache:",
        TEAM_CACHE_DIR,
    )

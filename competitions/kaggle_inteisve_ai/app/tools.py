"""External API tools: geocoding, facility search, ETA routing."""
import time
import math
import requests
from typing import List, Optional, Tuple
from app.models import Facility
from app.observability import METRICS, log_event

NOMINATIM = "https://nominatim.openstreetmap.org/search"
OSRM_ROUTE = "https://router.project-osrm.org/route/v1/driving"

HEADERS = {
    "User-Agent": "EmergencyCareNavigator/1.0 (local app)"
}


def tool_geocode(query: str, max_retries: int = 2) -> Tuple[float, float, str]:
    """Geocode location query with retry logic."""
    METRICS["tool_calls"] += 1
    log_event("tool_call", tool="geocode", query=query)
    params = {"q": query, "format": "json", "limit": 1}
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(NOMINATIM, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            data = r.json()
            if not data:
                raise ValueError(f"No geocode results for: {query}")
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            display = data[0].get("display_name", query)
            return lat, lon, display
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                log_event("tool_retry", tool="geocode", attempt=attempt+1, error=str(e))
            else:
                METRICS["errors"] += 1
                log_event("tool_error", tool="geocode", error=str(e), final=True)
        except (ValueError, KeyError, IndexError) as e:
            last_error = e
            METRICS["errors"] += 1
            log_event("tool_error", tool="geocode", error=str(e))
            break
    
    raise ValueError(
        f"Could not geocode location '{query}'. "
        f"Please try a more specific location (e.g., 'City, Country'). "
        f"Error: {last_error}"
    )


def tool_find_facilities(
    lat: float, lon: float, query: str, kind: str, limit: int = 7, max_retries: int = 2
) -> List[Facility]:
    """Find nearby facilities with retry logic."""
    METRICS["tool_calls"] += 1
    log_event("tool_call", tool="find_facilities", kind=kind, query=query, lat=lat, lon=lon)
    params = {
        "q": query,
        "format": "json",
        "limit": limit,
        "viewbox": f"{lon-0.15},{lat+0.15},{lon+0.15},{lat-0.15}",
        "bounded": 1
    }
    
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(NOMINATIM, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            data = r.json() or []
            facilities = []
            for item in data:
                name = item.get("display_name", "Unknown")
                facilities.append(Facility(
                    name=name.split(",")[0],
                    address=name,
                    lat=float(item["lat"]),
                    lon=float(item["lon"]),
                    kind=kind,
                    source="OSM/Nominatim"
                ))
            return facilities
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                log_event("tool_retry", tool="find_facilities", attempt=attempt+1, error=str(e))
            else:
                METRICS["errors"] += 1
                log_event("tool_error", tool="find_facilities", error=str(e), final=True)
                return []
        except (ValueError, KeyError) as e:
            METRICS["errors"] += 1
            log_event("tool_error", tool="find_facilities", error=str(e))
            return []
    
    return []


def tool_eta_minutes(
    origin: Tuple[float, float], dest: Tuple[float, float], max_retries: int = 1
) -> Optional[int]:
    """Get ETA from OSRM. Returns None if OSRM fails."""
    METRICS["tool_calls"] += 1
    (olat, olon) = origin
    (dlat, dlon) = dest
    log_event("tool_call", tool="eta", origin=origin, dest=dest)
    url = f"{OSRM_ROUTE}/{olon},{olat};{dlon},{dlat}"
    params = {"overview": "false"}
    
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
                log_event("tool_warn", tool="eta", status=r.status_code, note="OSRM failed, using distance-only")
                return None
            data = r.json()
            routes = data.get("routes") or []
            if not routes:
                log_event("tool_warn", tool="eta", note="No routes from OSRM, using distance-only")
                return None
            seconds = routes[0].get("duration")
            if seconds is None:
                log_event("tool_warn", tool="eta", note="No duration in OSRM response, using distance-only")
                return None
            return int(round(seconds / 60))
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(0.5)
                log_event("tool_retry", tool="eta", attempt=attempt+1, error=str(e))
            else:
                METRICS["errors"] += 1
                log_event("tool_error", tool="eta", error=str(e), note="OSRM failed, using distance-only")
                return None
    
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371.0
    p = math.pi / 180
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = math.sin(dlat/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))







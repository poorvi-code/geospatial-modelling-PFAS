import logging
import requests
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from implementation.data_sources.pdh_schema import PDHPoint, PDHExportResponse
from implementation.utils.retry_cache import retry, local_cache

log = logging.getLogger(__name__)

class PDHClient:
    BASE_URL = "https://pdh.cnrs.fr/api"
    CACHE_DIR = Path("outputs/cache/pdh")

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        # Ensure cache dir exists
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @retry(max_attempts=3)
    @local_cache(CACHE_DIR)
    def fetch_export(self, params: Optional[Dict[str, Any]] = None) -> List[PDHPoint]:
        """
        Fetches full dataset export.
        Params: country, dataset_id, category, etc.
        """
        url = f"{self.BASE_URL}/export"
        log.info(f"Fetching PDH export data from {url}...")
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        validated = PDHExportResponse(**data)
        return validated.results

    @retry(max_attempts=3)
    def fetch_map_data(self, bbox: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches summary data suitable for mapping.
        Bbox format: min_lon,min_lat,max_lon,max_lat
        """
        url = f"{self.BASE_URL}/map_data"
        params = {"bbox": bbox} if bbox else {}
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @retry(max_attempts=3)
    def fetch_point_details(self, point_id: Union[int, str]) -> Dict[str, Any]:
        """Fetches detailed metadata for a single point."""
        url = f"{self.BASE_URL}/point_details/{point_id}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

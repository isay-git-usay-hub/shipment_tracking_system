"""
External data services for fetching weather, port, and traffic data
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import aiohttp
from config.settings import settings

logger = logging.getLogger(__name__)


class WeatherService:
    """Service for fetching weather data"""

    def __init__(self):
        self.api_key = settings.WEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.cache = {}

    async def get_weather_data(self, port: str, date: datetime) -> Dict[str, Any]:
        """Get weather data for a port and date"""
        try:
            # Check cache first
            cache_key = f"{port}_{date.date()}"
            if cache_key in self.cache:
                cached_data, cached_time = self.cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < 3600:  # 1 hour cache
                    return cached_data

            # Get coordinates for port (simplified mapping)
            coords = self._get_port_coordinates(port)
            if not coords:
                return self._get_default_weather()

            # Fetch weather data
            if self.api_key:
                weather_data = await self._fetch_openweather_data(coords, date)
            else:
                weather_data = self._get_default_weather()

            # Cache result
            self.cache[cache_key] = (weather_data, datetime.now())

            return weather_data

        except Exception as e:
            logger.warning(f"Error fetching weather data for {port}: {e}")
            return self._get_default_weather()

    def _get_port_coordinates(self, port: str) -> Optional[Dict[str, float]]:
        """Get coordinates for major ports"""
        # Simplified port coordinates mapping
        port_coords = {
            'shanghai': {'lat': 31.2304, 'lon': 121.4737},
            'singapore': {'lat': 1.3521, 'lon': 103.8198},
            'rotterdam': {'lat': 51.9244, 'lon': 4.4777},
            'los angeles': {'lat': 33.7501, 'lon': -118.2537},
            'hamburg': {'lat': 53.5511, 'lon': 9.9937},
            'antwerp': {'lat': 51.2194, 'lon': 4.4025},
            'hong kong': {'lat': 22.3193, 'lon': 114.1694},
            'dubai': {'lat': 25.2048, 'lon': 55.2708},
            'new york': {'lat': 40.7128, 'lon': -74.0060},
            'bremen': {'lat': 53.0793, 'lon': 8.8017}
        }

        port_lower = port.lower()
        for key, coords in port_coords.items():
            if key in port_lower:
                return coords

        return None

    async def _fetch_openweather_data(self, coords: Dict[str, float], date: datetime) -> Dict[str, Any]:
        """Fetch data from OpenWeatherMap API"""
        url = f"{self.base_url}/weather"
        params = {
            'lat': coords['lat'],
            'lon': coords['lon'],
            'appid': self.api_key,
            'units': 'metric'
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'temperature': data['main']['temp'],
                        'humidity': data['main']['humidity'],
                        'wind_speed': data['wind']['speed'],
                        'precipitation': data.get('rain', {}).get('1h', 0),
                        'conditions': data['weather'][0]['description'],
                        'visibility': data.get('visibility', 10000) / 1000  # Convert to km
                    }
                else:
                    logger.warning(f"Weather API returned status {response.status}")
                    return self._get_default_weather()

    def _get_default_weather(self) -> Dict[str, Any]:
        """Get default weather data when API is unavailable"""
        return {
            'temperature': 20,
            'humidity': 60,
            'wind_speed': 5,
            'precipitation': 0,
            'conditions': 'clear',
            'visibility': 10
        }


class PortService:
    """Service for fetching port congestion and status data"""

    def __init__(self):
        self.api_key = settings.PORT_API_KEY
        self.cache = {}

    async def get_port_congestion(self, port: str) -> Dict[str, Any]:
        """Get port congestion data"""
        try:
            # Check cache first
            cache_key = f"port_{port}"
            if cache_key in self.cache:
                cached_data, cached_time = self.cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < 1800:  # 30 min cache
                    return cached_data

            # In a real implementation, this would call actual port APIs
            # For demo purposes, we'll simulate port congestion data
            congestion_data = self._simulate_port_congestion(port)

            # Cache result
            self.cache[cache_key] = (congestion_data, datetime.now())

            return congestion_data

        except Exception as e:
            logger.warning(f"Error fetching port data for {port}: {e}")
            return self._get_default_port_data()

    def _simulate_port_congestion(self, port: str) -> Dict[str, Any]:
        """Simulate port congestion data"""
        # Simulate different congestion levels for different ports
        port_congestion_map = {
            'shanghai': 0.7,  # High congestion
            'los angeles': 0.8,  # Very high congestion
            'singapore': 0.4,  # Moderate congestion
            'rotterdam': 0.3,  # Low congestion
            'hamburg': 0.3,  # Low congestion
            'antwerp': 0.4,  # Moderate congestion
        }

        congestion_level = port_congestion_map.get(port.lower(), 0.5)

        # Add some randomness based on time of day
        hour = datetime.now().hour
        if 6 <= hour <= 18:  # Business hours - higher congestion
            congestion_level = min(congestion_level + 0.1, 1.0)

        return {
            'congestion_level': congestion_level,
            'waiting_time_hours': congestion_level * 12,  # Up to 12 hours wait
            'available_berths': max(1, int(10 * (1 - congestion_level))),
            'total_berths': 10,
            'status': self._get_congestion_status(congestion_level),
            'last_updated': datetime.now().isoformat()
        }

    def _get_congestion_status(self, level: float) -> str:
        """Get congestion status description"""
        if level >= 0.8:
            return "severe"
        elif level >= 0.6:
            return "high"
        elif level >= 0.4:
            return "moderate"
        elif level >= 0.2:
            return "low"
        else:
            return "minimal"

    def _get_default_port_data(self) -> Dict[str, Any]:
        """Get default port data"""
        return {
            'congestion_level': 0.5,
            'waiting_time_hours': 6,
            'available_berths': 5,
            'total_berths': 10,
            'status': 'moderate',
            'last_updated': datetime.now().isoformat()
        }


class TrafficService:
    """Service for fetching traffic and route data"""

    def __init__(self):
        self.cache = {}

    async def get_route_conditions(self, origin: str, destination: str) -> Dict[str, Any]:
        """Get traffic conditions for a route"""
        try:
            cache_key = f"route_{origin}_{destination}"
            if cache_key in self.cache:
                cached_data, cached_time = self.cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < 1800:  # 30 min cache
                    return cached_data

            # Simulate route conditions
            route_data = self._simulate_route_conditions(origin, destination)

            # Cache result
            self.cache[cache_key] = (route_data, datetime.now())

            return route_data

        except Exception as e:
            logger.warning(f"Error fetching route data: {e}")
            return self._get_default_route_data()

    def _simulate_route_conditions(self, origin: str, destination: str) -> Dict[str, Any]:
        """Simulate route conditions"""
        # Calculate base complexity based on route
        distance_factor = self._estimate_route_complexity(origin, destination)

        return {
            'complexity_score': distance_factor,
            'estimated_transit_days': max(1, int(distance_factor * 15)),  # 1-15 days
            'route_type': 'international' if distance_factor > 0.5 else 'domestic',
            'intermediate_ports': max(0, int(distance_factor * 5)),
            'customs_checkpoints': max(1, int(distance_factor * 3)),
            'last_updated': datetime.now().isoformat()
        }

    def _estimate_route_complexity(self, origin: str, destination: str) -> float:
        """Estimate route complexity (0-1 scale)"""
        # Simplified complexity estimation
        major_routes = {
            ('shanghai', 'los angeles'): 0.8,
            ('singapore', 'rotterdam'): 0.9,
            ('hamburg', 'new york'): 0.7,
            ('dubai', 'singapore'): 0.6,
        }

        # Check for known routes
        route_key = (origin.lower(), destination.lower())
        reverse_key = (destination.lower(), origin.lower())

        if route_key in major_routes:
            return major_routes[route_key]
        elif reverse_key in major_routes:
            return major_routes[reverse_key]
        else:
            # Default complexity based on port names
            return 0.5


# Global service instances
weather_service = WeatherService()
port_service = PortService()
traffic_service = TrafficService()

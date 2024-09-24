# weather_handler.py

from plugins import weather_api

def get_weather(city_name: str) -> str:
    """处理天气查询逻辑，返回查询结果"""
    return weather_api.format_weather(city_name)

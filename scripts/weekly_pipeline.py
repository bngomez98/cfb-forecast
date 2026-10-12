"""Compatibility entry point for the Sunday service."""

import os
from cfb_forecast.sunday import sunday_run


def main():
    sunday_run(season=int(os.environ.get("CFB_FORECAST_SEASON", "2026")))


if __name__ == "__main__":
    main()

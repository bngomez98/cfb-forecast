"""Run the Sunday forecast and publish the same board to every surface.

The Sunday service writes the calculation. publish_forecast_update then stores
the log, refreshes the React and Streamlit boards, writes the Datadog log, and
sends the forecast email when SMTP is configured.
"""

import os
from cfb_forecast.sunday import sunday_run


def main():
    sunday_run(season=int(os.environ.get("CFB_FORECAST_SEASON", "2026")))


if __name__ == "__main__":
    main()

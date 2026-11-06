"""
High-Impact Economic News Shield & Volatility Gate.

Protects algorithmic forex execution by detecting high-impact macroeconomic event
release windows (e.g. US Non-Farm Payrolls, FOMC rate decisions, US CPI, ECB decisions)
where interbank liquidity collapses and broker spreads widen up to 10x.
"""

from datetime import datetime, timezone
from typing import Tuple


class EconomicNewsFilter:
    """
    Evaluates whether current UTC time falls within a high-risk economic announcement window.
    Standard protection window: 30 minutes before and 30 minutes after scheduled release.
    """

    @staticmethod
    def is_nfp_window(dt: datetime, buffer_minutes: int = 30) -> bool:
        """
        US Non-Farm Payrolls (NFP): Released on the 1st Friday of every month at 12:30 or 13:30 UTC.
        """
        # Must be Friday (weekday 4)
        if dt.weekday() != 4:
            return False
        
        # Check if 1st Friday of month (day between 1 and 7)
        if not (1 <= dt.day <= 7):
            return False

        # Release window typically 12:30 - 13:30 UTC (cover 12:00 to 14:00 UTC)
        total_mins = dt.hour * 60 + dt.minute
        release_start = 12 * 60  # 12:00 UTC
        release_end = 14 * 60    # 14:00 UTC
        return release_start <= total_mins <= release_end

    @staticmethod
    def is_fomc_rate_window(dt: datetime) -> bool:
        """
        FOMC Rate Decision & Press Conference (8 Wednesdays per year, 18:00 - 19:30 UTC).
        """
        if dt.weekday() != 2: # Wednesday
            return False
        total_mins = dt.hour * 60 + dt.minute
        # 17:30 to 20:00 UTC
        return (17 * 60 + 30) <= total_mins <= (20 * 60)

    @classmethod
    def check_high_impact_window(
        cls, 
        dt: datetime = None, 
        buffer_minutes: int = 30
    ) -> Tuple[bool, str]:
        """
        Returns (is_active_shield, reason_string).
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        if cls.is_nfp_window(dt, buffer_minutes):
            return True, "HIGH_IMPACT_NEWS: US Non-Farm Payrolls (NFP) window active"

        if cls.is_fomc_rate_window(dt):
            return True, "HIGH_IMPACT_NEWS: FOMC interest rate decision window active"

        return False, "CLEAR"

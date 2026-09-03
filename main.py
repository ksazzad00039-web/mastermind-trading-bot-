# ============================================================
# MARKET RESEARCH TELEGRAM BOT
# FINAL SINGLE-FILE VERSION
# Research / Paper-Trading Only
# ============================================================

import os
import re
import math
import logging
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from flask import Flask, jsonify

from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# New Google GenAI SDK
try:
    from google import genai
except Exception:
    genai = None


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash").strip()

ALPHA_VANTAGE_API_KEY = os.getenv(
    "ALPHA_VANTAGE_API_KEY", ""
).strip()

PORT = int(os.getenv("PORT", "10000"))


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("MarketResearchBot")


# ============================================================
# 3. FLASK HEALTH SERVER
# ============================================================

flask_app = Flask(__name__)


@flask_app.route("/")
def home():
    return jsonify({
        "status": "online",
        "bot": "Market Research Bot",
        "mode": "research/paper",
        "time_utc": datetime.now(timezone.utc).isoformat(),
    })


@flask_app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN),
        "gemini_configured": bool(GEMINI_API_KEY),
        "alpha_vantage_configured": bool(ALPHA_VANTAGE_API_KEY),
        "time_utc": datetime.now(timezone.utc).isoformat(),
    })


def run_flask():
    try:
        flask_app.run(
            host="0.0.0.0",
            port=PORT,
            use_reloader=False,
        )
    except Exception:
        logger.exception("Flask server stopped.")


# ============================================================
# 4. GEMINI
# ============================================================

gemini_client = None

if GEMINI_API_KEY and genai is not None:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )
        logger.info("Gemini client initialized.")
    except Exception:
        logger.exception("Gemini initialization failed.")


# ============================================================
# 5. SYMBOL MAP
# ============================================================

SYMBOL_MAP = {

    # Forex
    "EURUSD": "EURUSD=X",
    "EUR/USD": "EURUSD=X",

    "GBPUSD": "GBPUSD=X",
    "GBP/USD": "GBPUSD=X",

    "AUDUSD": "AUDUSD=X",
    "AUD/USD": "AUDUSD=X",

    "NZDUSD": "NZDUSD=X",
    "NZD/USD": "NZDUSD=X",

    "USDJPY": "JPY=X",
    "USD/JPY": "JPY=X",

    "USDCHF": "CHF=X",
    "USD/CHF": "CHF=X",

    "USDCAD": "CAD=X",
    "USD/CAD": "CAD=X",

    # Gold
    "GOLD": "GC=F",
    "XAUUSD": "GC=F",
    "XAU/USD": "GC=F",

    # Crypto
    "BTCUSD": "BTC-USD",
    "BTC/USD": "BTC-USD",

    "ETHUSD": "ETH-USD",
    "ETH/USD": "ETH-USD",
}


DISPLAY_NAMES = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "AUDUSD": "AUD/USD",
    "NZDUSD": "NZD/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "USDCAD": "USD/CAD",
    "GOLD": "Gold / XAUUSD",
    "XAUUSD": "Gold / XAUUSD",
    "BTCUSD": "Bitcoin / BTCUSD",
    "ETHUSD": "Ethereum / ETHUSD",
}


WATCHLIST = [
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "NZDUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "GOLD",
    "BTCUSD",
    "ETHUSD",
]


# ============================================================
# 6. TIMEFRAMES
# ============================================================

TIMEFRAMES = {
    "5m":  ("5d", "5m"),
    "15m": ("5d", "15m"),
    "30m": ("1mo", "30m"),
    "1h":  ("1mo", "1h"),
    "2h":  ("3mo", "1h"),
    "4h":  ("6mo", "1h"),
    "1D":  ("2y", "1d"),
}


# ============================================================
# 7. DATA HELPERS
# ============================================================

def normalize_symbol(text: str):
    if not text:
        return None

    value = text.upper().strip()
    value = re.sub(r"\s+", "", value)

    return value


def resolve_symbol(symbol: str):
    normalized = normalize_symbol(symbol)

    if normalized in SYMBOL_MAP:
        return normalized, SYMBOL_MAP[normalized]

    return None, None


def validate_dataframe(df):
    if df is None:
        return False, "No dataframe returned."

    if df.empty:
        return False, "Market data is empty."

    required = ["Open", "High", "Low", "Close"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        return False, f"Missing columns: {missing}"

    df = df.copy()

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=required
    )

    if len(df) < 60:
        return False, (
            f"Not enough candles. "
            f"Received {len(df)}, need at least 60."
        )

    if df.index.has_duplicates:
        df = df[
            ~df.index.duplicated(
                keep="last"
            )
        ]

    return True, "OK"


def get_market_data(
    symbol: str,
    period="1mo",
    interval="1h"
):
    display_symbol, yf_symbol = resolve_symbol(symbol)

    if not yf_symbol:
        return None, (
            f"Unknown symbol: {symbol}"
        )

    try:
        logger.info(
            "Downloading %s -> %s | %s | %s",
            display_symbol,
            yf_symbol,
            period,
            interval,
        )

        df = yf.download(
            yf_symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        valid, message = validate_dataframe(df)

        if not valid:
            logger.warning(
                "Data validation failed: %s",
                message,
            )
            return None, message

        return df, "OK"

    except Exception as e:
        logger.exception(
            "Market data error."
        )
        return None, str(e)


# ============================================================
# 8. INDICATORS
# ============================================================

def calculate_indicators(df):
    df = df.copy()

    try:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # EMA
        df["EMA_20"] = (
            close.ewm(
                span=20,
                adjust=False
            ).mean()
        )

        df["EMA_50"] = (
            close.ewm(
                span=50,
                adjust=False
            ).mean()
        )

        df["EMA_200"] = (
            close.ewm(
                span=200,
                adjust=False
            ).mean()
        )

        # SMA
        df["SMA_20"] = (
            close.rolling(20).mean()
        )

        df["SMA_50"] = (
            close.rolling(50).mean()
        )

        df["SMA_200"] = (
            close.rolling(200).mean()
        )

        # RSI - Wilder style
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(
            0,
            np.nan
        )

        df["RSI"] = (
            100 - (
                100 / (1 + rs)
            )
        )

        # MACD
        ema12 = close.ewm(
            span=12,
            adjust=False
        ).mean()

        ema26 = close.ewm(
            span=26,
            adjust=False
        ).mean()

        df["MACD"] = ema12 - ema26

        df["MACD_SIGNAL"] = (
            df["MACD"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
        )

        df["MACD_HIST"] = (
            df["MACD"]
            - df["MACD_SIGNAL"]
        )

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()

        df["BB_MID"] = bb_mid
        df["BB_UPPER"] = (
            bb_mid + 2 * bb_std
        )
        df["BB_LOWER"] = (
            bb_mid - 2 * bb_std
        )

        # ATR
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (
            high - prev_close
        ).abs()

        tr3 = (
            low - prev_close
        ).abs()

        true_range = pd.concat(
            [tr1, tr2, tr3],
            axis=1
        ).max(axis=1)

        df["TR"] = true_range

        df["ATR"] = (
            true_range
            .ewm(
                alpha=1 / 14,
                adjust=False
            )
            .mean()
        )

        df["ATR_PCT"] = (
            df["ATR"] / close
        ) * 100

        # Returns / volatility
        df["RETURN"] = close.pct_change()

        df["VOLATILITY"] = (
            df["RETURN"]
            .rolling(20)
            .std()
            * np.sqrt(20)
            * 100
        )

        # Candle statistics
        df["BODY"] = (
            df["Close"]
            - df["Open"]
        ).abs()

        df["RANGE"] = (
            df["High"]
            - df["Low"]
        )

        df["UPPER_WICK"] = (
            df["High"]
            - df[
                ["Open", "Close"]
            ].max(axis=1)
        )

        df["LOWER_WICK"] = (
            df[
                ["Open", "Close"]
            ].min(axis=1)
            - df["Low"]
        )

        return df

    except Exception as e:
        logger.exception(
            "Indicator calculation error."
        )
        return df


# ============================================================
# 9. MARKET REGIME
# ============================================================

def determine_market_regime(df):

    try:
        if len(df) < 50:
            return "UNKNOWN_REGIME"

        row = df.iloc[-1]

        ema20 = row.get(
            "EMA_20",
            np.nan
        )

        ema50 = row.get(
            "EMA_50",
            np.nan
        )

        rsi = row.get(
            "RSI",
            np.nan
        )

        atr_pct = row.get(
            "ATR_PCT",
            np.nan
        )

        if any(
            pd.isna(x)
            for x in [
                ema20,
                ema50,
                rsi,
                atr_pct,
            ]
        ):
            return "UNKNOWN_REGIME"

        if atr_pct > 1.5:
            volatility_state = (
                "HIGH_VOLATILITY"
            )
        else:
            volatility_state = (
                "NORMAL_VOLATILITY"
            )

        if (
            ema20 > ema50
            and rsi >= 55
        ):
            return (
                "BULLISH_TRENDING | "
                + volatility_state
            )

        if (
            ema20 < ema50
            and rsi <= 45
        ):
            return (
                "BEARISH_TRENDING | "
                + volatility_state
            )

        return (
            "SIDEWAYS_RANGE | "
            + volatility_state
        )

    except Exception:
        logger.exception(
            "Regime calculation error."
        )
        return "UNKNOWN_REGIME"


# ============================================================
# 10. SWING DETECTION
# ============================================================

def detect_swing_points(
    df,
    lookback=3
):
    df = df.copy()

    df["SWING_HIGH"] = False
    df["SWING_LOW"] = False

    try:
        for i in range(
            lookback,
            len(df) - lookback
        ):

            current_high = df["High"].iloc[i]
            current_low = df["Low"].iloc[i]

            left_highs = df[
                "High"
            ].iloc[
                i - lookback:i
            ]

            right_highs = df[
                "High"
            ].iloc[
                i + 1:i + 1 + lookback
            ]

            left_lows = df[
                "Low"
            ].iloc[
                i - lookback:i
            ]

            right_lows = df[
                "Low"
            ].iloc[
                i + 1:i + 1 + lookback
            ]

            if (
                current_high > left_highs.max()
                and current_high > right_highs.max()
            ):
                df.iloc[
                    i,
                    df.columns.get_loc(
                        "SWING_HIGH"
                    )
                ] = True

            if (
                current_low < left_lows.min()
                and current_low < right_lows.min()
            ):
                df.iloc[
                    i,
                    df.columns.get_loc(
                        "SWING_LOW"
                    )
                ] = True

        return df

    except Exception:
        logger.exception(
            "Swing detection error."
        )
        return df


# ============================================================
# 11. MARKET STRUCTURE
# ============================================================

def determine_structure(df):

    try:
        swing_highs = df[
            df["SWING_HIGH"]
        ]

        swing_lows = df[
            df["SWING_LOW"]
        ]

        high_values = (
            swing_highs["High"]
            .tail(4)
            .tolist()
        )

        low_values = (
            swing_lows["Low"]
            .tail(4)
            .tolist()
        )

        structure = "UNKNOWN"

        if len(high_values) >= 2:
            if (
                high_values[-1]
                > high_values[-2]
            ):
                structure = "HH"
            else:
                structure = "LH"

        if len(low_values) >= 2:
            if (
                low_values[-1]
                > low_values[-2]
            ):
                if structure == "HH":
                    structure = "HH + HL"
                else:
                    structure = "HL"
            else:
                if structure == "LH":
                    structure = "LH + LL"
                else:
                    structure = "LL"

        return {
            "structure": structure,
            "recent_swing_high": (
                high_values[-1]
                if high_values
                else None
            ),
            "previous_swing_high": (
                high_values[-2]
                if len(high_values) >= 2
                else None
            ),
            "recent_swing_low": (
                low_values[-1]
                if low_values
                else None
            ),
            "previous_swing_low": (
                low_values[-2]
                if len(low_values) >= 2
                else None
            ),
        }

    except Exception:
        logger.exception(
            "Structure error."
        )

        return {
            "structure": "UNKNOWN"
        }


# ============================================================
# 12. LIQUIDITY / BOS RESEARCH
# ============================================================

def detect_liquidity_event(df):

    try:
        if len(df) < 5:
            return {
                "event": "UNKNOWN"
            }

        last = df.iloc[-1]

        recent_high = (
            df["High"]
            .iloc[-6:-1]
            .max()
        )

        recent_low = (
            df["Low"]
            .iloc[-6:-1]
            .min()
        )

        # Sweep above high, close back below
        if (
            last["High"] > recent_high
            and last["Close"] < recent_high
        ):
            return {
                "event": "BUY_SIDE_LIQUIDITY_SWEEP",
                "level": float(recent_high),
            }

        # Sweep below low, close back above
        if (
            last["Low"] < recent_low
            and last["Close"] > recent_low
        ):
            return {
                "event": "SELL_SIDE_LIQUIDITY_SWEEP",
                "level": float(recent_low),
            }

        # Genuine close above
        if last["Close"] > recent_high:
            return {
                "event": "POTENTIAL_BULLISH_BOS",
                "level": float(recent_high),
            }

        # Genuine close below
        if last["Close"] < recent_low:
            return {
                "event": "POTENTIAL_BEARISH_BOS",
                "level": float(recent_low),
            }

        return {
            "event": "NO_CLEAR_LIQUIDITY_EVENT"
        }

    except Exception:
        logger.exception(
            "Liquidity detection error."
        )

        return {
            "event": "UNKNOWN"
        }


# ============================================================
# 13. FAIR VALUE GAP
# ============================================================

def detect_fvg(df):

    results = []

    try:
        start = max(2, len(df) - 30)

        for i in range(
            start,
            len(df)
        ):

            c1_high = df[
                "High"
            ].iloc[i - 2]

            c1_low = df[
                "Low"
            ].iloc[i - 2]

            c3_high = df[
                "High"
            ].iloc[i]

            c3_low = df[
                "Low"
            ].iloc[i]

            # Bullish FVG
            if c3_low > c1_high:

                results.append({
                    "type": "BULLISH_FVG",
                    "low": float(c1_high),
                    "high": float(c3_low),
                    "index": i,
                })

            # Bearish FVG
            if c3_high < c1_low:

                results.append({
                    "type": "BEARISH_FVG",
                    "low": float(c3_high),
                    "high": float(c1_low),
                    "index": i,
                })

        return results[-5:]

    except Exception:
        logger.exception(
            "FVG detection error."
        )
        return []


# ============================================================
# 14. ORDER BLOCK RESEARCH
# ============================================================

def detect_order_blocks(df):

    results = []

    try:
        for i in range(
            max(1, len(df) - 30),
            len(df)
        ):

            previous = df.iloc[i - 1]
            current = df.iloc[i]

            # Previous bearish candle,
            # current bullish displacement
            if (
                previous["Close"]
                < previous["Open"]
                and current["Close"]
                > previous["High"]
            ):
                results.append({
                    "type": "BULLISH_OB",
                    "low": float(previous["Low"]),
                    "high": float(previous["High"]),
                    "index": i - 1,
                })

            # Previous bullish candle,
            # current bearish displacement
            elif (
                previous["Close"]
                > previous["Open"]
                and current["Close"]
                < previous["Low"]
            ):
                results.append({
                    "type": "BEARISH_OB",
                    "low": float(previous["Low"]),
                    "high": float(previous["High"]),
                    "index": i - 1,
                })

        return results[-5:]

    except Exception:
        logger.exception(
            "Order block error."
        )
        return []


# ============================================================
# 15. PREMIUM / DISCOUNT
# ============================================================

def premium_discount_analysis(df):

    try:
        structure = determine_structure(df)

        high = structure.get(
            "recent_swing_high"
        )

        low = structure.get(
            "recent_swing_low"
        )

        if high is None or low is None:
            return {
                "zone": "UNKNOWN",
                "midpoint": None,
            }

        if high <= low:
            return {
                "zone": "UNKNOWN",
                "midpoint": None,
            }

        midpoint = (
            high + low
        ) / 2

        price = float(
            df["Close"].iloc[-1]
        )

        if price > midpoint:
            zone = "PREMIUM"
        elif price < midpoint:
            zone = "DISCOUNT"
        else:
            zone = "EQUILIBRIUM"

        return {
            "zone": zone,
            "midpoint": float(midpoint),
            "swing_high": float(high),
            "swing_low": float(low),
            "price": price,
        }

    except Exception:
        logger.exception(
            "Premium/Discount error."
        )

        return {
            "zone": "UNKNOWN"
        }


# ============================================================
# 16. TECHNICAL SUMMARY
# ============================================================

def technical_summary(df):

    try:
        row = df.iloc[-1]

        return {
            "price": float(
                row["Close"]
            ),

            "EMA20": float(
                row["EMA_20"]
            ),

            "EMA50": float(
                row["EMA_50"]
            ),

            "EMA200": (
                float(row["EMA_200"])
                if not pd.isna(
                    row["EMA_200"]
                )
                else None
            ),

            "RSI": float(
                row["RSI"]
            ),

            "MACD": float(
                row["MACD"]
            ),

            "MACD_SIGNAL": float(
                row["MACD_SIGNAL"]
            ),

            "ATR": float(
                row["ATR"]
            ),

            "ATR_PCT": float(
                row["ATR_PCT"]
            ),

            "VOLATILITY": float(
                row["VOLATILITY"]
            )
            if not pd.isna(
                row["VOLATILITY"]
            )
            else None,

            "regime": determine_market_regime(
                df
            ),
        }

    except Exception as e:
        logger.exception(
            "Technical summary error."
        )
        return {
            "error": str(e)
        }


# ============================================================
# 17. MULTI-TIMEFRAME ANALYSIS
# ============================================================

def analyze_timeframe(
    symbol,
    period,
    interval
):

    df, message = get_market_data(
        symbol,
        period,
        interval
    )

    if df is None:
        return {
            "status": "DATA_UNAVAILABLE",
            "message": message,
        }

    df = calculate_indicators(df)

    df = detect_swing_points(df)

    tech = technical_summary(df)

    structure = determine_structure(df)

    liquidity = detect_liquidity_event(
        df
    )

    pd_zone = premium_discount_analysis(
        df
    )

    fvg = detect_fvg(df)

    ob = detect_order_blocks(df)

    return {
        "status": "OK",
        "data_points": len(df),
        "technical": tech,
        "structure": structure,
        "liquidity": liquidity,
        "premium_discount": pd_zone,
        "fvg": fvg,
        "order_blocks": ob,
        "last_timestamp": str(
            df.index[-1]
        ),
    }


def analyze_multi_timeframe(symbol):

    results = {}

    for tf, config in TIMEFRAMES.items():

        period, interval = config

        # 2h is built by resampling 1h data
        if tf == "2h":

            df, message = get_market_data(
                symbol,
                period="3mo",
                interval="1h"
            )

            if df is None:
                results[tf] = {
                    "status": "DATA_UNAVAILABLE",
                    "message": message,
                }
                continue

            try:
                df = df.resample(
                    "2h"
                ).agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }).dropna()

                valid, msg = validate_dataframe(
                    df
                )

                if not valid:
                    results[tf] = {
                        "status": "DATA_UNAVAILABLE",
                        "message": msg,
                    }
                    continue

                df = calculate_indicators(
                    df
                )

                df = detect_swing_points(
                    df
                )

                results[tf] = {
                    "status": "OK",
                    "data_points": len(df),
                    "technical": technical_summary(
                        df
                    ),
                    "structure": determine_structure(
                        df
                    ),
                    "liquidity": detect_liquidity_event(
                        df
                    ),
                    "premium_discount": premium_discount_analysis(
                        df
                    ),
                }

            except Exception as e:
                results[tf] = {
                    "status": "ERROR",
                    "message": str(e),
                }

        else:

            results[tf] = analyze_timeframe(
                symbol,
                period,
                interval
            )

    return results


# ============================================================
# 18. NEWS ENGINE
# ============================================================

def get_news(
    query="forex"
):

    if not ALPHA_VANTAGE_API_KEY:
        return {
            "status": "UNAVAILABLE",
            "message": (
                "ALPHA_VANTAGE_API_KEY "
                "is not configured."
            ),
            "items": [],
        }

    try:

        url = (
            "https://www.alphavantage.co/"
            "query"
        )

        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": ALPHA_VANTAGE_API_KEY,
            "limit": 10,
        }

        if query:
            params["tickers"] = query

        response = requests.get(
            url,
            params=params,
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        feed = data.get(
            "feed",
            []
        )

        items = []

        for item in feed[:10]:

            items.append({
                "title": item.get(
                    "title",
                    "Untitled"
                ),
                "source": item.get(
                    "source",
                    "Unknown"
                ),
                "published": item.get(
                    "time_published",
                    "Unknown"
                ),
                "url": item.get(
                    "url",
                    ""
                ),
            })

        return {
            "status": "OK",
            "items": items,
        }

    except Exception as e:

        logger.exception(
            "News API error."
        )

        return {
            "status": "ERROR",
            "message": str(e),
            "items": [],
        }


# ============================================================
# 19. SESSION FILTER
# ============================================================

def session_filter():

    hour = datetime.now(
        timezone.utc
    ).hour

    if 7 <= hour < 11:
        return {
            "session": "LONDON",
            "state": "ACTIVE",
        }

    if 12 <= hour < 16:
        return {
            "session": "NEW_YORK",
            "state": "ACTIVE",
        }

    if 0 <= hour < 7:
        return {
            "session": "ASIA",
            "state": "ACTIVE",
        }

    if 11 <= hour < 12:
        return {
            "session": "LONDON/NY_TRANSITION",
            "state": "WATCH",
        }

    if 16 <= hour < 17:
        return {
            "session": "NY_LATE",
            "state": "WATCH",
        }

    return {
        "session": "LOW_ACTIVITY_WINDOW",
        "state": "WAIT",
    }


# ============================================================
# 20. CONFLUENCE
# ============================================================

def build_confluence(
    technical,
    structure,
    liquidity,
    pd_zone,
    fvg,
    order_blocks
):

    score = 0
    evidence = []
    conflicts = []

    regime = technical.get(
        "regime",
        ""
    )

    rsi = technical.get(
        "RSI",
        50
    )

    ema20 = technical.get(
        "EMA20"
    )

    ema50 = technical.get(
        "EMA50"
    )

    # Trend
    if (
        ema20 is not None
        and ema50 is not None
    ):

        if ema20 > ema50:
            score += 15
            evidence.append(
                "EMA20 above EMA50"
            )

        elif ema20 < ema50:
            score += 15
            evidence.append(
                "EMA20 below EMA50"
            )

    # RSI
    if rsi >= 55:
        score += 10
        evidence.append(
            "RSI bullish zone"
        )

    elif rsi <= 45:
        score += 10
        evidence.append(
            "RSI bearish zone"
        )

    else:
        evidence.append(
            "RSI neutral"
        )

    # Structure
    structure_name = structure.get(
        "structure",
        "UNKNOWN"
    )

    if "HH" in structure_name:
        score += 15
        evidence.append(
            "Higher-high structure evidence"
        )

    if "HL" in structure_name:
        score += 10
        evidence.append(
            "Higher-low structure evidence"
        )

    if "LH" in structure_name:
        score += 15
        evidence.append(
            "Lower-high structure evidence"
        )

    if "LL" in structure_name:
        score += 10
        evidence.append(
            "Lower-low structure evidence"
        )

    # Liquidity
    liquidity_event = liquidity.get(
        "event",
        ""
    )

    if "BOS" in liquidity_event:
        score += 15
        evidence.append(
            liquidity_event
        )

    elif "SWEEP" in liquidity_event:
        score += 10
        evidence.append(
            liquidity_event
        )

    # Premium / Discount
    zone = pd_zone.get(
        "zone",
        "UNKNOWN"
    )

    if zone == "DISCOUNT":
        evidence.append(
            "Price in discount"
        )

    elif zone == "PREMIUM":
        evidence.append(
            "Price in premium"
        )

    elif zone == "EQUILIBRIUM":
        evidence.append(
            "Price near equilibrium"
        )

    # FVG / OB
    if fvg:
        score += 5
        evidence.append(
            f"{len(fvg)} recent FVG zone(s)"
        )

    if order_blocks:
        score += 5
        evidence.append(
            f"{len(order_blocks)} recent OB zone(s)"
        )

    # Regime
    if (
        "TRENDING"
        in regime
    ):
        score += 10
        evidence.append(
            "Trending regime"
        )

    elif (
        "SIDEWAYS"
        in regime
    ):
        conflicts.append(
            "Sideways regime"
        )

    score = min(
        100,
        max(0, score)
    )

    if score >= 70:
        status = "STRONG_RESEARCH_ALIGNMENT"

    elif score >= 50:
        status = "MODERATE_RESEARCH_ALIGNMENT"

    elif score >= 30:
        status = "WEAK_RESEARCH_ALIGNMENT"

    else:
        status = "INSUFFICIENT_EVIDENCE"

    return {
        "score": score,
        "status": status,
        "evidence": evidence,
        "conflicts": conflicts,
    }


# ============================================================
# 21. RESEARCH TP / SL
# ============================================================

def calculate_research_tp_sl(
    df,
    bias
):

    try:

        row = df.iloc[-1]

        entry = float(
            row["Close"]
        )

        atr = float(
            row["ATR"]
        )

        if not math.isfinite(
            atr
        ) or atr <= 0:

            return {
                "status": "UNAVAILABLE"
            }

        recent_high = float(
            df["High"]
            .iloc[-10:-1]
            .max()
        )

        recent_low = float(
            df["Low"]
            .iloc[-10:-1]
            .min()
        )

        if bias == "BULLISH":

            structure_sl = (
                recent_low - 0.25 * atr
            )

            sl = structure_sl

            risk = entry - sl

            if risk <= 0:
                return {
                    "status": "UNAVAILABLE"
                }

            tp15 = (
                entry + risk * 1.5
            )

            tp20 = (
                entry + risk * 2
            )

            tp30 = (
                entry + risk * 3
            )

        elif bias == "BEARISH":

            structure_sl = (
                recent_high + 0.25 * atr
            )

            sl = structure_sl

            risk = sl - entry

            if risk <= 0:
                return {
                    "status": "UNAVAILABLE"
                }

            tp15 = (
                entry - risk * 1.5
            )

            tp20 = (
                entry - risk * 2
            )

            tp30 = (
                entry - risk * 3
            )

        else:

            return {
                "status": "NO_BIAS"
            }

        return {
            "status": "RESEARCH_ONLY",
            "entry_reference": entry,
            "sl_reference": sl,
            "risk_distance": risk,
            "tp_1_5R": tp15,
            "tp_2R": tp20,
            "tp_3R": tp30,
        }

    except Exception as e:

        logger.exception(
            "TP/SL research error."
        )

        return {
            "status": "ERROR",
            "message": str(e),
        }


# ============================================================
# 22. FULL RESEARCH PIPELINE
# ============================================================

def run_market_research(
    symbol
):

    normalized, yf_symbol = resolve_symbol(
        symbol
    )

    if not normalized:

        return {
            "status": "INVALID_SYMBOL",
            "message": (
                f"'{symbol}' is not "
                "supported."
            ),
        }

    # Main timeframe
    df, message = get_market_data(
        normalized,
        period="1mo",
        interval="1h"
    )

    if df is None:

        return {
            "status": "DATA_UNAVAILABLE",
            "symbol": normalized,
            "message": message,
        }

    df = calculate_indicators(
        df
    )

    df = detect_swing_points(
        df
    )

    technical = technical_summary(
        df
    )

    structure = determine_structure(
        df
    )

    liquidity = detect_liquidity_event(
        df
    )

    pd_zone = premium_discount_analysis(
        df
    )

    fvg = detect_fvg(
        df
    )

    order_blocks = detect_order_blocks(
        df
    )

    # Determine research bias
    bias = "NEUTRAL"

    regime = technical.get(
        "regime",
        ""
    )

    if "BULLISH" in regime:
        bias = "BULLISH"

    elif "BEARISH" in regime:
        bias = "BEARISH"

    # Session
    session = session_filter()

    # News
    news = get_news(
        "FOREX"
    )

    # Confluence
    confluence = build_confluence(
        technical,
        structure,
        liquidity,
        pd_zone,
        fvg,
        order_blocks,
    )

    # Research TP/SL
    research_levels = (
        calculate_research_tp_sl(
            df,
            bias
        )
    )

    # Protection / WAIT logic
    wait_reasons = []

    if session["state"] == "WAIT":
        wait_reasons.append(
            "Low-activity session window"
        )

    if (
        confluence["score"] < 50
    ):
        wait_reasons.append(
            "Insufficient confluence evidence"
        )

    if (
        technical.get("regime")
        == "UNKNOWN_REGIME"
    ):
        wait_reasons.append(
            "Unknown market regime"
        )

    if (
        news["status"] != "OK"
    ):
        wait_reasons.append(
            "News data unavailable"
        )

    if wait_reasons:
        final_state = "WAIT"
    else:
        final_state = (
            "RESEARCH_SETUP_DETECTED"
        )

    # Multi-timeframe
    mtf = analyze_multi_timeframe(
        normalized
    )

    return {
        "status": "OK",
        "symbol": normalized,
        "display_name": DISPLAY_NAMES.get(
            normalized,
            normalized
        ),
        "source": yf_symbol,
        "main_data_points": len(df),
        "last_timestamp": str(
            df.index[-1]
        ),
        "technical": technical,
        "structure": structure,
        "liquidity": liquidity,
        "premium_discount": pd_zone,
        "fvg": fvg,
        "order_blocks": order_blocks,
        "bias": bias,
        "session": session,
        "news": news,
        "confluence": confluence,
        "research_levels": research_levels,
        "multi_timeframe": mtf,
        "final_state": final_state,
        "wait_reasons": wait_reasons,
        "mode": "RESEARCH/PAPER",
    }


# ============================================================
# 23. FORMAT REPORT
# ============================================================

def format_research_report(
    result
):

    if result["status"] != "OK":

        return (
            "⚠️ MARKET RESEARCH\n\n"
            f"Status: {result['status']}\n"
            f"Reason: {result.get('message', 'Unknown')}\n\n"
            "আমি কোনো data তৈরি করে "
            "দিচ্ছি না।"
        )

    tech = result["technical"]
    structure = result["structure"]
    liquidity = result["liquidity"]
    pd_zone = result[
        "premium_discount"
    ]
    confluence = result[
        "confluence"
    ]
    session = result["session"]
    levels = result[
        "research_levels"
    ]

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"📊 {result['display_name']}"
    )

    lines.append(
        "🔬 MARKET RESEARCH REPORT"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"Mode: {result['mode']}"
    )

    lines.append(
        f"Data Source: {result['source']}"
    )

    lines.append(
        f"Data Points: {result['main_data_points']}"
    )

    lines.append(
        f"Last Candle: {result['last_timestamp']}"
    )

    lines.append("")

    # Price
    lines.append(
        f"💰 Price: {tech.get('price', 'N/A')}"
    )

    # Technical
    lines.append("")
    lines.append("📈 TECHNICAL")

    lines.append(
        f"EMA20: {tech.get('EMA20', 'N/A')}"
    )

    lines.append(
        f"EMA50: {tech.get('EMA50', 'N/A')}"
    )

    lines.append(
        f"RSI: {tech.get('RSI', 'N/A'):.2f}"
    )

    lines.append(
        f"MACD: {tech.get('MACD', 'N/A'):.6f}"
    )

    lines.append(
        f"ATR: {tech.get('ATR', 'N/A'):.6f}"
    )

    lines.append(
        f"Regime: {tech.get('regime', 'N/A')}"
    )

    # Structure
    lines.append("")
    lines.append("🏗 STRUCTURE")

    lines.append(
        f"Structure: {structure.get('structure', 'N/A')}"
    )

    # Liquidity
    lines.append("")
    lines.append("💧 LIQUIDITY")

    lines.append(
        f"Event: {liquidity.get('event', 'N/A')}"
    )

    # Premium discount
    lines.append("")
    lines.append("⚖️ PREMIUM / DISCOUNT")

    lines.append(
        f"Zone: {pd_zone.get('zone', 'N/A')}"
    )

    # Session
    lines.append("")
    lines.append("🕒 SESSION")

    lines.append(
        f"{session['session']} | {session['state']}"
    )

    # Confluence
    lines.append("")
    lines.append("🧠 CONFLUENCE")

    lines.append(
        f"Score: {confluence['score']}/100"
    )

    lines.append(
        f"Status: {confluence['status']}"
    )

    # Bias
    lines.append("")
    lines.append(
        f"🔎 Research Bias: {result['bias']}"
    )

    # Research levels
    lines.append("")
    lines.append(
        "📐 RESEARCH LEVELS"
    )

    if levels.get("status") == "RESEARCH_ONLY":

        lines.append(
            f"Reference Entry: "
            f"{levels['entry_reference']}"
        )

        lines.append(
            f"Research SL: "
            f"{levels['sl_reference']}"
        )

        lines.append(
            f"Research TP 1.5R: "
            f"{levels['tp_1_5R']}"
        )

        lines.append(
            f"Research TP 2R: "
            f"{levels['tp_2R']}"
        )

        lines.append(
            f"Research TP 3R: "
            f"{levels['tp_3R']}"
        )

    else:

        lines.append(
            "Research levels unavailable."
        )

    # Final
    lines.append("")
    lines.append("🚦 FINAL STATE")

    if result["final_state"] == "WAIT":

        lines.append(
            "🟡 WAIT / INSUFFICIENT CONFIRMATION"
        )

        for reason in result[
            "wait_reasons"
        ]:

            lines.append(
                f"• {reason}"
            )

    else:

        lines.append(
            "🟢 RESEARCH SETUP DETECTED"
        )

    lines.append("")

    lines.append(
        "⚠️ এটি research/paper analysis। "
        "এটি guaranteed prediction বা "
        "real-money trade instruction নয়।"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    return "\n".join(lines)


# ============================================================
# 24. START MENU
# ============================================================

def main_keyboard():

    return InlineKeyboardMarkup([

        [
            InlineKeyboardButton(
                "📊 Watchlist",
                callback_data="watchlist"
            ),

            InlineKeyboardButton(
                "❤️ System Health",
                callback_data="health"
            ),
        ],

        [
            InlineKeyboardButton(
                "🔬 Analyze Guide",
                callback_data="analyze_guide"
            ),

            InlineKeyboardButton(
                "📰 News",
                callback_data="news"
            ),
        ],

        [
            InlineKeyboardButton(
                "🤖 AI Chat",
                callback_data="ai"
            ),
        ],
    ])


# ============================================================
# 25. /START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        "👋 আসসালামু আলাইকুম!\n\n"
        "🤖 Market Research Bot অনলাইনে আছে।\n\n"

        "📌 Supported assets:\n"
        "EURUSD\n"
        "GBPUSD\n"
        "AUDUSD\n"
        "NZDUSD\n"
        "USDJPY\n"
        "USDCHF\n"
        "USDCAD\n"
        "GOLD / XAUUSD\n"
        "BTCUSD\n"
        "ETHUSD\n\n"

        "🔬 সরাসরি asset name লিখতে পারো:\n"
        "EURUSD\n"
        "AUDUSD\n"
        "GOLD\n\n"

        "অথবা:\n"
        "/analyze EURUSD\n\n"

        "⚠️ Mode: Research / Paper Trading"
    )

    await update.message.reply_text(
        text,
        reply_markup=main_keyboard()
    )


# ============================================================
# 26. /HELP
# ============================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        "📚 COMMAND GUIDE\n\n"

        "/start — Main menu\n"
        "/help — Commands\n"
        "/price EURUSD — Price/data check\n"
        "/analyze EURUSD — Full research\n"
        "/news — News check\n"
        "/health — System health\n\n"

        "Direct asset:\n"
        "EURUSD\n"
        "AUDUSD\n"
        "GOLD\n"
        "BTCUSD\n\n"

        "সাধারণ কথাও লিখতে পারো। "
        "Bot চুপ থাকবে না।"
    )

    await update.message.reply_text(
        text
    )


# ============================================================
# 27. /HEALTH
# ============================================================

async def health_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        "❤️ SYSTEM HEALTH\n\n"

        f"Telegram: "
        f"{'READY' if TELEGRAM_BOT_TOKEN else 'MISSING'}\n"

        f"Gemini: "
        f"{'READY' if gemini_client else 'NOT CONFIGURED'}\n"

        f"News API: "
        f"{'READY' if ALPHA_VANTAGE_API_KEY else 'NOT CONFIGURED'}\n"

        f"Market Data: Yahoo Finance adapter\n"

        f"Mode: Research / Paper\n\n"

        "No fake market data is generated."
    )

    await update.message.reply_text(
        text
    )


# ============================================================
# 28. /PRICE
# ============================================================

async def price_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not context.args:

        await update.message.reply_text(
            "ব্যবহার:\n"
            "/price EURUSD\n"
            "/price GOLD\n"
            "/price BTCUSD"
        )

        return

    symbol = context.args[0]

    df, message = get_market_data(
        symbol,
        period="5d",
        interval="1h"
    )

    if df is None:

        await update.message.reply_text(
            "⚠️ PRICE DATA UNAVAILABLE\n\n"
            f"Symbol: {symbol}\n"
            f"Reason: {message}\n\n"
            "আমি কোনো fake price দেখাচ্ছি না।"
        )

        return

    last = df.iloc[-1]

    text = (
        "💰 PRICE / DATA CHECK\n\n"
        f"Symbol: {symbol.upper()}\n"
        f"Price: {float(last['Close'])}\n"
        f"Last candle: {df.index[-1]}\n"
        f"Candles: {len(df)}\n\n"
        "⚠️ এই data exact broker OTC feed "
        "নাও হতে পারে।"
    )

    await update.message.reply_text(
        text
    )


# ============================================================
# 29. /ANALYZE
# ============================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not context.args:

        await update.message.reply_text(
            "ব্যবহার:\n\n"
            "/analyze EURUSD\n"
            "/analyze AUDUSD\n"
            "/analyze GOLD"
        )

        return

    symbol = context.args[0]

    await update.message.reply_text(
        f"🔄 {symbol.upper()} research শুরু হয়েছে...\n"
        "Data → Technical → MTF → Structure → "
        "Liquidity → P/D → Confluence..."
    )

    try:

        result = run_market_research(
            symbol
        )

        report = format_research_report(
            result
        )

        await update.message.reply_text(
            report
        )

    except Exception as e:

        logger.exception(
            "Analyze command failed."
        )

        await update.message.reply_text(
            "❌ ANALYSIS ERROR\n\n"
            f"Reason: {str(e)}\n\n"
            "Bot চুপ করবে না; error এখানে দেখানো হলো।"
        )


# ============================================================
# 30. /NEWS
# ============================================================

async def news_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📰 News engine চালু করা হচ্ছে..."
    )

    result = get_news(
        "FOREX"
    )

    if result["status"] != "OK":

        await update.message.reply_text(
            "📰 NEWS STATUS\n\n"
            f"{result.get('message', 'Unavailable')}\n\n"
            "API unavailable হলে bot "
            "fake news তৈরি করবে না।"
        )

        return

    items = result["items"]

    if not items:

        await update.message.reply_text(
            "📰 কোনো news item পাওয়া যায়নি।"
        )

        return

    lines = [
        "📰 LATEST AVAILABLE NEWS\n"
    ]

    for i, item in enumerate(
        items[:8],
        start=1
    ):

        lines.append(
            f"{i}. {item['title']}\n"
            f"Source: {item['source']}\n"
            f"Published: {item['published']}\n"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# 31. GEMINI CHAT
# ============================================================

async def gemini_chat(
    user_text
):

    if gemini_client is None:

        return (
            "🤖 AI Chat এখন unavailable.\n\n"
            "GEMINI_API_KEY configure করা হয়নি "
            "অথবা Gemini client initialize হয়নি।"
        )

    prompt = f"""
You are the educational AI assistant inside a
market research and paper-trading application.

User message:
{user_text}

Rules:
- Explain clearly.
- Do not claim guaranteed future price movement.
- Do not fabricate live market data.
- If live data is unavailable, say so.
- Do not provide real-money execution instructions.
- Keep the answer educational and research-focused.
"""

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        text = getattr(
            response,
            "text",
            None
        )

        if text:
            return text

        return (
            "🤖 Gemini কোনো text response দেয়নি।"
        )

    except Exception as e:

        logger.exception(
            "Gemini error."
        )

        return (
            "🤖 AI ERROR\n\n"
            f"{str(e)}"
        )


# ============================================================
# 32. BUTTON HANDLER
# ============================================================

async def button_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    await query.answer()

    data = query.data

    if data == "watchlist":

        text = (
            "📊 WATCHLIST\n\n"
            + "\n".join(
                f"• {x}"
                for x in WATCHLIST
            )
            + "\n\n"
            "Example:\n"
            "EURUSD"
        )

        await query.edit_message_text(
            text
        )

    elif data == "health":

        text = (
            "❤️ SYSTEM HEALTH\n\n"
            f"Telegram: "
            f"{'READY' if TELEGRAM_BOT_TOKEN else 'MISSING'}\n"
            f"Gemini: "
            f"{'READY' if gemini_client else 'NOT CONFIGURED'}\n"
            f"News API: "
            f"{'READY' if ALPHA_VANTAGE_API_KEY else 'NOT CONFIGURED'}\n"
            "Market Data: Yahoo adapter\n"
            "Mode: Research/Paper"
        )

        await query.edit_message_text(
            text
        )

    elif data == "analyze_guide":

        text = (
            "🔬 ANALYZE\n\n"
            "লিখো:\n\n"
            "/analyze EURUSD\n"
            "/analyze AUDUSD\n"
            "/analyze GOLD\n\n"

            "Pipeline:\n"
            "Data validation\n"
            "Technical indicators\n"
            "Multi-timeframe\n"
            "Market structure\n"
            "Liquidity\n"
            "FVG\n"
            "Order Block research\n"
            "Premium/Discount\n"
            "Session\n"
            "News availability\n"
            "Confluence\n"
            "Research TP/SL\n"
            "Final WAIT/Research state"
        )

        await query.edit_message_text(
            text
        )

    elif data == "news":

        result = get_news(
            "FOREX"
        )

        if result["status"] != "OK":

            await query.edit_message_text(
                "📰 NEWS UNAVAILABLE\n\n"
                + result.get(
                    "message",
                    "Unknown error"
                )
            )

            return

        items = result["items"]

        if not items:

            await query.edit_message_text(
                "📰 No news data available."
            )

            return

        text = "📰 NEWS\n\n"

        for i, item in enumerate(
            items[:5],
            1
        ):

            text += (
                f"{i}. {item['title']}\n"
                f"{item['source']}\n\n"
            )

        await query.edit_message_text(
            text
        )

    elif data == "ai":

        await query.edit_message_text(
            "🤖 AI CHAT\n\n"
            "AI-কে প্রশ্ন করতে নিচে "
            "সাধারণ message পাঠাও।"
        )


# ============================================================
# 33. DIRECT ASSET DETECTOR
# ============================================================

def detect_asset_from_message(
    text
):

    normalized = normalize_symbol(
        text
    )

    if normalized in SYMBOL_MAP:
        return normalized

    return None


# ============================================================
# 34. GENERAL MESSAGE HANDLER
# ============================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:
        return

    text = (
        update.message.text
        or ""
    ).strip()

    if not text:

        await update.message.reply_text(
            "আমি empty message পেয়েছি। "
            "EURUSD / GOLD / Hi লিখে চেষ্টা করো।"
        )

        return

    # ----------------------------------------
    # 1. DIRECT ASSET
    # ----------------------------------------

    asset = detect_asset_from_message(
        text
    )

    if asset:

        await update.message.reply_text(
            f"🔎 {DISPLAY_NAMES.get(asset, asset)} "
            "detected.\n\n"
            "🔄 Research pipeline শুরু করছি..."
        )

        try:

            result = run_market_research(
                asset
            )

            report = format_research_report(
                result
            )

            await update.message.reply_text(
                report
            )

        except Exception as e:

            logger.exception(
                "Direct asset analysis failed."
            )

            await update.message.reply_text(
                "❌ Analysis error\n\n"
                f"{str(e)}"
            )

        return

    # ----------------------------------------
    # 2. NATURAL COMMAND-LIKE TEXT
    # ----------------------------------------

    lower = text.lower()

    if (
        "analyze" in lower
        or "analysis" in lower
        or "analyse" in lower
    ):

        found = None

        for key in SYMBOL_MAP:

            if key.lower() in lower:

                found = key
                break

        if found:

            await update.message.reply_text(
                f"🔎 {found} detected.\n"
                "Research শুরু করছি..."
            )

            try:

                result = run_market_research(
                    found
                )

                await update.message.reply_text(
                    format_research_report(
                        result
                    )
                )

            except Exception as e:

                logger.exception(
                    "Natural analysis failed."
                )

                await update.message.reply_text(
                    f"❌ Error: {str(e)}"
                )

            return

    # ----------------------------------------
    # 3. GREETING
    # ----------------------------------------

    greetings = {
        "hi",
        "hello",
        "hey",
        "হাই",
        "হ্যালো",
        "আসসালামু আলাইকুম",
        "salam",
    }

    if lower in greetings:

        await update.message.reply_text(
            "👋 Wa Alaikum Assalam!\n\n"
            "আমি অনলাইনে আছি। 😊\n\n"
            "তুমি EURUSD, AUDUSD, GOLD "
            "বা /analyze EURUSD লিখতে পারো।"
        )

        return

    # ----------------------------------------
    # 4. HELP KEYWORDS
    # ----------------------------------------

    if lower in {
        "help",
        "menu",
        "guide",
        "কী করতে পারি",
        "কি করতে পারি",
    }:

        await update.message.reply_text(
            "📚 তুমি করতে পারো:\n\n"
            "• EURUSD\n"
            "• AUDUSD\n"
            "• GOLD\n"
            "• /analyze EURUSD\n"
            "• /price GOLD\n"
            "• /news\n"
            "• /health\n"
            "• সাধারণ প্রশ্ন"
        )

        return

    # ----------------------------------------
    # 5. GEMINI FALLBACK
    # ----------------------------------------

    reply = await gemini_chat(
        text
    )

    await update.message.reply_text(
        reply
    )


# ============================================================
# 35. GLOBAL ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    logger.exception(
        "Unhandled Telegram error:",
        exc_info=context.error
    )

    try:

        if isinstance(
            update,
            Update
        ) and update.effective_message:

            await update.effective_message.reply_text(
                "⚠️ Bot-এর ভিতরে একটি error হয়েছে।\n\n"
                f"{context.error}\n\n"
                "Bot silent failure না করে error দেখিয়েছে।"
            )

    except Exception:
        logger.exception(
            "Could not send error message."
        )


# ============================================================
# 36. BOT COMMAND SETUP
# ============================================================

async def post_init(
    application: Application
):

    try:

        await application.bot.set_my_commands([
            ("start", "Open main menu"),
            ("help", "Show help"),
            ("price", "Check market data"),
            ("analyze", "Research an asset"),
            ("news", "Check available news"),
            ("health", "System health"),
        ])

        logger.info(
            "Bot commands registered."
        )

    except Exception:
        logger.exception(
            "Could not register commands."
        )


# ============================================================
# 37. MAIN
# ============================================================

def main():

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing."
        )

    # Start Render health server
    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )

    flask_thread.start()

    logger.info(
        "Flask health server started."
    )

    # Telegram application
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Commands
    application.add_handler(
        CommandHandler(
            "start",
            start_command
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            help_command
        )
    )

    application.add_handler(
        CommandHandler(
            "health",
            health_command
        )
    )

    application.add_handler(
        CommandHandler(
            "price",
            price_command
        )
    )

    application.add_handler(
        CommandHandler(
            "analyze",
            analyze_command
        )
    )

    application.add_handler(
        CommandHandler(
            "news",
            news_command
        )
    )

    # Buttons
    application.add_handler(
        CallbackQueryHandler(
            button_handler
        )
    )

    # ALL TEXT
    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            handle_message
        )
    )

    # Errors
    application.add_error_handler(
        error_handler
    )

    logger.info(
        "================================"
    )

    logger.info(
        "MARKET RESEARCH BOT STARTING"
    )

    logger.info(
        "================================"
    )

    application.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# 38. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

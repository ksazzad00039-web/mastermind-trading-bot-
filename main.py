# ============================================================
# MASTER MARKET RESEARCH BOT
# SINGLE-FILE VERSION
# Research / Backtesting / Paper Trading Only
# ============================================================

import os
import re
import math
import time
import json
import sqlite3
import logging
import threading
from datetime import datetime, timezone, timedelta

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
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ------------------------------------------------------------
# GEMINI
# ------------------------------------------------------------

try:
    from google import genai
except Exception:
    genai = None


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", ""
).strip()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY", ""
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash"
).strip()

ALPHA_VANTAGE_API_KEY = os.getenv(
    "ALPHA_VANTAGE_API_KEY",
    ""
).strip()

PORT = int(os.getenv("PORT", "10000"))

DB_FILE = os.getenv(
    "DB_FILE",
    "market_research.db"
)

SCAN_INTERVAL = int(
    os.getenv("SCAN_INTERVAL", "300")
)

MIN_CONFLUENCE = 60


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )
)

logger = logging.getLogger("MASTER_MARKET_BOT")


# ============================================================
# FLASK HEALTH SERVER
# ============================================================

flask_app = Flask(__name__)


@flask_app.route("/")
def home():
    return jsonify({
        "status": "online",
        "bot": "Master Market Research Bot",
        "mode": "research/paper",
        "time_utc": datetime.now(
            timezone.utc
        ).isoformat()
    })


@flask_app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "telegram": bool(TELEGRAM_BOT_TOKEN),
        "gemini": bool(GEMINI_API_KEY),
        "news_api": bool(ALPHA_VANTAGE_API_KEY),
        "database": DB_FILE,
        "time_utc": datetime.now(
            timezone.utc
        ).isoformat()
    })


def run_flask():

    try:

        flask_app.run(
            host="0.0.0.0",
            port=PORT,
            use_reloader=False
        )

    except Exception:

        logger.exception(
            "Flask server stopped."
        )


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = None

if GEMINI_API_KEY and genai is not None:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        logger.info(
            "Gemini initialized."
        )

    except Exception:

        logger.exception(
            "Gemini initialization failed."
        )


# ============================================================
# FIVE-MARKET UNIVERSE
# ============================================================

MARKETS = {

    "EURUSD": {
        "name": "EUR/USD",
        "yf": "EURUSD=X",
        "news": "FOREX"
    },

    "BTCUSD": {
        "name": "Bitcoin / BTCUSD",
        "yf": "BTC-USD",
        "news": "CRYPTO:BTC"
    },

    "GOLD": {
        "name": "Gold / XAUUSD",
        "yf": "GC=F",
        "news": "FOREX"
    },

    "ETHUSD": {
        "name": "Ethereum / ETHUSD",
        "yf": "ETH-USD",
        "news": "CRYPTO:ETH"
    },

    "USDJPY": {
        "name": "USD/JPY",
        "yf": "JPY=X",
        "news": "FOREX"
    }

}


ALIASES = {

    "EUR/USD": "EURUSD",
    "EUR": "EURUSD",

    "BTC": "BTCUSD",
    "BTC/USD": "BTCUSD",
    "BITCOIN": "BTCUSD",

    "XAUUSD": "GOLD",
    "XAU/USD": "GOLD",
    "XAU": "GOLD",
    "GOLD": "GOLD",

    "ETH": "ETHUSD",
    "ETH/USD": "ETHUSD",
    "ETHEREUM": "ETHUSD",

    "USD/JPY": "USDJPY",
    "JPY": "USDJPY"

}


# ============================================================
# TIMEFRAMES
# ============================================================

TIMEFRAMES = {

    "5m": {
        "period": "5d",
        "interval": "5m"
    },

    "15m": {
        "period": "5d",
        "interval": "15m"
    },

    "30m": {
        "period": "1mo",
        "interval": "30m"
    },

    "1h": {
        "period": "1mo",
        "interval": "1h"
    },

    "4h": {
        "period": "6mo",
        "interval": "1h"
    },

    "1D": {
        "period": "2y",
        "interval": "1d"
    }

}


# ============================================================
# DATABASE
# ============================================================

def init_database():

    try:

        conn = sqlite3.connect(
            DB_FILE
        )

        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                bias TEXT,
                entry REAL,
                sl REAL,
                tp REAL,
                result TEXT,
                created_at TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS research_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                score REAL,
                state TEXT,
                regime TEXT,
                created_at TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS scan_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                score REAL,
                state TEXT,
                created_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    except Exception:

        logger.exception(
            "Database initialization error."
        )


init_database()


# ============================================================
# SYMBOL NORMALIZATION
# ============================================================

def normalize_symbol(text):

    if not text:
        return None

    value = text.upper().strip()

    value = re.sub(
        r"\s+",
        "",
        value
    )

    if value in MARKETS:
        return value

    if value in ALIASES:
        return ALIASES[value]

    return None


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataframe(df):

    if df is None:
        return False, "No data returned."

    if df.empty:
        return False, "Empty market data."

    required = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    missing = [
        x for x in required
        if x not in df.columns
    ]

    if missing:

        return False, (
            "Missing columns: "
            + str(missing)
        )

    df = df.copy()

    for col in required:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df.dropna(
        subset=required,
        inplace=True
    )

    if len(df) < 60:

        return False, (
            f"Only {len(df)} candles. "
            "Need at least 60."
        )

    if df.index.has_duplicates:

        df = df[
            ~df.index.duplicated(
                keep="last"
            )
        ]

    return True, "OK"


# ============================================================
# MARKET DATA ENGINE
# ============================================================

def get_market_data(
    symbol,
    period="1mo",
    interval="1h"
):

    symbol = normalize_symbol(
        symbol
    )

    if not symbol:

        return None, "Unsupported market."

    yf_symbol = MARKETS[
        symbol
    ]["yf"]

    try:

        logger.info(
            "DATA %s -> %s | %s | %s",
            symbol,
            yf_symbol,
            period,
            interval
        )

        df = yf.download(
            yf_symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        valid, message = (
            validate_dataframe(df)
        )

        if not valid:

            return None, message

        return df, "OK"

    except Exception as e:

        logger.exception(
            "Market data error."
        )

        return None, str(e)


# ============================================================
# TECHNICAL ENGINE
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # SMA
    # --------------------------------------------------------

    df["SMA20"] = close.rolling(
        20
    ).mean()

    df["SMA50"] = close.rolling(
        50
    ).mean()

    df["SMA200"] = close.rolling(
        200
    ).mean()

    # --------------------------------------------------------
    # RSI - Wilder style
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["RSI"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_HIST"] = (
        df["MACD"] -
        df["MACD_SIGNAL"]
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    bb_mid = close.rolling(
        20
    ).mean()

    bb_std = close.rolling(
        20
    ).std()

    df["BB_MID"] = bb_mid

    df["BB_UPPER"] = (
        bb_mid +
        2 * bb_std
    )

    df["BB_LOWER"] = (
        bb_mid -
        2 * bb_std
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    prev_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high -
        prev_close
    ).abs()

    tr3 = (
        low -
        prev_close
    ).abs()

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    df["TR"] = tr

    df["ATR"] = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    df["ATR_PCT"] = (
        df["ATR"] /
        close *
        100
    )

    # --------------------------------------------------------
    # Returns / Volatility
    # --------------------------------------------------------

    df["RETURN"] = (
        close.pct_change()
    )

    df["VOLATILITY"] = (
        df["RETURN"]
        .rolling(20)
        .std() *
        np.sqrt(20) *
        100
    )

    # --------------------------------------------------------
    # Candle Anatomy
    # --------------------------------------------------------

    df["BODY"] = (
        df["Close"] -
        df["Open"]
    ).abs()

    df["RANGE"] = (
        df["High"] -
        df["Low"]
    )

    df["UPPER_WICK"] = (
        df["High"] -
        df[
            ["Open", "Close"]
        ].max(axis=1)
    )

    df["LOWER_WICK"] = (
        df[
            ["Open", "Close"]
        ].min(axis=1) -
        df["Low"]
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if "Volume" in df.columns:

        df["VOLUME_SMA20"] = (
            df["Volume"]
            .rolling(20)
            .mean()
        )

    return df


# ============================================================
# MARKET REGIME
# ============================================================

def determine_regime(df):

    try:

        row = df.iloc[-1]

        ema20 = row["EMA20"]
        ema50 = row["EMA50"]
        rsi = row["RSI"]
        atr_pct = row["ATR_PCT"]

        if any(
            pd.isna(x)
            for x in [
                ema20,
                ema50,
                rsi,
                atr_pct
            ]
        ):

            return "UNKNOWN"

        recent_range = (
            df["High"]
            .tail(20)
            .max()
            -
            df["Low"]
            .tail(20)
            .min()
        )

        current_atr = (
            row["ATR"]
        )

        high_vol = (
            atr_pct >
            df["ATR_PCT"]
            .tail(100)
            .quantile(
                0.80
            )
            if len(df) >= 100
            else False
        )

        if (
            ema20 > ema50
            and rsi >= 55
        ):

            if high_vol:
                return "BULLISH_TREND_HIGH_VOL"

            return "BULLISH_TREND"

        if (
            ema20 < ema50
            and rsi <= 45
        ):

            if high_vol:
                return "BEARISH_TREND_HIGH_VOL"

            return "BEARISH_TREND"

        if (
            current_atr > 0
            and recent_range <
            current_atr * 4
        ):

            return "CONSOLIDATION"

        return "RANGE"

    except Exception:

        logger.exception(
            "Regime error."
        )

        return "UNKNOWN"


# ============================================================
# SWING ENGINE
# ============================================================

def detect_swings(
    df,
    lookback=3
):

    df = df.copy()

    df["SWING_HIGH"] = False
    df["SWING_LOW"] = False

    for i in range(
        lookback,
        len(df) - lookback
    ):

        current_high = (
            df["High"].iloc[i]
        )

        current_low = (
            df["Low"].iloc[i]
        )

        left_high = (
            df["High"]
            .iloc[
                i - lookback:i
            ]
            .max()
        )

        right_high = (
            df["High"]
            .iloc[
                i + 1:
                i + 1 + lookback
            ]
            .max()
        )

        left_low = (
            df["Low"]
            .iloc[
                i - lookback:i
            ]
            .min()
        )

        right_low = (
            df["Low"]
            .iloc[
                i + 1:
                i + 1 + lookback
            ]
            .min()
        )

        if (
            current_high >
            left_high
            and
            current_high >
            right_high
        ):

            df.loc[
                df.index[i],
                "SWING_HIGH"
            ] = True

        if (
            current_low <
            left_low
            and
            current_low <
            right_low
        ):

            df.loc[
                df.index[i],
                "SWING_LOW"
            ] = True

    return df


# ============================================================
# MARKET STRUCTURE
# ============================================================

def market_structure(df):

    try:

        highs = df[
            df["SWING_HIGH"]
        ]["High"].tolist()

        lows = df[
            df["SWING_LOW"]
        ]["Low"].tolist()

        structure = "UNKNOWN"

        if len(highs) >= 2:

            if highs[-1] > highs[-2]:
                structure = "HH"

            else:
                structure = "LH"

        if len(lows) >= 2:

            if lows[-1] > lows[-2]:

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

            "recent_high": (
                highs[-1]
                if highs
                else None
            ),

            "previous_high": (
                highs[-2]
                if len(highs) >= 2
                else None
            ),

            "recent_low": (
                lows[-1]
                if lows
                else None
            ),

            "previous_low": (
                lows[-2]
                if len(lows) >= 2
                else None
            )

        }

    except Exception:

        logger.exception(
            "Structure error."
        )

        return {
            "structure": "UNKNOWN"
        }


# ============================================================
# BOS / LIQUIDITY SWEEP
# ============================================================

def liquidity_event(df):

    try:

        if len(df) < 10:

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

        # Buy-side sweep

        if (
            last["High"] >
            recent_high
            and
            last["Close"] <
            recent_high
        ):

            return {

                "event":
                "BUY_SIDE_LIQUIDITY_SWEEP",

                "level":
                float(recent_high)

            }

        # Sell-side sweep

        if (
            last["Low"] <
            recent_low
            and
            last["Close"] >
            recent_low
        ):

            return {

                "event":
                "SELL_SIDE_LIQUIDITY_SWEEP",

                "level":
                float(recent_low)

            }

        # Bullish BOS candidate

        if (
            last["Close"] >
            recent_high
        ):

            return {

                "event":
                "POTENTIAL_BULLISH_BOS",

                "level":
                float(recent_high)

            }

        # Bearish BOS candidate

        if (
            last["Close"] <
            recent_low
        ):

            return {

                "event":
                "POTENTIAL_BEARISH_BOS",

                "level":
                float(recent_low)

            }

        return {
            "event":
            "NO_CLEAR_EVENT"
        }

    except Exception:

        logger.exception(
            "Liquidity error."
        )

        return {
            "event": "UNKNOWN"
        }


# ============================================================
# FVG ENGINE
# ============================================================

def detect_fvg(df):

    results = []

    start = max(
        2,
        len(df) - 40
    )

    for i in range(
        start,
        len(df)
    ):

        candle1_high = (
            df["High"]
            .iloc[i - 2]
        )

        candle1_low = (
            df["Low"]
            .iloc[i - 2]
        )

        candle3_high = (
            df["High"]
            .iloc[i]
        )

        candle3_low = (
            df["Low"]
            .iloc[i]
        )

        # Bullish FVG

        if candle3_low > candle1_high:

            results.append({

                "type":
                "BULLISH_FVG",

                "low":
                float(candle1_high),

                "high":
                float(candle3_low),

                "index":
                i

            })

        # Bearish FVG

        if candle3_high < candle1_low:

            results.append({

                "type":
                "BEARISH_FVG",

                "low":
                float(candle3_high),

                "high":
                float(candle1_low),

                "index":
                i

            })

    return results[-5:]


# ============================================================
# ORDER BLOCK RESEARCH
# ============================================================

def detect_order_blocks(df):

    results = []

    start = max(
        1,
        len(df) - 40
    )

    for i in range(
        start,
        len(df)
    ):

        previous = df.iloc[i - 1]
        current = df.iloc[i]

        # Bullish OB

        if (
            previous["Close"] <
            previous["Open"]
            and
            current["Close"] >
            previous["High"]
        ):

            results.append({

                "type":
                "BULLISH_OB",

                "low":
                float(previous["Low"]),

                "high":
                float(previous["High"]),

                "index":
                i - 1

            })

        # Bearish OB

        elif (
            previous["Close"] >
            previous["Open"]
            and
            current["Close"] <
            previous["Low"]
        ):

            results.append({

                "type":
                "BEARISH_OB",

                "low":
                float(previous["Low"]),

                "high":
                float(previous["High"]),

                "index":
                i - 1

            })

    return results[-5:]


# ============================================================
# PREMIUM / DISCOUNT
# ============================================================

def premium_discount(df):

    structure = market_structure(
        df
    )

    high = structure.get(
        "recent_high"
    )

    low = structure.get(
        "recent_low"
    )

    if (
        high is None
        or low is None
        or high <= low
    ):

        return {
            "zone": "UNKNOWN"
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

        "midpoint":
        float(midpoint),

        "swing_high":
        float(high),

        "swing_low":
        float(low),

        "price":
        price

    }


# ============================================================
# CANDLE PRICE ACTION
# ============================================================

def candle_analysis(df):

    row = df.iloc[-1]

    body = float(
        row["BODY"]
    )

    candle_range = float(
        row["RANGE"]
    )

    upper = float(
        row["UPPER_WICK"]
    )

    lower = float(
        row["LOWER_WICK"]
    )

    if candle_range <= 0:

        return "UNKNOWN"

    body_ratio = (
        body /
        candle_range
    )

    if (
        body_ratio >= 0.70
        and
        row["Close"] >
        row["Open"]
    ):

        return "STRONG_BULLISH_CLOSE"

    if (
        body_ratio >= 0.70
        and
        row["Close"] <
        row["Open"]
    ):

        return "STRONG_BEARISH_CLOSE"

    if (
        lower >
        body * 1.5
    ):

        return "LOWER_WICK_REJECTION"

    if (
        upper >
        body * 1.5
    ):

        return "UPPER_WICK_REJECTION"

    return "NEUTRAL_CANDLE"


# ============================================================
# SESSION ENGINE
# ============================================================

def session_status():

    now = datetime.now(
        timezone.utc
    )

    hour = now.hour

    if 0 <= hour < 7:

        return {
            "session": "ASIA",
            "state": "ACTIVE"
        }

    if 7 <= hour < 12:

        return {
            "session": "LONDON",
            "state": "ACTIVE"
        }

    if 12 <= hour < 16:

        return {
            "session":
            "NEW_YORK",
            "state": "ACTIVE"
        }

    if 16 <= hour < 18:

        return {
            "session":
            "NY_LATE",
            "state": "WATCH"
        }

    return {

        "session":
        "LOW_ACTIVITY",

        "state":
        "WAIT"

    }


# ============================================================
# NEWS ENGINE
# ============================================================

def get_news(
    symbol=None
):

    if not ALPHA_VANTAGE_API_KEY:

        return {

            "status":
            "UNAVAILABLE",

            "items": []

        }

    try:

        params = {

            "function":
            "NEWS_SENTIMENT",

            "apikey":
            ALPHA_VANTAGE_API_KEY,

            "limit": 10

        }

        if symbol in [
            "BTCUSD",
            "ETHUSD"
        ]:

            ticker = (
                "CRYPTO:" +
                (
                    "BTC"
                    if symbol ==
                    "BTCUSD"
                    else
                    "ETH"
                )
            )

            params[
                "tickers"
            ] = ticker

        else:

            params[
                "topics"
            ] = "forex"

        response = requests.get(

            "https://www.alphavantage.co/query",

            params=params,

            timeout=20

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

                "title":
                item.get(
                    "title",
                    "Untitled"
                ),

                "source":
                item.get(
                    "source",
                    "Unknown"
                ),

                "published":
                item.get(
                    "time_published",
                    "Unknown"
                ),

                "url":
                item.get(
                    "url",
                    ""
                )

            })

        return {

            "status":
            "OK",

            "items":
            items

        }

    except Exception as e:

        logger.exception(
            "News error."
        )

        return {

            "status":
            "ERROR",

            "message":
            str(e),

            "items": []

        }


# ============================================================
# DATA QUALITY
# ============================================================

def data_quality(df):

    score = 100

    issues = []

    if df.empty:

        return {
            "score": 0,
            "issues":
            ["EMPTY_DATA"]
        }

    if df.index.has_duplicates:

        score -= 15

        issues.append(
            "DUPLICATE_TIMESTAMP"
        )

    if df[
        ["Open", "High", "Low", "Close"]
    ].isna().any().any():

        score -= 20

        issues.append(
            "MISSING_OHLC"
        )

    if len(df) < 100:

        score -= 10

        issues.append(
            "LIMITED_HISTORY"
        )

    if not issues:

        issues.append(
            "NO_MAJOR_DATA_ISSUE_DETECTED"
        )

    return {

        "score":
        max(0, score),

        "issues":
        issues

    }


# ============================================================
# MULTI-TIMEFRAME ENGINE
# ============================================================

def analyze_tf(
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

            "status":
            "DATA_UNAVAILABLE",

            "message":
            message

        }

    df = calculate_indicators(
        df
    )

    df = detect_swings(
        df
    )

    row = df.iloc[-1]

    return {

        "status": "OK",

        "price":
        float(row["Close"]),

        "rsi":
        float(row["RSI"]),

        "ema20":
        float(row["EMA20"]),

        "ema50":
        float(row["EMA50"]),

        "atr":
        float(row["ATR"]),

        "regime":
        determine_regime(df),

        "structure":
        market_structure(df),

        "liquidity":
        liquidity_event(df),

        "pd":
        premium_discount(df),

        "last_timestamp":
        str(df.index[-1])

    }


def multi_timeframe_analysis(
    symbol
):

    results = {}

    for tf, config in TIMEFRAMES.items():

        try:

            if tf == "4h":

                raw, message = (
                    get_market_data(
                        symbol,
                        "6mo",
                        "1h"
                    )
                )

                if raw is None:

                    results[tf] = {
                        "status":
                        "DATA_UNAVAILABLE",
                        "message":
                        message
                    }

                    continue

                raw = raw.resample(
                    "4h"
                ).agg({

                    "Open":
                    "first",

                    "High":
                    "max",

                    "Low":
                    "min",

                    "Close":
                    "last",

                    "Volume":
                    "sum"

                }).dropna()

                valid, msg = (
                    validate_dataframe(
                        raw
                    )
                )

                if not valid:

                    results[tf] = {
                        "status":
                        "DATA_UNAVAILABLE",
                        "message":
                        msg
                    }

                    continue

                raw = calculate_indicators(
                    raw
                )

                raw = detect_swings(
                    raw
                )

                row = raw.iloc[-1]

                results[tf] = {

                    "status":
                    "OK",

                    "price":
                    float(row["Close"]),

                    "rsi":
                    float(row["RSI"]),

                    "regime":
                    determine_regime(
                        raw
                    ),

                    "structure":
                    market_structure(
                        raw
                    ),

                    "liquidity":
                    liquidity_event(
                        raw
                    ),

                    "pd":
                    premium_discount(
                        raw
                    ),

                    "last_timestamp":
                    str(raw.index[-1])

                }

            else:

                results[tf] = analyze_tf(
                    symbol,
                    config["period"],
                    config["interval"]
                )

        except Exception as e:

            logger.exception(
                "MTF error %s",
                tf
            )

            results[tf] = {

                "status":
                "ERROR",

                "message":
                str(e)

            }

    return results


# ============================================================
# MTF ALIGNMENT
# ============================================================

def mtf_bias(mtf):

    bullish = 0
    bearish = 0

    for tf, data in mtf.items():

        if data.get(
            "status"
        ) != "OK":

            continue

        regime = data.get(
            "regime",
            ""
        )

        if "BULLISH" in regime:

            bullish += 1

        elif "BEARISH" in regime:

            bearish += 1

    if bullish > bearish:

        return "BULLISH"

    if bearish > bullish:

        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# CONFLUENCE ENGINE
# ============================================================

def confluence_score(
    technical,
    structure,
    liquidity,
    pd_zone,
    fvg,
    order_blocks,
    mtf,
    session
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

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    if (
        ema20 is not None
        and ema50 is not None
    ):

        if ema20 > ema50:

            score += 12

            evidence.append(
                "EMA trend bullish"
            )

        elif ema20 < ema50:

            score += 12

            evidence.append(
                "EMA trend bearish"
            )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi >= 55:

        score += 8

        evidence.append(
            "RSI bullish zone"
        )

    elif rsi <= 45:

        score += 8

        evidence.append(
            "RSI bearish zone"
        )

    else:

        conflicts.append(
            "RSI neutral"
        )

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    s = structure.get(
        "structure",
        ""
    )

    if "HH" in s:

        score += 10

        evidence.append(
            "Higher-high structure"
        )

    if "HL" in s:

        score += 8

        evidence.append(
            "Higher-low structure"
        )

    if "LH" in s:

        score += 10

        evidence.append(
            "Lower-high structure"
        )

    if "LL" in s:

        score += 8

        evidence.append(
            "Lower-low structure"
        )

    # --------------------------------------------------------
    # Liquidity
    # --------------------------------------------------------

    le = liquidity.get(
        "event",
        ""
    )

    if "BOS" in le:

        score += 12

        evidence.append(
            le
        )

    elif "SWEEP" in le:

        score += 8

        evidence.append(
            le
        )

    # --------------------------------------------------------
    # P/D
    # --------------------------------------------------------

    zone = pd_zone.get(
        "zone"
    )

    if zone in [
        "PREMIUM",
        "DISCOUNT"
    ]:

        score += 5

        evidence.append(
            "Premium/Discount context available"
        )

    # --------------------------------------------------------
    # FVG
    # --------------------------------------------------------

    if fvg:

        score += 5

        evidence.append(
            f"{len(fvg)} FVG zone(s)"
        )

    # --------------------------------------------------------
    # OB
    # --------------------------------------------------------

    if order_blocks:

        score += 5

        evidence.append(
            f"{len(order_blocks)} OB zone(s)"
        )

    # --------------------------------------------------------
    # MTF
    # --------------------------------------------------------

    multi_bias = mtf_bias(
        mtf
    )

    if (
        multi_bias ==
        "BULLISH"
        and
        "BULLISH" in regime
    ):

        score += 10

        evidence.append(
            "MTF bullish alignment"
        )

    elif (
        multi_bias ==
        "BEARISH"
        and
        "BEARISH" in regime
    ):

        score += 10

        evidence.append(
            "MTF bearish alignment"
        )

    else:

        conflicts.append(
            "MTF not fully aligned"
        )

    # --------------------------------------------------------
    # Session
    # --------------------------------------------------------

    if session.get(
        "state"
    ) == "ACTIVE":

        score += 5

        evidence.append(
            "Active session"
        )

    else:

        conflicts.append(
            "Low/transition session"
        )

    score = min(
        100,
        max(
            0,
            score
        )
    )

    if score >= 75:

        status = (
            "STRONG_RESEARCH_ALIGNMENT"
        )

    elif score >= 60:

        status = (
            "MODERATE_RESEARCH_ALIGNMENT"
        )

    elif score >= 40:

        status = (
            "WEAK_ALIGNMENT"
        )

    else:

        status = (
            "INSUFFICIENT_EVIDENCE"
        )

    return {

        "score":
        score,

        "status":
        status,

        "evidence":
        evidence,

        "conflicts":
        conflicts,

        "mtf_bias":
        multi_bias

    }


# ============================================================
# RESEARCH BIAS
# ============================================================

def calculate_bias(
    technical,
    structure,
    liquidity,
    mtf
):

    bull = 0
    bear = 0

    regime = technical.get(
        "regime",
        ""
    )

    if "BULLISH" in regime:

        bull += 2

    if "BEARISH" in regime:

        bear += 2

    s = structure.get(
        "structure",
        ""
    )

    if "HH" in s:
        bull += 1

    if "HL" in s:
        bull += 1

    if "LH" in s:
        bear += 1

    if "LL" in s:
        bear += 1

    le = liquidity.get(
        "event",
        ""
    )

    if "BULLISH_BOS" in le:
        bull += 2

    if "BEARISH_BOS" in le:
        bear += 2

    if "SELL_SIDE" in le:
        bull += 1

    if "BUY_SIDE" in le:
        bear += 1

    mb = mtf_bias(
        mtf
    )

    if mb == "BULLISH":

        bull += 2

    elif mb == "BEARISH":

        bear += 2

    if bull > bear:

        return "BULLISH"

    if bear > bull:

        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# RESEARCH TP / SL
# ============================================================

def research_levels(
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

        if (
            not math.isfinite(atr)
            or atr <= 0
        ):

            return {
                "status":
                "UNAVAILABLE"
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

            sl = (
                recent_low -
                0.25 * atr
            )

            risk = (
                entry - sl
            )

            if risk <= 0:

                return {
                    "status":
                    "UNAVAILABLE"
                }

            return {

                "status":
                "RESEARCH_ONLY",

                "reference_entry":
                entry,

                "research_sl":
                sl,

                "tp_1_5R":
                entry +
                risk * 1.5,

                "tp_2R":
                entry +
                risk * 2,

                "tp_3R":
                entry +
                risk * 3

            }

        if bias == "BEARISH":

            sl = (
                recent_high +
                0.25 * atr
            )

            risk = (
                sl - entry
            )

            if risk <= 0:

                return {
                    "status":
                    "UNAVAILABLE"
                }

            return {

                "status":
                "RESEARCH_ONLY",

                "reference_entry":
                entry,

                "research_sl":
                sl,

                "tp_1_5R":
                entry -
                risk * 1.5,

                "tp_2R":
                entry -
                risk * 2,

                "tp_3R":
                entry -
                risk * 3

            }

        return {
            "status":
            "NO_BIAS"
        }

    except Exception:

        logger.exception(
            "Research levels error."
        )

        return {
            "status":
            "ERROR"
        }


# ============================================================
# RESEARCH ENGINE
# ============================================================

def full_research(
    symbol
):

    symbol = normalize_symbol(
        symbol
    )

    if not symbol:

        return {

            "status":
            "INVALID_SYMBOL",

            "message":
            "Market not supported."

        }

    # --------------------------------------------------------
    # Main timeframe
    # --------------------------------------------------------

    df, message = (
        get_market_data(
            symbol,
            "1mo",
            "1h"
        )
    )

    if df is None:

        return {

            "status":
            "DATA_UNAVAILABLE",

            "message":
            message

        }

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    df = calculate_indicators(
        df
    )

    # --------------------------------------------------------
    # Swings
    # --------------------------------------------------------

    df = detect_swings(
        df
    )

    # --------------------------------------------------------
    # Technical
    # --------------------------------------------------------

    row = df.iloc[-1]

    technical = {

        "price":
        float(row["Close"]),

        "EMA20":
        float(row["EMA20"]),

        "EMA50":
        float(row["EMA50"]),

        "EMA200":
        (
            float(row["EMA200"])
            if not pd.isna(
                row["EMA200"]
            )
            else None
        ),

        "RSI":
        float(row["RSI"]),

        "MACD":
        float(row["MACD"]),

        "MACD_SIGNAL":
        float(row["MACD_SIGNAL"]),

        "ATR":
        float(row["ATR"]),

        "ATR_PCT":
        float(row["ATR_PCT"]),

        "VOLATILITY":
        (
            float(
                row["VOLATILITY"]
            )
            if not pd.isna(
                row["VOLATILITY"]
            )
            else None
        ),

        "regime":
        determine_regime(df),

        "candle":
        candle_analysis(df)

    }

    # --------------------------------------------------------
    # SMC
    # --------------------------------------------------------

    structure = (
        market_structure(df)
    )

    liquidity = (
        liquidity_event(df)
    )

    fvg = detect_fvg(
        df
    )

    order_blocks = (
        detect_order_blocks(df)
    )

    pd_zone = (
        premium_discount(df)
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    session = (
        session_status()
    )

    news = get_news(
        symbol
    )

    # --------------------------------------------------------
    # MTF
    # --------------------------------------------------------

    mtf = (
        multi_timeframe_analysis(
            symbol
        )
    )

    # --------------------------------------------------------
    # Bias
    # --------------------------------------------------------

    bias = calculate_bias(
        technical,
        structure,
        liquidity,
        mtf
    )

    # --------------------------------------------------------
    # Confluence
    # --------------------------------------------------------

    confluence = confluence_score(

        technical,

        structure,

        liquidity,

        pd_zone,

        fvg,

        order_blocks,

        mtf,

        session

    )

    # --------------------------------------------------------
    # Research Levels
    # --------------------------------------------------------

    levels = research_levels(
        df,
        bias
    )

    # --------------------------------------------------------
    # Data Quality
    # --------------------------------------------------------

    quality = data_quality(
        df
    )

    # --------------------------------------------------------
    # WAIT GATE
    # --------------------------------------------------------

    wait_reasons = []

    if quality["score"] < 80:

        wait_reasons.append(
            "Data quality below preferred threshold"
        )

    if session["state"] == "WAIT":

        wait_reasons.append(
            "Low-activity session"
        )

    if confluence["score"] < MIN_CONFLUENCE:

        wait_reasons.append(
            "Insufficient confluence"
        )

    if technical["regime"] == "UNKNOWN":

        wait_reasons.append(
            "Unknown market regime"
        )

    if bias == "NEUTRAL":

        wait_reasons.append(
            "No directional research bias"
        )

    if news["status"] != "OK":

        wait_reasons.append(
            "News data unavailable"
        )

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    if wait_reasons:

        final_state = "WAIT"

    else:

        final_state = (
            "RESEARCH_SETUP_DETECTED"
        )

    result = {

        "status":
        "OK",

        "symbol":
        symbol,

        "name":
        MARKETS[
            symbol
        ]["name"],

        "source":
        MARKETS[
            symbol
        ]["yf"],

        "last_timestamp":
        str(df.index[-1]),

        "technical":
        technical,

        "structure":
        structure,

        "liquidity":
        liquidity,

        "fvg":
        fvg,

        "order_blocks":
        order_blocks,

        "premium_discount":
        pd_zone,

        "session":
        session,

        "news":
        news,

        "mtf":
        mtf,

        "bias":
        bias,

        "confluence":
        confluence,

        "levels":
        levels,

        "quality":
        quality,

        "final_state":
        final_state,

        "wait_reasons":
        wait_reasons,

        "mode":
        "RESEARCH/PAPER"

    }

    # --------------------------------------------------------
    # Save research log
    # --------------------------------------------------------

    try:

        conn = sqlite3.connect(
            DB_FILE
        )

        conn.execute("""

            INSERT INTO research_logs
            (
                symbol,
                timeframe,
                score,
                state,
                regime,
                created_at
            )

            VALUES (?, ?, ?, ?, ?, ?)

        """, (

            symbol,

            "1h",

            confluence["score"],

            final_state,

            technical["regime"],

            datetime.now(
                timezone.utc
            ).isoformat()

        ))

        conn.commit()
        conn.close()

    except Exception:

        logger.exception(
            "Research log failed."
        )

    return result


# ============================================================
# REPORT FORMATTER
# ============================================================

def format_report(
    result
):

    if result["status"] != "OK":

        return (
            "⚠️ MARKET RESEARCH\n\n"
            f"Status: "
            f"{result['status']}\n\n"
            f"Reason: "
            f"{result.get('message', 'Unknown')}\n\n"
            "Fake data তৈরি করা হচ্ছে না."
        )

    tech = result[
        "technical"
    ]

    structure = result[
        "structure"
    ]

    liquidity = result[
        "liquidity"
    ]

    pdz = result[
        "premium_discount"
    ]

    confluence = result[
        "confluence"
    ]

    session = result[
        "session"
    ]

    levels = result[
        "levels"
    ]

    quality = result[
        "quality"
    ]

    lines = [

        "━━━━━━━━━━━━━━━━━━━━",

        f"📊 {result['name']}",

        "🧠 MASTER MARKET RESEARCH",

        "━━━━━━━━━━━━━━━━━━━━",

        f"Mode: {result['mode']}",

        f"Data: {result['source']}",

        f"Last Candle: "
        f"{result['last_timestamp']}",

        "",

        "💰 PRICE",

        f"{tech['price']}",

        "",

        "📈 TECHNICAL",

        f"EMA20: {tech['EMA20']:.6f}",

        f"EMA50: {tech['EMA50']:.6f}",

        (
            f"EMA200: "
            f"{tech['EMA200']:.6f}"
            if tech["EMA200"]
            is not None
            else
            "EMA200: N/A"
        ),

        f"RSI: {tech['RSI']:.2f}",

        f"MACD: {tech['MACD']:.6f}",

        f"ATR: {tech['ATR']:.6f}",

        f"Regime: "
        f"{tech['regime']}",

        f"Candle: "
        f"{tech['candle']}",

        "",

        "🏗 MARKET STRUCTURE",

        f"{structure.get('structure', 'N/A')}",

        "",

        "💧 LIQUIDITY",

        f"{liquidity.get('event', 'N/A')}",

        "",

        "⚖️ PREMIUM / DISCOUNT",

        f"{pdz.get('zone', 'N/A')}",

        "",

        "🕒 SESSION",

        f"{session['session']} "
        f"| {session['state']}",

        "",

        "🧠 MTF",

        f"MTF Bias: "
        f"{confluence['mtf_bias']}",

        "",

        "🔥 CONFLUENCE",

        f"Score: "
        f"{confluence['score']}/100",

        f"Status: "
        f"{confluence['status']}",

        "",

        "🎯 RESEARCH BIAS",

        result["bias"],

        "",

        "📐 RESEARCH LEVELS"

    ]

    if (
        levels.get("status")
        == "RESEARCH_ONLY"
    ):

        lines.extend([

            f"Reference Entry: "
            f"{levels['reference_entry']}",

            f"Research SL: "
            f"{levels['research_sl']}",

            f"Research TP 1.5R: "
            f"{levels['tp_1_5R']}",

            f"Research TP 2R: "
            f"{levels['tp_2R']}",

            f"Research TP 3R: "
            f"{levels['tp_3R']}"

        ])

    else:

        lines.append(
            "Research levels unavailable."
        )

    lines.extend([

        "",

        "🛡 DATA QUALITY",

        f"Score: "
        f"{quality['score']}/100",

        "",

        "🚦 FINAL STATE"

    ])

    if result[
        "final_state"
    ] == "WAIT":

        lines.append(
            "🟡 WAIT"
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

    lines.extend([

        "",

        "⚠️ Research/Paper mode.",

        "এটি guaranteed future prediction "
        "বা real-money execution নয়.",

        "━━━━━━━━━━━━━━━━━━━━"

    ])

    return "\n".join(
        lines
    )


# ============================================================
# PAPER TRADING
# ============================================================

def save_paper_trade(
    symbol,
    timeframe,
    bias,
    entry,
    sl,
    tp
):

    try:

        conn = sqlite3.connect(
            DB_FILE
        )

        conn.execute("""

            INSERT INTO paper_trades
            (
                symbol,
                timeframe,
                bias,
                entry,
                sl,
                tp,
                result,
                created_at
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            symbol,

            timeframe,

            bias,

            entry,

            sl,

            tp,

            "OPEN",

            datetime.now(
                timezone.utc
            ).isoformat()

        ))

        conn.commit()
        conn.close()

        return True

    except Exception:

        logger.exception(
            "Paper trade save error."
        )

        return False


# ============================================================
# HISTORICAL EVENT STUDY
# ============================================================

def historical_event_study(
    symbol,
    period="1y",
    interval="1d"
):

    df, message = (
        get_market_data(
            symbol,
            period,
            interval
        )
    )

    if df is None:

        return {

            "status":
            "DATA_UNAVAILABLE",

            "message":
            message

        }

    df = calculate_indicators(
        df
    )

    events = 0

    bullish_follow = 0

    bearish_follow = 0

    forward_bars = 3

    for i in range(
        50,
        len(df) -
        forward_bars
    ):

        ema20 = (
            df["EMA20"]
            .iloc[i]
        )

        ema50 = (
            df["EMA50"]
            .iloc[i]
        )

        rsi = (
            df["RSI"]
            .iloc[i]
        )

        future_close = (
            df["Close"]
            .iloc[
                i + forward_bars
            ]
        )

        current_close = (
            df["Close"]
            .iloc[i]
        )

        if (
            ema20 > ema50
            and
            rsi >= 55
        ):

            events += 1

            if (
                future_close >
                current_close
            ):

                bullish_follow += 1

        elif (
            ema20 < ema50
            and
            rsi <= 45
        ):

            events += 1

            if (
                future_close <
                current_close
            ):

                bearish_follow += 1

    directional_rate = (

        (
            bullish_follow +
            bearish_follow
        )
        /
        events *
        100

        if events
        else 0

    )

    return {

        "status":
        "OK",

        "events":
        events,

        "bullish_follow":
        bullish_follow,

        "bearish_follow":
        bearish_follow,

        "historical_follow_rate":
        round(
            directional_rate,
            2
        )

    }


# ============================================================
# SIMPLE WALK-FORWARD RESEARCH
# ============================================================

def walk_forward_research(
    symbol
):

    df, message = (
        get_market_data(
            symbol,
            "2y",
            "1d"
        )
    )

    if df is None:

        return {

            "status":
            "DATA_UNAVAILABLE",

            "message":
            message

        }

    if len(df) < 250:

        return {

            "status":
            "INSUFFICIENT_HISTORY",

            "candles":
            len(df)

        }

    split = int(
        len(df) * 0.70
    )

    train = df.iloc[
        :split
    ].copy()

    test = df.iloc[
        split:
    ].copy()

    train = calculate_indicators(
        train
    )

    test = calculate_indicators(
        test
    )

    train_events = 0
    train_follow = 0

    test_events = 0
    test_follow = 0

    for i in range(
        50,
        len(train) - 3
    ):

        if (
            train["EMA20"].iloc[i]
            >
            train["EMA50"].iloc[i]
            and
            train["RSI"].iloc[i]
            >= 55
        ):

            train_events += 1

            if (
                train["Close"].iloc[i + 3]
                >
                train["Close"].iloc[i]
            ):

                train_follow += 1

        elif (
            train["EMA20"].iloc[i]
            <
            train["EMA50"].iloc[i]
            and
            train["RSI"].iloc[i]
            <= 45
        ):

            train_events += 1

            if (
                train["Close"].iloc[i + 3]
                <
                train["Close"].iloc[i]
            ):

                train_follow += 1

    for i in range(
        50,
        len(test) - 3
    ):

        if (
            test["EMA20"].iloc[i]
            >
            test["EMA50"].iloc[i]
            and
            test["RSI"].iloc[i]
            >= 55
        ):

            test_events += 1

            if (
                test["Close"].iloc[i + 3]
                >
                test["Close"].iloc[i]
            ):

                test_follow += 1

        elif (
            test["EMA20"].iloc[i]
            <
            test["EMA50"].iloc[i]
            and
            test["RSI"].iloc[i]
            <= 45
        ):

            test_events += 1

            if (
                test["Close"].iloc[i + 3]
                <
                test["Close"].iloc[i]
            ):

                test_follow += 1

    train_rate = (

        train_follow /
        train_events *
        100

        if train_events
        else 0

    )

    test_rate = (

        test_follow /
        test_events *
        100

        if test_events
        else 0

    )

    return {

        "status":
        "OK",

        "train_events":
        train_events,

        "train_rate":
        round(
            train_rate,
            2
        ),

        "test_events":
        test_events,

        "test_rate":
        round(
            test_rate,
            2
        )

    }


# ============================================================
# MONTE CARLO STYLE RESEARCH
# ============================================================

def monte_carlo_research(
    historical_results,
    simulations=1000
):

    if not historical_results:

        return {
            "status":
            "NO_DATA"
        }

    values = np.array(
        historical_results,
        dtype=float
    )

    if len(values) < 20:

        return {

            "status":
            "INSUFFICIENT_DATA",

            "samples":
            len(values)

        }

    final_values = []

    for _ in range(
        simulations
    ):

        sample = np.random.choice(
            values,
            size=len(values),
            replace=True
        )

        final_values.append(
            np.sum(sample)
        )

    return {

        "status":
        "OK",

        "simulations":
        simulations,

        "mean":
        float(
            np.mean(
                final_values
            )
        ),

        "median":
        float(
            np.median(
                final_values
            )
        ),

        "p05":
        float(
            np.percentile(
                final_values,
                5
            )
        ),

        "p95":
        float(
            np.percentile(
                final_values,
                95
            )
        )

    }


# ============================================================
# CORRELATION ENGINE
# ============================================================

def cross_market_correlation():

    closes = {}

    for symbol in MARKETS:

        df, message = (
            get_market_data(
                symbol,
                "3mo",
                "1d"
            )
        )

        if df is None:
            continue

        closes[
            symbol
        ] = df["Close"]

    if len(closes) < 2:

        return None

    combined = pd.concat(
        closes,
        axis=1
    )

    return combined.pct_change().corr()


# ============================================================
# AI EDUCATIONAL ASSISTANT
# ============================================================

async def ask_gemini(
    question
):

    if gemini_client is None:

        return (
            "🤖 Gemini unavailable.\n\n"
            "GEMINI_API_KEY configure করা হয়নি."
        )

    prompt = f"""

You are an educational market-research
assistant.

User question:

{question}

Rules:

1. Never claim guaranteed future price movement.
2. Never fabricate live market data.
3. Clearly distinguish historical research
   from future uncertainty.
4. Explain technical concepts clearly.
5. Use research/paper-trading framing.
6. Do not execute trades.
7. Do not provide guaranteed signals.
8. If data is unavailable, say so.

Answer in clear Bengali when possible.

"""

    try:

        response = (
            await gemini_client
            .aio
            .models
            .generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
        )

        text = getattr(
            response,
            "text",
            None
        )

        if text:

            return text

        return (
            "Gemini কোনো text response দেয়নি."
        )

    except Exception as e:

        logger.exception(
            "Gemini error."
        )

        return (
            "🤖 AI ERROR\n\n"
            + str(e)
        )


# ============================================================
# TELEGRAM KEYBOARD
# ============================================================

def main_keyboard():

    return InlineKeyboardMarkup([

        [

            InlineKeyboardButton(
                "📊 Markets",
                callback_data="markets"
            ),

            InlineKeyboardButton(
                "❤️ Health",
                callback_data="health"
            )

        ],

        [

            InlineKeyboardButton(
                "🔬 Research",
                callback_data="research"
            ),

            InlineKeyboardButton(
                "📰 News",
                callback_data="news"
            )

        ],

        [

            InlineKeyboardButton(
                "📚 Backtest",
                callback_data="backtest"
            ),

            InlineKeyboardButton(
                "🤖 AI",
                callback_data="ai"
            )

        ]

    ])


# ============================================================
# /START
# ============================================================

async def start_command(
    update,
    context
):

    text = """

👋 আসসালামু আলাইকুম!

🧠 MASTER MARKET RESEARCH BOT

এই bot এখন research/paper mode-এ কাজ করে।

📊 Supported Markets:

• EURUSD
• BTCUSD
• GOLD / XAUUSD
• ETHUSD
• USDJPY

সরাসরি লিখতে পারো:

EURUSD
BTCUSD
GOLD
ETHUSD
USDJPY

অথবা:

/analyze EURUSD

Commands:

/start
/help
/health
/price EURUSD
/analyze EURUSD
/news
/backtest EURUSD
/walkforward EURUSD
/correlation

⚠️ এটি guaranteed prediction
বা real-money execution bot নয়।

"""

    await update.message.reply_text(
        text,
        reply_markup=
        main_keyboard()
    )


# ============================================================
# /HELP
# ============================================================

async def help_command(
    update,
    context
):

    text = """

📚 COMMAND GUIDE

/start
→ Main menu

/help
→ Commands

/health
→ System health

/price EURUSD
→ Available market data

/analyze EURUSD
→ Full research pipeline

/news
→ Available news

/backtest EURUSD
→ Historical event study

/walkforward EURUSD
→ Train/Test research

/correlation
→ Five-market correlation

Direct message:

EURUSD
BTCUSD
GOLD
ETHUSD
USDJPY

সাধারণ প্রশ্নও করা যাবে।

"""

    await update.message.reply_text(
        text
    )


# ============================================================
# /HEALTH
# ============================================================

async def health_command(
    update,
    context
):

    text = f"""

❤️ SYSTEM HEALTH

Telegram:
{'READY' if TELEGRAM_BOT_TOKEN else 'MISSING'}

Gemini:
{'READY' if gemini_client else 'NOT CONFIGURED'}

News:
{'READY' if ALPHA_VANTAGE_API_KEY else 'NOT CONFIGURED'}

Database:
READY

Market Data:
Yahoo Finance adapter

Mode:
RESEARCH / PAPER

Scanner:
BACKGROUND RESEARCH

Fake data:
DISABLED

"""

    await update.message.reply_text(
        text
    )


# ============================================================
# /PRICE
# ============================================================

async def price_command(
    update,
    context
):

    if not context.args:

        await update.message.reply_text(
            "ব্যবহার:\n"
            "/price EURUSD\n"
            "/price GOLD\n"
            "/price BTCUSD"
        )

        return

    symbol = normalize_symbol(
        context.args[0]
    )

    if not symbol:

        await update.message.reply_text(
            "❌ Supported markets:\n"
            "EURUSD\n"
            "BTCUSD\n"
            "GOLD\n"
            "ETHUSD\n"
            "USDJPY"
        )

        return

    df, message = (
        get_market_data(
            symbol,
            "5d",
            "1h"
        )
    )

    if df is None:

        await update.message.reply_text(

            "⚠️ DATA UNAVAILABLE\n\n"
            f"Reason: {message}\n\n"
            "Fake price দেখানো হবে না."

        )

        return

    row = df.iloc[-1]

    text = f"""

💰 MARKET DATA

Market:
{MARKETS[symbol]['name']}

Price:
{float(row['Close'])}

Last Candle:
{df.index[-1]}

Candles:
{len(df)}

Source:
{MARKETS[symbol]['yf']}

⚠️ এটি exact broker OTC feed নয়।

"""

    await update.message.reply_text(
        text
    )


# ============================================================
# /ANALYZE
# ============================================================

async def analyze_command(
    update,
    context
):

    if not context.args:

        await update.message.reply_text(
            "ব্যবহার:\n"
            "/analyze EURUSD\n"
            "/analyze GOLD\n"
            "/analyze BTCUSD"
        )

        return

    symbol = normalize_symbol(
        context.args[0]
    )

    if not symbol:

        await update.message.reply_text(
            "❌ Market supported নয়."
        )

        return

    await update.message.reply_text(

        f"🔄 {symbol} research শুরু...\n\n"
        "Data → Technical → MTF → "
        "Structure → Liquidity → FVG → "
        "OB → P/D → Session → News → "
        "Confluence → Research Levels"

    )

    try:

        result = full_research(
            symbol
        )

        await update.message.reply_text(
            format_report(
                result
            )
        )

    except Exception as e:

        logger.exception(
            "Analyze failed."
        )

        await update.message.reply_text(

            "❌ ANALYSIS ERROR\n\n"
            + str(e)

        )


# ============================================================
# /NEWS
# ============================================================

async def news_command(
    update,
    context
):

    symbol = None

    if context.args:

        symbol = normalize_symbol(
            context.args[0]
        )

    result = get_news(
        symbol
    )

    if result["status"] != "OK":

        await update.message.reply_text(

            "📰 NEWS UNAVAILABLE\n\n"
            +
            result.get(
                "message",
                "API unavailable"
            )

        )

        return

    items = result[
        "items"
    ]

    if not items:

        await update.message.reply_text(
            "📰 কোনো news item পাওয়া যায়নি."
        )

        return

    lines = [
        "📰 AVAILABLE NEWS",
        ""
    ]

    for i, item in enumerate(
        items[:8],
        1
    ):

        lines.append(
            f"{i}. "
            f"{item['title']}"
        )

        lines.append(
            f"Source: "
            f"{item['source']}"
        )

        lines.append(
            f"Published: "
            f"{item['published']}"
        )

        lines.append("")

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# /BACKTEST
# ============================================================

async def backtest_command(
    update,
    context
):

    if not context.args:

        await update.message.reply_text(
            "ব্যবহার:\n"
            "/backtest EURUSD"
        )

        return

    symbol = normalize_symbol(
        context.args[0]
    )

    if not symbol:

        await update.message.reply_text(
            "❌ Market supported নয়."
        )

        return

    await update.message.reply_text(
        f"📚 {symbol} historical research চলছে..."
    )

    result = historical_event_study(
        symbol
    )

    text = """

📚 HISTORICAL EVENT STUDY

"""

    text += json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )

    text += """

⚠️ Historical result future result
guarantee করে না।
"""

    await update.message.reply_text(
        text
    )


# ============================================================
# /WALKFORWARD
# ============================================================

async def walkforward_command(
    update,
    context
):

    if not context.args:

        await update.message.reply_text(
            "ব্যবহার:\n"
            "/walkforward EURUSD"
        )

        return

    symbol = normalize_symbol(
        context.args[0]
    )

    if not symbol:

        await update.message.reply_text(
            "❌ Market supported নয়."
        )

        return

    await update.message.reply_text(
        "🧪 Walk-forward research চলছে..."
    )

    result = walk_forward_research(
        symbol
    )

    text = """

🧪 WALK-FORWARD RESEARCH

"""

    text += json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )

    await update.message.reply_text(
        text
    )


# ============================================================
# /CORRELATION
# ============================================================

async def correlation_command(
    update,
    context
):

    await update.message.reply_text(
        "🔗 Cross-market correlation গণনা হচ্ছে..."
    )

    matrix = (
        cross_market_correlation()
    )

    if matrix is None:

        await update.message.reply_text(
            "❌ Correlation data unavailable."
        )

        return

    text = (
        "🔗 FIVE-MARKET CORRELATION\n\n"
        +
        matrix.round(2).to_string()
    )

    await update.message.reply_text(
        text
    )


# ============================================================
# BUTTON HANDLER
# ============================================================

async def button_handler(
    update,
    context
):

    query = (
        update.callback_query
    )

    await query.answer()

    data = query.data

    if data == "markets":

        text = (
            "📊 SUPPORTED MARKETS\n\n"
            +
            "\n".join(
                "• " +
                MARKETS[x]["name"]
                for x in MARKETS
            )
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

            f"News: "
            f"{'READY' if ALPHA_VANTAGE_API_KEY else 'NOT CONFIGURED'}\n"

            "Database: READY\n"

            "Mode: RESEARCH/PAPER"

        )

        await query.edit_message_text(
            text
        )

    elif data == "research":

        await query.edit_message_text(

            "🔬 RESEARCH\n\n"
            "/analyze EURUSD\n"
            "/analyze BTCUSD\n"
            "/analyze GOLD\n"
            "/analyze ETHUSD\n"
            "/analyze USDJPY"

        )

    elif data == "news":

        result = get_news()

        if result["status"] != "OK":

            await query.edit_message_text(
                "📰 News unavailable."
            )

            return

        items = result[
            "items"
        ]

        if not items:

            await query.edit_message_text(
                "No news available."
            )

            return

        text = "📰 NEWS\n\n"

        for i, item in enumerate(
            items[:5],
            1
        ):

            text += (
                f"{i}. "
                f"{item['title']}\n"
                f"{item['source']}\n\n"
            )

        await query.edit_message_text(
            text
        )

    elif data == "backtest":

        await query.edit_message_text(

            "📚 BACKTEST\n\n"
            "/backtest EURUSD\n"
            "/backtest BTCUSD\n"
            "/backtest GOLD\n"
            "/backtest ETHUSD\n"
            "/backtest USDJPY"

        )

    elif data == "ai":

        await query.edit_message_text(

            "🤖 AI RESEARCH ASSISTANT\n\n"
            "যেকোনো market-research প্রশ্ন "
            "message হিসেবে পাঠাও।"

        )


# ============================================================
# DIRECT MARKET DETECTOR
# ============================================================

def detect_market(
    text
):

    normalized = normalize_symbol(
        text
    )

    if normalized:

        return normalized

    upper = (
        text.upper()
    )

    for alias, symbol in ALIASES.items():

        if alias in upper:

            return symbol

    for symbol in MARKETS:

        if symbol in upper:

            return symbol

    return None


# ============================================================
# GENERAL MESSAGE HANDLER
# ============================================================

async def handle_message(
    update,
    context
):

    if not update.message:

        return

    text = (
        update.message.text or ""
    ).strip()

    if not text:

        await update.message.reply_text(
            "Empty message received."
        )

        return

    # --------------------------------------------------------
    # Direct market
    # --------------------------------------------------------

    market = detect_market(
        text
    )

    if market:

        await update.message.reply_text(

            f"🔎 {MARKETS[market]['name']} detected.\n\n"
            "🔄 Research pipeline শুরু হচ্ছে..."

        )

        try:

            result = full_research(
                market
            )

            await update.message.reply_text(
                format_report(
                    result
                )
            )

        except Exception as e:

            logger.exception(
                "Direct market error."
            )

            await update.message.reply_text(

                "❌ Analysis error\n\n"
                + str(e)

            )

        return

    # --------------------------------------------------------
    # Greetings
    # --------------------------------------------------------

    greetings = {

        "hi",
        "hello",
        "hey",
        "হাই",
        "হ্যালো",
        "salam",
        "assalamualaikum",
        "আসসালামু আলাইকুম"

    }

    if text.lower() in greetings:

        await update.message.reply_text(

            "👋 Wa Alaikum Assalam!\n\n"
            "আমি online আছি।\n\n"
            "EURUSD / GOLD / BTCUSD "
            "লিখে research শুরু করতে পারো।"

        )

        return

    # --------------------------------------------------------
    # Help words
    # --------------------------------------------------------

    if text.lower() in {

        "help",
        "menu",
        "guide",
        "কি করতে পারি",
        "কী করতে পারি"

    }:

        await update.message.reply_text(

            "📚 তুমি করতে পারো:\n\n"
            "• EURUSD\n"
            "• BTCUSD\n"
            "• GOLD\n"
            "• ETHUSD\n"
            "• USDJPY\n\n"
            "• /analyze EURUSD\n"
            "• /price GOLD\n"
            "• /news\n"
            "• /backtest EURUSD\n"
            "• /walkforward EURUSD\n"
            "• /correlation"

        )

        return

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    reply = await ask_gemini(
        text
    )

    await update.message.reply_text(
        reply
    )


# ============================================================
# BACKGROUND SCANNER
# ============================================================

scanner_running = True

last_scan_state = {}


def background_scanner():

    global scanner_running

    logger.info(
        "Background scanner started."
    )

    while scanner_running:

        try:

            for symbol in MARKETS:

                try:

                    result = full_research(
                        symbol
                    )

                    score = (
                        result
                        .get(
                            "confluence",
                            {}
                        )
                        .get(
                            "score",
                            0
                        )
                    )

                    state = (
                        result.get(
                            "final_state",
                            "WAIT"
                        )
                    )

                    last_scan_state[
                        symbol
                    ] = {

                        "score":
                        score,

                        "state":
                        state,

                        "time":
                        datetime.now(
                            timezone.utc
                        ).isoformat()

                    }

                    try:

                        conn = sqlite3.connect(
                            DB_FILE
                        )

                        conn.execute("""

                            INSERT INTO scan_logs
                            (
                                symbol,
                                score,
                                state,
                                created_at
                            )

                            VALUES (?, ?, ?, ?)

                        """, (

                            symbol,

                            score,

                            state,

                            datetime.now(
                                timezone.utc
                            ).isoformat()

                        ))

                        conn.commit()
                        conn.close()

                    except Exception:

                        logger.exception(
                            "Scan log error."
                        )

                except Exception:

                    logger.exception(
                        "Scanner market error: %s",
                        symbol
                    )

        except Exception:

            logger.exception(
                "Scanner cycle error."
            )

        time.sleep(
            SCAN_INTERVAL
        )


# ============================================================
# /SCANNER
# ============================================================

async def scanner_command(
    update,
    context
):

    if not last_scan_state:

        await update.message.reply_text(

            "🔄 Scanner এখনো প্রথম cycle শেষ করেনি."

        )

        return

    lines = [
        "📡 SCANNER STATUS",
        ""
    ]

    for symbol, data in (
        last_scan_state.items()
    ):

        lines.append(

            f"{symbol} | "
            f"Score {data['score']} | "
            f"{data['state']}"

        )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update,
    context
):

    logger.exception(
        "Unhandled Telegram error",
        exc_info=context.error
    )

    try:

        if (
            isinstance(
                update,
                Update
            )
            and
            update.effective_message
        ):

            await update.effective_message.reply_text(

                "⚠️ Bot-এর ভিতরে error হয়েছে.\n\n"
                + str(
                    context.error
                )

            )

    except Exception:

        logger.exception(
            "Could not send error."
        )


# ============================================================
# POST INIT
# ============================================================

async def post_init(
    application
):

    try:

        await application.bot.set_my_commands([

            (
                "start",
                "Open main menu"
            ),

            (
                "help",
                "Show commands"
            ),

            (
                "health",
                "System health"
            ),

            (
                "price",
                "Market data"
            ),

            (
                "analyze",
                "Full research"
            ),

            (
                "news",
                "Available news"
            ),

            (
                "backtest",
                "Historical study"
            ),

            (
                "walkforward",
                "Walk-forward research"
            ),

            (
                "correlation",
                "Market correlation"
            ),

            (
                "scanner",
                "Scanner status"
            )

        ])

        logger.info(
            "Telegram commands registered."
        )

    except Exception:

        logger.exception(
            "Command registration failed."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing."
        )

    # --------------------------------------------------------
    # Flask
    # --------------------------------------------------------

    flask_thread = threading.Thread(

        target=run_flask,

        daemon=True

    )

    flask_thread.start()

    # --------------------------------------------------------
    # Background Scanner
    # --------------------------------------------------------

    scanner_thread = threading.Thread(

        target=background_scanner,

        daemon=True

    )

    scanner_thread.start()

    # --------------------------------------------------------
    # Telegram
    # --------------------------------------------------------

    application = (

        ApplicationBuilder()

        .token(
            TELEGRAM_BOT_TOKEN
        )

        .post_init(
            post_init
        )

        .build()

    )

    # --------------------------------------------------------
    # Commands
    # --------------------------------------------------------

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

    application.add_handler(
        CommandHandler(
            "backtest",
            backtest_command
        )
    )

    application.add_handler(
        CommandHandler(
            "walkforward",
            walkforward_command
        )
    )

    application.add_handler(
        CommandHandler(
            "correlation",
            correlation_command
        )
    )

    application.add_handler(
        CommandHandler(
            "scanner",
            scanner_command
        )
    )

    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------

    application.add_handler(
        CallbackQueryHandler(
            button_handler
        )
    )

    # --------------------------------------------------------
    # Text messages
    # --------------------------------------------------------

    application.add_handler(

        MessageHandler(

            filters.TEXT
            &
            ~filters.COMMAND,

            handle_message

        )

    )

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    application.add_error_handler(
        error_handler
    )

    logger.info(
        "======================================"
    )

    logger.info(
        "MASTER MARKET RESEARCH BOT STARTING"
    )

    logger.info(
        "Markets: EURUSD / BTCUSD / GOLD / ETHUSD / USDJPY"
    )

    logger.info(
        "Mode: RESEARCH / PAPER"
    )

    logger.info(
        "======================================"
    )

    application.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()×××××××××2×××
××××

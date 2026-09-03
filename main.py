# ============================================================
# MASTER MARKET RESEARCH BOT
# STEP 2A + STEP 2B
# ============================================================

import os
import logging
import threading

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from flask import Flask

from dotenv import load_dotenv

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash"
)

ALPHA_VANTAGE_API_KEY = os.getenv(
    "ALPHA_VANTAGE_API_KEY"
)

PORT = int(
    os.getenv("PORT", "10000")
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format=(
        "%(asctime)s - "
        "%(name)s - "
        "%(levelname)s - "
        "%(message)s"
    ),
    level=logging.INFO,
)

logger = logging.getLogger(
    "MasterMarketBot"
)


# ============================================================
# ENV VALIDATION
# ============================================================

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN is missing."
    )

if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY is missing. "
        "AI Chat will be unavailable."
    )

if not ALPHA_VANTAGE_API_KEY:
    logger.warning(
        "ALPHA_VANTAGE_API_KEY is missing. "
        "News engine will be unavailable."
    )


# ============================================================
# GEMINI
# ============================================================

gemini_client = None

if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        logger.info(
            "Gemini client initialized."
        )

    except Exception as error:

        logger.exception(
            "Gemini initialization failed: %s",
            error
        )


# ============================================================
# FLASK HEALTH SERVER
# ============================================================

flask_app = Flask(__name__)


@flask_app.route("/")
def home():

    return (
        "Master Market Research Bot "
        "is running."
    )


@flask_app.route("/health")
def health():

    return {
        "status": "online",
        "telegram": bool(
            TELEGRAM_BOT_TOKEN
        ),
        "gemini": bool(
            gemini_client
        ),
        "alpha_vantage": bool(
            ALPHA_VANTAGE_API_KEY
        ),
        "market_data": "Yahoo Finance",
        "trading_execution": "DISABLED",
    }


def run_flask():

    flask_app.run(
        host="0.0.0.0",
        port=PORT,
        use_reloader=False,
    )


# ============================================================
# SYMBOL MAP
# ============================================================

SYMBOL_MAP = {

    "EURUSD": "EURUSD=X",

    "GBPUSD": "GBPUSD=X",

    "USDJPY": "JPY=X",

    "USDCHF": "CHF=X",

    "AUDUSD": "AUDUSD=X",

    "USDCAD": "CAD=X",

    "NZDUSD": "NZDUSD=X",

    "XAUUSD": "GC=F",

    "GOLD": "GC=F",

    "BTCUSD": "BTC-USD",

    "ETHUSD": "ETH-USD",
}


WATCHLIST = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "XAUUSD",
    "BTCUSD",
    "ETHUSD",
]


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataframe(df):

    if df is None:
        return False, "DATA_IS_NONE"

    if df.empty:
        return False, "EMPTY_DATA"

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        return (
            False,
            "MISSING_COLUMNS: "
            + ",".join(missing)
        )

    data = df[
        required_columns
    ].copy()

    if data.isnull().all().any():

        return False, "INVALID_NULL_DATA"

    if len(data) < 20:

        return (
            False,
            "INSUFFICIENT_DATA"
        )

    return True, "OK"


# ============================================================
# MARKET DATA ENGINE
# ============================================================

def get_market_data(
    symbol,
    period="5d",
    interval="1h",
):

    symbol = symbol.upper().strip()

    yahoo_symbol = SYMBOL_MAP.get(
        symbol
    )

    if not yahoo_symbol:

        return {
            "status": "UNKNOWN_SYMBOL",
            "symbol": symbol,
        }

    try:

        df = yf.download(
            yahoo_symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = [
                column[0]
                for column in df.columns
            ]

        valid, reason = (
            validate_dataframe(df)
        )

        if not valid:

            return {
                "status": "DATA_UNAVAILABLE",
                "symbol": symbol,
                "reason": reason,
            }

        latest = df.iloc[-1]

        timestamp = str(
            df.index[-1]
        )

        return {

            "status": "DATA_RECEIVED",

            "symbol": symbol,

            "yahoo_symbol": yahoo_symbol,

            "price": float(
                latest["Close"]
            ),

            "open": float(
                latest["Open"]
            ),

            "high": float(
                latest["High"]
            ),

            "low": float(
                latest["Low"]
            ),

            "timestamp": timestamp,

            "rows": len(df),

            "source": (
                "Yahoo Finance"
            ),
        }

    except Exception as error:

        logger.exception(
            "Market data error: %s",
            error
        )

        return {
            "status": "ERROR",
            "symbol": symbol,
            "reason": str(error),
        }


# ============================================================
# PRICE FORMATTER
# ============================================================

def format_price(data):

    if data["status"] != "DATA_RECEIVED":

        return (
            f"❌ Data unavailable\n\n"
            f"Symbol: "
            f"{data.get('symbol')}\n"
            f"Reason: "
            f"{data.get('reason', 'Unknown')}"
        )

    return (
        "📊 *MARKET DATA*\n\n"
        f"Symbol: `{data['symbol']}`\n"
        f"Price: `{data['price']}`\n"
        f"Open: `{data['open']}`\n"
        f"High: `{data['high']}`\n"
        f"Low: `{data['low']}`\n"
        f"Time: `{data['timestamp']}`\n"
        f"Rows: `{data['rows']}`\n"
        f"Source: `{data['source']}`\n\n"
        "⚠️ This is research data and "
        "may not match an OTC broker feed."
    )


# ============================================================
# NEWS ENGINE
# ============================================================

def get_news(
    query="forex",
    limit=5,
):

    if not ALPHA_VANTAGE_API_KEY:

        return {
            "status": "NEWS_UNAVAILABLE",
            "reason": (
                "ALPHA_VANTAGE_API_KEY missing"
            ),
            "articles": [],
        }

    try:

        url = (
            "https://www.alphavantage.co/"
            "query"
        )

        params = {

            "function":
                "NEWS_SENTIMENT",

            "apikey":
                ALPHA_VANTAGE_API_KEY,

            "topics":
                query,

            "sort":
                "LATEST",

            "limit":
                limit,
        }

        response = requests.get(
            url,
            params=params,
            timeout=20,
        )

        response.raise_for_status()

        payload = response.json()

        if "feed" not in payload:

            return {
                "status":
                    "NEWS_UNAVAILABLE",
                "reason":
                    payload.get(
                        "Information",
                        payload.get(
                            "Note",
                            "No news feed"
                        )
                    ),
                "articles": [],
            }

        articles = []

        for item in payload["feed"]:

            articles.append({

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
                    ),
            })

        return {

            "status":
                "NEWS_RECEIVED",

            "articles":
                articles,
        }

    except Exception as error:

        logger.exception(
            "News error: %s",
            error
        )

        return {

            "status":
                "NEWS_ERROR",

            "reason":
                str(error),

            "articles":
                [],
        }


def format_news(data):

    if data["status"] != "NEWS_RECEIVED":

        return (
            "📰 *NEWS ENGINE*\n\n"
            "❌ News unavailable.\n"
            f"Reason: "
            f"{data.get('reason', 'Unknown')}"
        )

    text = (
        "📰 *LATEST MARKET NEWS*\n\n"
    )

    for index, article in enumerate(
        data["articles"],
        start=1
    ):

        text += (
            f"*{index}. "
            f"{article['title']}*\n"
            f"Source: "
            f"{article['source']}\n"
            f"Published: "
            f"{article['published']}\n\n"
        )

    return text


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(df):

    try:

        data = df.copy()

        close = data["Close"].astype(
            float
        )

        high = data["High"].astype(
            float
        )

        low = data["Low"].astype(
            float
        )

        # ----------------------------------------------------
        # EMA
        # ----------------------------------------------------

        data["EMA_20"] = (
            close.ewm(
                span=20,
                adjust=False
            ).mean()
        )

        data["EMA_50"] = (
            close.ewm(
                span=50,
                adjust=False
            ).mean()
        )

        data["EMA_200"] = (
            close.ewm(
                span=200,
                adjust=False
            ).mean()
        )

        # ----------------------------------------------------
        # SMA
        # ----------------------------------------------------

        data["SMA_20"] = (
            close.rolling(20).mean()
        )

        data["SMA_50"] = (
            close.rolling(50).mean()
        )

        data["SMA_200"] = (
            close.rolling(200).mean()
        )

        # ----------------------------------------------------
        # RSI 14 — Wilder style
        # ----------------------------------------------------

        delta = close.diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        avg_gain = gain.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        rs = (
            avg_gain
            / avg_loss.replace(
                0,
                np.nan
            )
        )

        data["RSI_14"] = (
            100
            - (
                100
                / (1 + rs)
            )
        )

        # ----------------------------------------------------
        # MACD
        # ----------------------------------------------------

        ema12 = close.ewm(
            span=12,
            adjust=False
        ).mean()

        ema26 = close.ewm(
            span=26,
            adjust=False
        ).mean()

        data["MACD"] = (
            ema12 - ema26
        )

        data["MACD_SIGNAL"] = (
            data["MACD"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
        )

        data["MACD_HIST"] = (
            data["MACD"]
            - data["MACD_SIGNAL"]
        )

        # ----------------------------------------------------
        # Bollinger Bands
        # ----------------------------------------------------

        bb_middle = (
            close.rolling(20).mean()
        )

        bb_std = (
            close.rolling(20).std()
        )

        data["BB_MIDDLE"] = (
            bb_middle
        )

        data["BB_UPPER"] = (
            bb_middle
            + (2 * bb_std)
        )

        data["BB_LOWER"] = (
            bb_middle
            - (2 * bb_std)
        )

        data["BB_WIDTH"] = (
            data["BB_UPPER"]
            - data["BB_LOWER"]
        )

        # ----------------------------------------------------
        # ATR
        # ----------------------------------------------------

        previous_close = close.shift(
            1
        )

        tr1 = high - low

        tr2 = (
            high - previous_close
        ).abs()

        tr3 = (
            low - previous_close
        ).abs()

        true_range = pd.concat(
            [
                tr1,
                tr2,
                tr3,
            ],
            axis=1
        ).max(axis=1)

        data["ATR_14"] = (
            true_range.ewm(
                alpha=1 / 14,
                adjust=False,
                min_periods=14
            ).mean()
        )

        # ----------------------------------------------------
        # ATR %
        # ----------------------------------------------------

        data["ATR_PERCENT"] = (
            data["ATR_14"]
            / close
        ) * 100

        # ----------------------------------------------------
        # Volatility
        # ----------------------------------------------------

        returns = close.pct_change()

        data["VOLATILITY_20"] = (
            returns.rolling(20).std()
            * np.sqrt(20)
        )

        # ----------------------------------------------------
        # Remove invalid numeric values
        # ----------------------------------------------------

        data.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        return data

    except Exception as error:

        logger.exception(
            "Indicator error: %s",
            error
        )

        return df


# ============================================================
# TREND STRENGTH
# ============================================================

def calculate_trend_strength(df):

    try:

        if len(df) < 50:

            return {
                "status":
                    "INSUFFICIENT_DATA",

                "score":
                    0,

                "direction":
                    "UNKNOWN",
            }

        row = df.iloc[-1]

        score = 0

        ema20 = row.get(
            "EMA_20"
        )

        ema50 = row.get(
            "EMA_50"
        )

        ema200 = row.get(
            "EMA_200"
        )

        rsi = row.get(
            "RSI_14"
        )

        macd = row.get(
            "MACD"
        )

        macd_signal = row.get(
            "MACD_SIGNAL"
        )

        # EMA 20/50

        if (
            pd.notna(ema20)
            and pd.notna(ema50)
        ):

            if ema20 > ema50:
                score += 1

            elif ema20 < ema50:
                score -= 1

        # EMA 50/200

        if (
            pd.notna(ema50)
            and pd.notna(ema200)
        ):

            if ema50 > ema200:
                score += 1

            elif ema50 < ema200:
                score -= 1

        # RSI

        if pd.notna(rsi):

            if rsi > 55:
                score += 1

            elif rsi < 45:
                score -= 1

        # MACD

        if (
            pd.notna(macd)
            and pd.notna(macd_signal)
        ):

            if macd > macd_signal:
                score += 1

            elif macd < macd_signal:
                score -= 1

        if score >= 3:

            direction = "BULLISH"

        elif score <= -3:

            direction = "BEARISH"

        else:

            direction = "NEUTRAL"

        return {

            "status":
                "OK",

            "score":
                score,

            "direction":
                direction,
        }

    except Exception as error:

        logger.exception(
            "Trend error: %s",
            error
        )

        return {

            "status":
                "ERROR",

            "score":
                0,

            "direction":
                "UNKNOWN",
        }


# ============================================================
# MARKET REGIME
# ============================================================

def determine_market_regime(df):

    try:

        if len(df) < 50:

            return {

                "regime":
                    "UNKNOWN",

                "reason":
                    "INSUFFICIENT_DATA",
            }

        row = df.iloc[-1]

        close = row.get(
            "Close"
        )

        ema20 = row.get(
            "EMA_20"
        )

        ema50 = row.get(
            "EMA_50"
        )

        rsi = row.get(
            "RSI_14"
        )

        atr_percent = row.get(
            "ATR_PERCENT"
        )

        if any(
            pd.isna(value)
            for value in [
                close,
                ema20,
                ema50,
                rsi,
            ]
        ):

            return {

                "regime":
                    "UNKNOWN",

                "reason":
                    "MISSING_INDICATOR_DATA",
            }

        trend_score = 0

        if ema20 > ema50:

            trend_score += 1

        elif ema20 < ema50:

            trend_score -= 1

        if rsi > 55:

            trend_score += 1

        elif rsi < 45:

            trend_score -= 1

        # ----------------------------------------------------
        # Regime
        # ----------------------------------------------------

        if (
            45 <= rsi <= 55
            and abs(trend_score) <= 1
        ):

            regime = "RANGING"

        elif trend_score >= 2:

            regime = "BULLISH_TREND"

        elif trend_score <= -2:

            regime = "BEARISH_TREND"

        else:

            regime = "TRANSITION"

        # ----------------------------------------------------
        # Volatility
        # ----------------------------------------------------

        volatility = "NORMAL"

        if pd.notna(
            atr_percent
        ):

            if atr_percent > 2:

                volatility = "HIGH"

            elif atr_percent < 0.5:

                volatility = "LOW"

        return {

            "regime":
                regime,

            "volatility":
                volatility,

            "trend_score":
                trend_score,
        }

    except Exception as error:

        logger.exception(
            "Regime error: %s",
            error
        )

        return {

            "regime":
                "UNKNOWN",

            "reason":
                "CALCULATION_ERROR",
        }


# ============================================================
# MULTI-TIMEFRAME CONFIG
# ============================================================

TIMEFRAMES = {

    "1m": {
        "period": "1d",
        "interval": "1m",
    },

    "5m": {
        "period": "5d",
        "interval": "5m",
    },

    "15m": {
        "period": "5d",
        "interval": "15m",
    },

    "30m": {
        "period": "1mo",
        "interval": "30m",
    },

    "1h": {
        "period": "1mo",
        "interval": "1h",
    },

    "2h": {
        "period": "3mo",
        "interval": "2h",
    },

    "4h": {
        "period": "6mo",
        "interval": "4h",
    },

    "1D": {
        "period": "2y",
        "interval": "1d",
    },
}


# ============================================================
# SINGLE TIMEFRAME ANALYSIS
# ============================================================

def analyze_timeframe(
    yahoo_symbol,
    timeframe,
):

    config = TIMEFRAMES.get(
        timeframe
    )

    if not config:

        return {

            "status":
                "INVALID_TIMEFRAME",

            "timeframe":
                timeframe,
        }

    try:

        df = yf.download(

            yahoo_symbol,

            period=config["period"],

            interval=config["interval"],

            progress=False,

            auto_adjust=False,

            threads=False,
        )

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = [
                column[0]
                for column in df.columns
            ]

        valid, reason = (
            validate_dataframe(df)
        )

        if not valid:

            return {

                "status":
                    "DATA_UNAVAILABLE",

                "timeframe":
                    timeframe,

                "reason":
                    reason,
            }

        df = calculate_indicators(
            df
        )

        regime = (
            determine_market_regime(
                df
            )
        )

        trend = (
            calculate_trend_strength(
                df
            )
        )

        latest = df.iloc[-1]

        def safe_float(
            value
        ):

            if pd.isna(value):
                return None

            return float(value)

        return {

            "status":
                "OK",

            "timeframe":
                timeframe,

            "timestamp":
                str(df.index[-1]),

            "close":
                safe_float(
                    latest["Close"]
                ),

            "ema20":
                safe_float(
                    latest["EMA_20"]
                ),

            "ema50":
                safe_float(
                    latest["EMA_50"]
                ),

            "ema200":
                safe_float(
                    latest["EMA_200"]
                ),

            "sma20":
                safe_float(
                    latest["SMA_20"]
                ),

            "sma50":
                safe_float(
                    latest["SMA_50"]
                ),

            "sma200":
                safe_float(
                    latest["SMA_200"]
                ),

            "rsi":
                safe_float(
                    latest["RSI_14"]
                ),

            "macd":
                safe_float(
                    latest["MACD"]
                ),

            "macd_signal":
                safe_float(
                    latest[
                        "MACD_SIGNAL"
                    ]
                ),

            "atr":
                safe_float(
                    latest["ATR_14"]
                ),

            "atr_percent":
                safe_float(
                    latest[
                        "ATR_PERCENT"
                    ]
                ),

            "regime":
                regime,

            "trend":
                trend,

            "rows":
                len(df),
        }

    except Exception as error:

        logger.exception(
            "MTF error: %s",
            error
        )

        return {

            "status":
                "ERROR",

            "timeframe":
                timeframe,

            "reason":
                str(error),
        }


# ============================================================
# MULTI-TIMEFRAME ANALYSIS
# ============================================================

def analyze_multi_timeframe(
    symbol
):

    symbol = symbol.upper().strip()

    yahoo_symbol = SYMBOL_MAP.get(
        symbol
    )

    if not yahoo_symbol:

        return {

            "status":
                "UNKNOWN_SYMBOL",

            "symbol":
                symbol,
        }

    results = {}

    for timeframe in TIMEFRAMES:

        results[timeframe] = (
            analyze_timeframe(
                yahoo_symbol,
                timeframe,
            )
        )

    return {

        "status":
            "COMPLETED",

        "symbol":
            symbol,

        "timeframes":
            results,
    }


# ============================================================
# TECHNICAL SUMMARY
# ============================================================

def create_technical_summary(
    analysis
):

    if (
        analysis.get("status")
        != "COMPLETED"
    ):

        return (
            "❌ Technical analysis "
            "unavailable."
        )

    text = (

        f"📊 *TECHNICAL ANALYSIS — "
        f"{analysis['symbol']}*\n\n"
    )

    for timeframe, result in (
        analysis["timeframes"].items()
    ):

        if result.get(
            "status"
        ) != "OK":

            text += (

                f"⏱ *{timeframe}*\n"
                f"❌ DATA UNAVAILABLE\n"
                f"Reason: "
                f"{result.get('reason')}\n\n"
            )

            continue

        regime = result[
            "regime"
        ]

        trend = result[
            "trend"
        ]

        text += (

            f"⏱ *{timeframe}*\n"

            f"Price: "
            f"`{result['close']}`\n"

            f"EMA20: "
            f"`{result['ema20']}`\n"

            f"EMA50: "
            f"`{result['ema50']}`\n"

            f"EMA200: "
            f"`{result['ema200']}`\n"

            f"RSI: "
            f"`{result['rsi']}`\n"

            f"MACD: "
            f"`{result['macd']}`\n"

            f"ATR: "
            f"`{result['atr']}`\n"

            f"Regime: "
            f"`{regime.get('regime')}`\n"

            f"Volatility: "
            f"`{regime.get('volatility', 'N/A')}`\n"

            f"Trend: "
            f"`{trend.get('direction', 'N/A')}`\n"

            f"Trend Score: "
            f"`{trend.get('score', 0)}`\n\n"
        )

    return text


# ============================================================
# TELEGRAM /START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    keyboard = [

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
                "📰 News",
                callback_data="news"
            ),

            InlineKeyboardButton(
                "📈 Analysis",
                callback_data="analysis"
            ),
        ],

        [
            InlineKeyboardButton(
                "🤖 AI Chat",
                callback_data="chat"
            ),

            InlineKeyboardButton(
                "📖 Guide",
                callback_data="guide"
            ),
        ],
    ]

    reply_markup = (
        InlineKeyboardMarkup(
            keyboard
        )
    )

    await update.message.reply_text(

        "🧠 *MASTER MARKET RESEARCH BOT*\n\n"

        "Step 2A + Step 2B Online.\n\n"

        "Available:\n"
        "• Market Data\n"
        "• News Engine\n"
        "• Technical Indicators\n"
        "• Multi-Timeframe Analysis\n"
        "• Market Regime\n"
        "• Gemini AI\n\n"

        "⚠️ Research / educational mode.\n"
        "No real-money trade execution.",

        parse_mode="Markdown",

        reply_markup=reply_markup,
    )


# ============================================================
# /PRICE
# ============================================================

async def price_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not context.args:

        await update.message.reply_text(

            "Example:\n\n"

            "/price EURUSD\n"
            "/price XAUUSD\n"
            "/price BTCUSD"
        )

        return

    symbol = (
        context.args[0]
        .upper()
    )

    await update.message.reply_text(

        "🔎 Fetching market data..."
    )

    data = get_market_data(

        symbol=symbol,

        period="5d",

        interval="1h",
    )

    await update.message.reply_text(

        format_price(data),

        parse_mode="Markdown",
    )


# ============================================================
# /ANALYZE
# ============================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not context.args:

        await update.message.reply_text(

            "Example:\n\n"
            "/analyze EURUSD\n"
            "/analyze XAUUSD\n"
            "/analyze BTCUSD"
        )

        return

    symbol = (
        context.args[0]
        .upper()
    )

    await update.message.reply_text(

        f"🔬 Analyzing `{symbol}` "
        f"across multiple timeframes...",
        parse_mode="Markdown",
    )

    analysis = (
        analyze_multi_timeframe(
            symbol
        )
    )

    text = (
        create_technical_summary(
            analysis
        )
    )

    await update.message.reply_text(

        text,

        parse_mode="Markdown",
    )


# ============================================================
# /NEWS
# ============================================================

async def news_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = "forex"

    if context.args:

        query = " ".join(
            context.args
        )

    await update.message.reply_text(

        "📰 Fetching latest news..."
    )

    data = get_news(
        query=query,
        limit=5,
    )

    await update.message.reply_text(

        format_news(data),

        parse_mode="Markdown",
    )


# ============================================================
# BUTTON HANDLER
# ============================================================

async def button_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    action = query.data

    # --------------------------------------------------------
    # WATCHLIST
    # --------------------------------------------------------

    if action == "watchlist":

        text = (
            "📊 *WATCHLIST*\n\n"
        )

        for symbol in WATCHLIST:

            text += (
                f"• `{symbol}`\n"
            )

        text += (
            "\nUse:\n"
            "`/price EURUSD`\n"
            "`/analyze EURUSD`"
        )

        await query.edit_message_text(

            text,

            parse_mode="Markdown",
        )

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    elif action == "health":

        text = (

            "❤️ *SYSTEM HEALTH*\n\n"

            "Telegram: `ONLINE`\n"

            f"Gemini: "
            f"`{'ONLINE' if gemini_client else 'OFFLINE'}`\n"

            f"News Engine: "
            f"`{'ONLINE' if ALPHA_VANTAGE_API_KEY else 'OFFLINE'}`\n"

            "Market Data: `CONNECTED`\n"

            "Technical Engine: `ONLINE`\n"

            "Trading Execution: `DISABLED`"
        )

        await query.edit_message_text(

            text,

            parse_mode="Markdown",
        )

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    elif action == "news":

        data = get_news(
            query="forex",
            limit=5,
        )

        await query.edit_message_text(

            format_news(data),

            parse_mode="Markdown",
        )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    elif action == "analysis":

        await query.edit_message_text(

            "📈 *TECHNICAL ANALYSIS*\n\n"

            "Use commands:\n\n"

            "`/analyze EURUSD`\n"
            "`/analyze XAUUSD`\n"
            "`/analyze BTCUSD`\n\n"

            "The engine checks multiple "
            "timeframes and indicators.",

            parse_mode="Markdown",
        )

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    elif action == "chat":

        await query.edit_message_text(

            "🤖 *AI CHAT*\n\n"

            "Send a message such as:\n\n"

            "`Explain RSI`\n"
            "`Explain MACD`\n"
            "`What is market regime?`\n"
            "`Explain liquidity`\n\n"

            "AI answers are educational "
            "and do not represent guaranteed "
            "future market outcomes.",

            parse_mode="Markdown",
        )

    # --------------------------------------------------------
    # GUIDE
    # --------------------------------------------------------

    elif action == "guide":

        await query.edit_message_text(

            "📖 *BOT GUIDE*\n\n"

            "/price SYMBOL\n"
            "→ Market data\n\n"

            "/analyze SYMBOL\n"
            "→ Multi-timeframe technical "
            "analysis\n\n"

            "/news forex\n"
            "→ Latest market news\n\n"

            "Supported examples:\n"
            "EURUSD\n"
            "GBPUSD\n"
            "USDJPY\n"
            "XAUUSD\n"
            "BTCUSD\n"
            "ETHUSD",

            parse_mode="Markdown",
        )


# ============================================================
# GEMINI CHAT
# ============================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    user_text = (
        update.message.text
    )

    if not gemini_client:

        await update.message.reply_text(

            "❌ Gemini AI is currently "
            "unavailable."
        )

        return

    system_prompt = (

        "You are an educational "
        "market research assistant. "

        "Explain trading concepts, "
        "technical analysis, market "
        "structure, indicators, ICT/SMC "
        "concepts and financial concepts "
        "clearly. "

        "Do not claim guaranteed accuracy. "

        "Do not pretend to have live market "
        "data unless it is actually provided "
        "by the data engine. "

        "Do not provide instructions for "
        "real-money gambling or binary "
        "options trading."
    )

    prompt = (
        system_prompt
        + "\n\nUser question:\n"
        + user_text
    )

    try:

        response = (
            await gemini_client.aio.models
            .generate_content(

                model=GEMINI_MODEL,

                contents=prompt,
            )
        )

        answer = (
            response.text
            if response.text
            else "No response."
        )

        await update.message.reply_text(
            answer
        )

    except Exception as error:

        logger.exception(
            "Gemini error: %s",
            error
        )

        await update.message.reply_text(

            "❌ AI request failed.\n"
            "Please try again later."
        )


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):

    logger.exception(
        "Telegram error:",
        exc_info=context.error
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Start Flask in background
    flask_thread = threading.Thread(

        target=run_flask,

        daemon=True,
    )

    flask_thread.start()

    logger.info(
        "Health server started."
    )

    # Telegram application
    application = (
        Application.builder()
        .token(
            TELEGRAM_BOT_TOKEN
        )
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

    # AI Chat
    application.add_handler(

        MessageHandler(

            filters.TEXT
            & ~filters.COMMAND,

            handle_message,
        )
    )

    # Errors
    application.add_error_handler(
        error_handler
    )

    logger.info(
        "Telegram bot starting..."
    )

    application.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

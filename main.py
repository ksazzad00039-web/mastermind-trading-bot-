import os
import logging
import threading
from datetime import datetime, timezone

import requests
import pandas as pd
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
    ContextTypes,
    MessageHandler,
    filters,
)

from google import genai


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

PORT = int(os.getenv("PORT", "10000"))

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("MastermindBot")


# ============================================================
# ENV VALIDATION
# ============================================================

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing.")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing.")


# ============================================================
# GEMINI
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# FLASK HEALTH SERVER
# ============================================================

flask_app = Flask(__name__)


@flask_app.route("/")
def home():

    return {
        "service": "Mastermind Market Intelligence Engine",
        "status": "online",

        "telegram": "configured",
        "gemini": "configured",

        "market_data": "active",
        "news_engine": (
            "configured"
            if ALPHA_VANTAGE_API_KEY
            else "not_configured"
        ),

        "analysis_engine": "step_2_foundation",

        "trading_execution": "disabled",
    }


@flask_app.route("/health")
def health():

    return {
        "status": "healthy",

        "telegram": bool(TELEGRAM_BOT_TOKEN),
        "gemini": bool(GEMINI_API_KEY),

        "market_data": True,

        "news": bool(ALPHA_VANTAGE_API_KEY),

        "execution": False,
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
    "EUR/USD": "EURUSD=X",

    "GBPUSD": "GBPUSD=X",
    "GBP/USD": "GBPUSD=X",

    "USDJPY": "JPY=X",
    "USD/JPY": "JPY=X",

    "USDCHF": "CHF=X",
    "USD/CHF": "CHF=X",

    "AUDUSD": "AUDUSD=X",
    "AUD/USD": "AUDUSD=X",

    "USDCAD": "CAD=X",
    "USD/CAD": "CAD=X",

    "NZDUSD": "NZDUSD=X",
    "NZD/USD": "NZDUSD=X",

    "GOLD": "GC=F",
    "XAUUSD": "GC=F",
    "XAU/USD": "GC=F",

    "BTCUSD": "BTC-USD",
    "BTC/USD": "BTC-USD",

    "ETHUSD": "ETH-USD",
    "ETH/USD": "ETH-USD",
}


# ============================================================
# WATCHLIST
# ============================================================

WATCHLIST = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD",
    "XAU/USD",
    "BTC/USD",
    "ETH/USD",
]


# ============================================================
# DATA QUALITY CHECK
# ============================================================

def validate_dataframe(df):

    if df is None:
        return False, "DATAFRAME_NONE"

    if df.empty:
        return False, "EMPTY_DATA"

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for column in required:

        if column not in df.columns:
            return False, f"MISSING_{column}"

    if df[required].isnull().any().any():
        return False, "NULL_VALUES"

    if len(df) < 2:
        return False, "INSUFFICIENT_DATA"

    if not df.index.is_monotonic_increasing:
        return False, "INVALID_TIME_ORDER"

    return True, "OK"


# ============================================================
# MARKET DATA ENGINE
# ============================================================

def get_market_data(symbol, period="5d", interval="1h"):

    yahoo_symbol = SYMBOL_MAP.get(
        symbol.upper().strip()
    )

    if not yahoo_symbol:

        return {
            "status": "DATA_UNAVAILABLE",
            "reason": "UNKNOWN_SYMBOL",
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

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = [
                column[0]
                for column in df.columns
            ]

        valid, reason = validate_dataframe(df)

        if not valid:

            return {
                "status": "DATA_UNAVAILABLE",
                "reason": reason,
                "symbol": symbol,
            }

        latest = df.iloc[-1]

        timestamp = df.index[-1]

        return {
            "status": "DATA_RECEIVED",
            "symbol": symbol,
            "source": "Yahoo Finance",
            "provider_symbol": yahoo_symbol,

            "timestamp": str(timestamp),

            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "close": float(latest["Close"]),

            "rows": len(df),
        }

    except Exception as error:

        logger.exception(
            "Market data error: %s",
            error,
        )

        return {
            "status": "DATA_UNAVAILABLE",
            "reason": "FETCH_ERROR",
            "symbol": symbol,
        }


# ============================================================
# NEWS ENGINE
# ============================================================

def get_news(query="forex"):

    if not ALPHA_VANTAGE_API_KEY:

        return {
            "status": "NEWS_UNAVAILABLE",
            "reason": "API_KEY_NOT_CONFIGURED",
            "items": [],
        }

    url = (
        "https://www.alphavantage.co/query"
    )

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": query,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15,
        )

        response.raise_for_status()

        data = response.json()

        feed = data.get("feed", [])

        items = []

        for item in feed[:10]:

            items.append({

                "title": item.get(
                    "title",
                    "Unknown",
                ),

                "source": item.get(
                    "source",
                    "Unknown",
                ),

                "published": item.get(
                    "time_published",
                    "Unknown",
                ),

                "url": item.get(
                    "url",
                    "",
                ),

                "summary": item.get(
                    "summary",
                    "",
                ),

            })

        return {
            "status": "NEWS_RECEIVED",
            "items": items,
        }

    except Exception as error:

        logger.exception(
            "News error: %s",
            error,
        )

        return {
            "status": "NEWS_UNAVAILABLE",
            "reason": "FETCH_ERROR",
            "items": [],
        }


# ============================================================
# FORMAT PRICE
# ============================================================

def format_price(data):

    if data["status"] != "DATA_RECEIVED":

        return (
            "❌ *DATA UNAVAILABLE*\n\n"
            f"Symbol: {data.get('symbol')}\n"
            f"Reason: {data.get('reason')}\n\n"
            "No price was invented."
        )

    return (
        "📊 *MARKET DATA*\n\n"

        f"Symbol: `{data['symbol']}`\n"

        f"Price: `{data['close']}`\n"

        f"Open: `{data['open']}`\n"
        f"High: `{data['high']}`\n"
        f"Low: `{data['low']}`\n\n"

        f"Time: `{data['timestamp']}`\n"

        f"Source: `{data['source']}`\n"

        f"Rows: `{data['rows']}`\n\n"

        "⚠️ Data source is Yahoo Finance. "
        "It is not an exact OTC broker feed."
    )


# ============================================================
# /START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    keyboard = [

        [
            InlineKeyboardButton(
                "📊 Watchlist",
                callback_data="watchlist",
            ),

            InlineKeyboardButton(
                "🩺 Health",
                callback_data="health",
            ),
        ],

        [
            InlineKeyboardButton(
                "📰 News",
                callback_data="news",
            ),

            InlineKeyboardButton(
                "📘 Guide",
                callback_data="guide",
            ),
        ],
    ]

    text = (
        "🧠 *MASTERMIND MARKET INTELLIGENCE*\n\n"

        "Step 2A is active.\n\n"

        "✅ Telegram Core\n"
        "✅ Market Data Engine\n"
        "✅ Data Validation\n"
        "✅ News Engine\n"
        "✅ Gemini AI\n\n"

        "⏳ ICT/SMC Engine\n"
        "⏳ Multi-Timeframe Engine\n"
        "⏳ Backtesting Engine\n\n"

        "Select an option."
    )

    await update.message.reply_text(
        text,
        reply_markup=InlineKeyboardMarkup(
            keyboard
        ),
        parse_mode="Markdown",
    )


# ============================================================
# /PRICE COMMAND
# ============================================================

async def price_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not context.args:

        await update.message.reply_text(
            "Example:\n\n"
            "`/price EURUSD`\n"
            "`/price XAUUSD`\n"
            "`/price BTCUSD`",
            parse_mode="Markdown",
        )

        return

    symbol = context.args[0]

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
# /NEWS COMMAND
# ============================================================

async def news_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = "forex"

    if context.args:

        query = context.args[0]

    await update.message.reply_text(
        "📰 Fetching available news..."
    )

    result = get_news(query)

    if result["status"] != "NEWS_RECEIVED":

        await update.message.reply_text(
            "❌ *NEWS UNAVAILABLE*\n\n"
            f"Reason: `{result['reason']}`\n\n"
            "No news has been invented.",
            parse_mode="Markdown",
        )

        return

    items = result["items"]

    if not items:

        await update.message.reply_text(
            "ℹ️ No news items returned."
        )

        return

    text = "📰 *LATEST AVAILABLE NEWS*\n\n"

    for index, item in enumerate(
        items[:5],
        start=1,
    ):

        title = item["title"]

        source = item["source"]

        published = item["published"]

        text += (
            f"*{index}.* {title}\n"
            f"Source: {source}\n"
            f"Published: {published}\n\n"
        )

    await update.message.reply_text(
        text,
        parse_mode="Markdown",
    )


# ============================================================
# WATCHLIST BUTTON
# ============================================================

async def show_watchlist(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    text = (
        "📊 *WATCHLIST*\n\n"
        + "\n".join(
            f"• {symbol}"
            for symbol in WATCHLIST
        )
    )

    keyboard = [

        [
            InlineKeyboardButton(
                "⬅️ Back",
                callback_data="back",
            )
        ]

    ]

    await query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(
            keyboard
        ),
        parse_mode="Markdown",
    )


# ============================================================
# HEALTH BUTTON
# ============================================================

async def show_health(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    news_status = (
        "CONFIGURED"
        if ALPHA_VANTAGE_API_KEY
        else "NOT CONFIGURED"
    )

    text = (
        "🩺 *SYSTEM HEALTH*\n\n"

        "🟢 Telegram: ONLINE\n"
        "🟢 Flask: ONLINE\n"
        "🟢 Gemini: CONFIGURED\n"
        "🟢 Market Data: ACTIVE\n"

        f"🟡 News: {news_status}\n"

        "🟡 ICT/SMC: NOT ACTIVE\n"
        "🟡 Backtesting: NOT ACTIVE\n"
        "🟡 Paper Trading: NOT ACTIVE\n\n"

        "🔒 Real-money execution: DISABLED"
    )

    keyboard = [

        [
            InlineKeyboardButton(
                "🔄 Refresh",
                callback_data="health",
            ),

            InlineKeyboardButton(
                "⬅️ Back",
                callback_data="back",
            ),
        ]

    ]

    await query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(
            keyboard
        ),
        parse_mode="Markdown",
    )


# ============================================================
# NEWS BUTTON
# ============================================================

async def show_news(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    result = get_news("forex")

    if result["status"] != "NEWS_RECEIVED":

        text = (
            "❌ *NEWS ENGINE*\n\n"
            "News data is currently unavailable.\n\n"
            f"Reason: {result['reason']}"
        )

    else:

        items = result["items"]

        text = "📰 *NEWS ENGINE*\n\n"

        for item in items[:5]:

            text += (
                f"• {item['title']}\n"
                f"  Source: {item['source']}\n"
                f"  Time: {item['published']}\n\n"
            )

    keyboard = [

        [
            InlineKeyboardButton(
                "⬅️ Back",
                callback_data="back",
            )
        ]

    ]

    await query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(
            keyboard
        ),
        parse_mode="Markdown",
    )


# ============================================================
# GUIDE
# ============================================================

async def show_guide(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    text = (
        "📘 *STEP 2 ROADMAP*\n\n"

        "1️⃣ Data Integrity\n"
        "2️⃣ Market Data\n"
        "3️⃣ News Intelligence\n"
        "4️⃣ Multi-Timeframe\n"
        "5️⃣ Indicators\n"
        "6️⃣ Market Structure\n"
        "7️⃣ ICT/SMC Research Layer\n"
        "8️⃣ Statistical Engine\n"
        "9️⃣ Backtesting\n"
        "🔟 Paper Trading\n\n"

        "The system will never create a "
        "price or news item when reliable data "
        "is unavailable."
    )

    keyboard = [

        [
            InlineKeyboardButton(
                "⬅️ Back",
                callback_data="back",
            )
        ]

    ]

    await query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(
            keyboard
        ),
        parse_mode="Markdown",
    )


# ============================================================
# BACK
# ============================================================

async def back_to_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    keyboard = [

        [
            InlineKeyboardButton(
                "📊 Watchlist",
                callback_data="watchlist",
            ),

            InlineKeyboardButton(
                "🩺 Health",
                callback_data="health",
            ),
        ],

        [
            InlineKeyboardButton(
                "📰 News",
                callback_data="news",
            ),

            InlineKeyboardButton(
                "📘 Guide",
                callback_data="guide",
            ),
        ],

    ]

    await query.edit_message_text(
        "🧠 *MASTERMIND MARKET INTELLIGENCE*\n\n"
        "Choose an option.",
        reply_markup=InlineKeyboardMarkup(
            keyboard
        ),
        parse_mode="Markdown",
    )


# ============================================================
# GEMINI CHAT
# ============================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not update.message:
        return

    user_text = update.message.text.strip()

    if not user_text:
        return

    await update.message.chat.send_action(
        "typing"
    )

    prompt = f"""
You are the educational AI assistant
inside the Mastermind Market Intelligence
research system.

You may explain:

- market structure
- price action
- ICT concepts
- SMC concepts
- indicators
- trading psychology
- risk management
- statistics
- backtesting
- financial-market concepts

Rules:

1. Never invent market data.
2. Never claim live data unless it is actually supplied.
3. Never guarantee future prices.
4. Never claim guaranteed profit.
5. Clearly distinguish facts from interpretation.
6. Explain uncertainty when evidence is incomplete.

User question:

{user_text}
"""

    try:

        response = await (
            gemini_client
            .aio
            .models
            .generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        )

        answer = response.text

        if not answer:

            answer = (
                "No response was generated."
            )

        await update.message.reply_text(
            answer
        )

    except Exception as error:

        logger.exception(
            "Gemini error: %s",
            error,
        )

        await update.message.reply_text(
            "⚠️ AI service is temporarily unavailable."
        )


# ============================================================
# BUTTON ROUTER
# ============================================================

async def button_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    if query.data == "watchlist":

        await show_watchlist(
            update,
            context,
        )

    elif query.data == "health":

        await show_health(
            update,
            context,
        )

    elif query.data == "news":

        await show_news(
            update,
            context,
        )

    elif query.data == "guide":

        await show_guide(
            update,
            context,
        )

    elif query.data == "back":

        await back_to_menu(
            update,
            context,
        )


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):

    logger.exception(
        "Unhandled error:",
        exc_info=context.error,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting Mastermind Bot..."
    )

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True,
    )

    flask_thread.start()

    application = (
        Application
        .builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    application.add_handler(
        CommandHandler(
            "price",
            price_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "news",
            news_command,
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            button_handler,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            handle_message,
        )
    )

    application.add_error_handler(
        error_handler
    )

    logger.info(
        "Telegram bot is running..."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

import os
import logging
import threading

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
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PORT = int(os.getenv("PORT", "10000"))

# Use a current Gemini model available to your API account.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash"
)


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("MastermindBot")


# ============================================================
# 3. BASIC VALIDATION
# ============================================================

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN is missing from environment variables."
    )

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from environment variables."
    )


# ============================================================
# 4. GEMINI CLIENT
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 5. FLASK HEALTH SERVER
# ============================================================

flask_app = Flask(__name__)


@flask_app.route("/")
def home():
    return {
        "status": "online",
        "service": "Mastermind Research Bot",
        "telegram": "configured",
        "gemini": "configured",
        "market_data": "not_connected",
        "analysis_engine": "not_active",
        "paper_trading": "not_active",
    }


@flask_app.route("/health")
def health():
    return {
        "status": "healthy",
        "telegram": bool(TELEGRAM_BOT_TOKEN),
        "gemini": bool(GEMINI_API_KEY),
    }


def run_flask():
    """Run Flask in a separate thread."""
    flask_app.run(
        host="0.0.0.0",
        port=PORT,
        use_reloader=False,
    )


# ============================================================
# 6. WATCHLIST
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
# 7. /START
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    keyboard = [
        [
            InlineKeyboardButton(
                "📊 Watchlist",
                callback_data="watchlist"
            ),
            InlineKeyboardButton(
                "🩺 System Health",
                callback_data="health"
            ),
        ],
        [
            InlineKeyboardButton(
                "📘 Bot Guide",
                callback_data="guide"
            ),
            InlineKeyboardButton(
                "🤖 AI Chat",
                callback_data="chat"
            ),
        ],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    text = (
        "🧠 *MASTERMIND RESEARCH BOT*\n\n"
        "Welcome!\n\n"
        "This is Step 1 of the research system.\n\n"
        "✅ Telegram Core: ACTIVE\n"
        "✅ Gemini AI: CONFIGURED\n"
        "⏳ Market Data: NOT CONNECTED\n"
        "⏳ Analysis Engine: NOT ACTIVE\n"
        "⏳ Backtesting: NOT ACTIVE\n"
        "⏳ Paper Trading: NOT ACTIVE\n\n"
        "Choose an option below."
    )

    await update.message.reply_text(
        text,
        reply_markup=reply_markup,
        parse_mode="Markdown",
    )


# ============================================================
# 8. WATCHLIST
# ============================================================

async def show_watchlist(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query
    await query.answer()

    symbols = "\n".join(
        f"• {symbol}" for symbol in WATCHLIST
    )

    text = (
        "📊 *CURRENT WATCHLIST*\n\n"
        f"{symbols}\n\n"
        "⚠️ These are watchlist labels only.\n"
        "No live market data is connected yet."
    )

    keyboard = [
        [
            InlineKeyboardButton(
                "⬅️ Back",
                callback_data="back"
            )
        ]
    ]

    await query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )


# ============================================================
# 9. SYSTEM HEALTH
# ============================================================

async def show_health(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query
    await query.answer()

    text = (
        "🩺 *SYSTEM HEALTH*\n\n"
        "🟢 Telegram Core: ONLINE\n"
        "🟢 Flask Server: ONLINE\n"
        "🟢 Gemini Client: CONFIGURED\n"
        "🟡 Market Data Engine: NOT CONNECTED\n"
        "🟡 Analysis Engine: NOT ACTIVE\n"
        "🟡 News Engine: NOT CONNECTED\n"
        "🟡 Backtesting Engine: NOT ACTIVE\n"
        "🟡 Paper Trading: NOT ACTIVE\n\n"
        "System foundation is ready."
    )

    keyboard = [
        [
            InlineKeyboardButton(
                "🔄 Refresh",
                callback_data="health"
            ),
            InlineKeyboardButton(
                "⬅️ Back",
                callback_data="back"
            ),
        ]
    ]

    await query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )


# ============================================================
# 10. BOT GUIDE
# ============================================================

async def show_guide(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query
    await query.answer()

    text = (
        "📘 *BOT GUIDE*\n\n"
        "*Step 1 — Core*\n"
        "Telegram + Gemini + Health Server\n\n"
        "*Step 2 — Data*\n"
        "Historical/market data pipeline\n\n"
        "*Step 3 — Analysis*\n"
        "Indicators + market structure + statistics\n\n"
        "*Step 4 — Context*\n"
        "Sessions, volatility, news and correlations\n\n"
        "*Step 5 — Backtesting*\n"
        "Historical testing with no look-ahead\n\n"
        "*Step 6 — Paper Trading*\n"
        "Simulation and performance tracking\n\n"
        "⚠️ No real-money execution is included."
    )

    keyboard = [
        [
            InlineKeyboardButton(
                "⬅️ Back",
                callback_data="back"
            )
        ]
    ]

    await query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )


# ============================================================
# 11. AI CHAT INFO
# ============================================================

async def show_chat_info(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query
    await query.answer()

    text = (
        "🤖 *AI CHAT*\n\n"
        "Send me a normal text message and Gemini "
        "will answer.\n\n"
        "Example:\n"
        "• Explain RSI\n"
        "• What is market structure?\n"
        "• Explain support and resistance\n\n"
        "The AI does not have live market data yet."
    )

    keyboard = [
        [
            InlineKeyboardButton(
                "⬅️ Back",
                callback_data="back"
            )
        ]
    ]

    await query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )


# ============================================================
# 12. BACK BUTTON
# ============================================================

async def back_to_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query
    await query.answer()

    keyboard = [
        [
            InlineKeyboardButton(
                "📊 Watchlist",
                callback_data="watchlist"
            ),
            InlineKeyboardButton(
                "🩺 System Health",
                callback_data="health"
            ),
        ],
        [
            InlineKeyboardButton(
                "📘 Bot Guide",
                callback_data="guide"
            ),
            InlineKeyboardButton(
                "🤖 AI Chat",
                callback_data="chat"
            ),
        ],
    ]

    text = (
        "🧠 *MASTERMIND RESEARCH BOT*\n\n"
        "Choose an option below."
    )

    await query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )


# ============================================================
# 13. GEMINI AI CHAT
# ============================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()

    if not user_text:
        return

    await update.message.chat.send_action("typing")

    prompt = f"""
You are the educational AI assistant inside
the Mastermind Research Bot.

Important rules:

1. Give educational explanations.
2. Do not claim to have live market data unless
   live data is actually provided.
3. Do not guarantee future market direction.
4. Do not claim guaranteed accuracy or profit.
5. Explain technical concepts clearly.
6. If information is missing, say so.

User message:

{user_text}
"""

    try:

        response = await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer = response.text

        if not answer:
            answer = (
                "I couldn't generate a response right now."
            )

        await update.message.reply_text(answer)

    except Exception as error:

        logger.exception(
            "Gemini request failed: %s",
            error
        )

        await update.message.reply_text(
            "⚠️ AI service is temporarily unavailable. "
            "Please try again later."
        )


# ============================================================
# 14. BUTTON HANDLER
# ============================================================

async def button_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    if query.data == "watchlist":
        await show_watchlist(update, context)

    elif query.data == "health":
        await show_health(update, context)

    elif query.data == "guide":
        await show_guide(update, context)

    elif query.data == "chat":
        await show_chat_info(update, context)

    elif query.data == "back":
        await back_to_menu(update, context)

    else:
        await query.answer(
            "Unknown option.",
            show_alert=True
        )


# ============================================================
# 15. GLOBAL ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    logger.exception(
        "Unhandled Telegram error:",
        exc_info=context.error,
    )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    logger.info("Starting Mastermind Research Bot...")

    # Start Flask health server
    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True,
    )

    flask_thread.start()

    logger.info(
        "Health server started on port %s",
        PORT
    )

    # Create Telegram application
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    # Handlers
    application.add_handler(
        CommandHandler("start", start)
    )

    application.add_handler(
        CallbackQueryHandler(button_handler)
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    application.add_error_handler(
        error_handler
    )

    logger.info("Telegram bot is starting...")

    # Start polling
    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


# ============================================================
# 17. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

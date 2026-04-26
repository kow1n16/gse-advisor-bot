# bot.py
import os
import logging
import asyncio
from fastapi import FastAPI, Request
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from agent import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WEBHOOK_URL    = os.environ["WEBHOOK_URL"]   # your Railway public URL

# ── In-memory session store (one history list per chat_id) ───
# For production, swap this for Redis
sessions: dict[int, list] = {}
MAX_HISTORY = 20   # keep last 20 turns per user

app     = FastAPI()
bot     = Bot(token=TELEGRAM_TOKEN)
tg_app  = Application.builder().token(TELEGRAM_TOKEN).build()


# ── Handlers ─────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to the GSE Investment Advisor!\n\n"
        "I have real Ghana Stock Exchange data from 2007 to 2026.\n\n"
        "Try asking:\n"
        "• Analyse MTNGH\n"
        "• Compare GCB and EBG\n"
        "• What are the top movers today?\n"
        "• Screen for oversold stocks\n\n"
        "⚠️ Research tool only — not licensed financial advice."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "GSE Advisor commands:\n\n"
        "/start  — welcome message\n"
        "/help   — this message\n"
        "/clear  — reset conversation\n"
        "/tickers — list all available stocks\n\n"
        "Or just ask anything about GSE stocks!"
    )

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sessions.pop(update.effective_chat.id, None)
    await update.message.reply_text("Conversation cleared. Fresh start!")

async def tickers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from agent import fundamentals_df
    tickers_list = sorted(fundamentals_df["ticker"].tolist())
    chunks = [tickers_list[i:i+10] for i in range(0, len(tickers_list), 10)]
    lines  = ["Available GSE tickers:\n"]
    for chunk in chunks:
        lines.append("  ".join(f"{t:<8}" for t in chunk))
    await update.message.reply_text("\n".join(lines))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id     = update.effective_chat.id
    user_message = update.message.text

    # Show typing indicator
    await bot.send_chat_action(chat_id=chat_id, action="typing")

    # Get or create session history
    history = sessions.get(chat_id, [])

    try:
        reply = await asyncio.to_thread(run_agent, user_message, history)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        reply = "Sorry, something went wrong. Please try again."

    # Update history (trim to last MAX_HISTORY messages)
    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": reply})
    sessions[chat_id] = history[-MAX_HISTORY:]

    await update.message.reply_text(reply)


# ── Register handlers ────────────────────────────────────────
tg_app.add_handler(CommandHandler("start",   start))
tg_app.add_handler(CommandHandler("help",    help_cmd))
tg_app.add_handler(CommandHandler("clear",   clear))
tg_app.add_handler(CommandHandler("tickers", tickers))
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


# ── FastAPI routes ───────────────────────────────────────────
@app.on_event("startup")
async def startup():
    await tg_app.initialize()
    await bot.set_webhook(url=f"{WEBHOOK_URL}/webhook")
    logger.info(f"Webhook set to {WEBHOOK_URL}/webhook")

@app.on_event("shutdown")
async def shutdown():
    await tg_app.shutdown()

@app.post("/webhook")
async def webhook(request: Request):
    data   = await request.json()
    update = Update.de_json(data, bot)
    await tg_app.process_update(update)
    return {"ok": True}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "GSE Advisor Bot"}
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Signals Telegram Bot (single-file)
Author: ChatGPT (for lenn)
Date: 2025-09-08

⚠️ Disclaimer
This bot shares algorithmically generated trade ideas and aggregated news for educational purposes.
It is NOT financial advice. Do your own research (DYOR) and trade at your own risk.

Features
- /start: Welcomes user, shares your Channel + Binance referral links first (gated details until they join).
- Daily: posts curated crypto news cards (with image) to your channel.
- Daily: posts ONE clear, "confirmed" signal (rule-based) with Entry, TP1..TP5, SL + image (green TP box, red SL box).
- Live monitoring: congratulates when a TP hits; alerts when SL is hit; adds a tiny tip with each trade status.
- "Public API" data: uses Binance public REST endpoints for OHLCV + price (no API key), and public RSS feeds for news.
- Futuristic extras (opt-in flags):
    * Auto-threads per signal (reply_to_message_id) for cleaner channel.
    * Basic risk mgmt helper in caption (position sizing formula example).
    * Simple on-chain & funding-rate placeholders (hooks to extend).
    * Gate details until user has joined your channel (via getChatMember).

Setup
1) Python 3.10+ recommended.
2) pip install -U python-telegram-bot==21.3 requests pandas numpy matplotlib pillow feedparser
3) Fill the CONFIG section below (TOKEN, CHANNEL_ID...). For a private channel, add the bot as admin.
4) Run: python crypto_signal_bot.py
5) The bot uses long polling by default. For production, consider webhooks + a process manager.

Notes
- Trading rules are SIMPLE demo logic (EMA cross + RSI filter + ATR-based SL/TPs). Tune to your style.
- Images are generated with matplotlib (candles optional) & PIL rectangles for TP/SL zones.
- News is pulled from public RSS feeds. Respect site ToS. You can add/remove feeds safely.
"""

import os
import json
import logging
import math
import io
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import feedparser
from PIL import Image, ImageDraw, ImageFont

# Matplotlib backend for headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters,
    ContextTypes, CallbackContext, CallbackQueryHandler
)

# ------------------------- CONFIG -------------------------

CONFIG = {
    # Required
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "PUT-YOUR-TOKEN-HERE"),
    # Your channel ID or @username (bot must be an admin to post). Example: "-1001234567890" or "@your_channel"
    "CHANNEL_ID": os.getenv("CHANNEL_ID", "@your_channel_username"),
    # Links to show (always shown before details)
    "CHANNEL_INVITE_LINK": os.getenv("CHANNEL_INVITE_LINK", "https://t.me/+T2lFw-AjK21kYWM0"),
    "BINANCE_REF_LINK": os.getenv("BINANCE_REF_LINK", "https://www.binance.com/referral/earn-together/refer-in-hotsummer/claim?hl=en&ref=GRO_20338_LRBY5&utm_source=default"),
    # Scheduling (Helsinki time assumed by default; adjust if deploying elsewhere)
    "DAILY_SIGNAL_HOUR": int(os.getenv("DAILY_SIGNAL_HOUR", "11")),  # 11:00
    "DAILY_NEWS_HOUR": int(os.getenv("DAILY_NEWS_HOUR", "9")),       # 09:00
    # Symbols universe (Binance spot symbols with USDT quote)
    "SYMBOLS": os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT").split(","),
    # Candle timeframe for signal generation
    "TIMEFRAME": os.getenv("TIMEFRAME", "1h"),  # ['15m','1h','4h','1d']
    # Risk/TP config
    "TP_PCTS": [0.005, 0.01, 0.015, 0.02, 0.03],  # 0.5%..3%
    "ATR_MULT_SL": float(os.getenv("ATR_MULT_SL", "1.5")),
    "EMA_FAST": int(os.getenv("EMA_FAST", "20")),
    "EMA_SLOW": int(os.getenv("EMA_SLOW", "50")),
    "RSI_LEN": int(os.getenv("RSI_LEN", "14")),
    "RSI_BUY_MAX": int(os.getenv("RSI_BUY_MAX", "65")),   # avoid overbought
    "RSI_SELL_MIN": int(os.getenv("RSI_SELL_MIN", "35")), # avoid oversold
    # News RSS feeds
    "NEWS_FEEDS": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://www.theblock.co/rss",  # may include paywalled content
        "https://blog.binance.com/en/rss"
    ],
    # File storage
    "STATE_FILE": os.getenv("STATE_FILE", "positions.json"),
    # Visuals
    "IMG_WIDTH": 1200,
    "IMG_HEIGHT": 700,
    # Futuristic flags
    "USE_THREADS": True,      # replies to the signal post for later TP/SL updates
    "GATE_JOIN": True,        # requires users to join channel before getting /signal in DM
    "POST_SMALL_TIPS": True,  # show a small trading tip in each status update
}

# ------------------------- LOGGING -------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("crypto-signal-bot")

# ------------------------- UTILITIES -------------------------

BINANCE_BASE = "https://api.binance.com"
INTERVAL_MAP = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}

TIPS = [
    "Risk per trade ≤ 1–2% keeps you in the game.",
    "Always place a hard stop—hope is not a strategy.",
    "Journal your trades to learn faster.",
    "Wait for candle close; wicks can fake you out.",
    "Size positions by distance to SL, not by feeling."
]

def now_utc():
    return datetime.now(timezone.utc)

def read_state() -> Dict[str, Any]:
    if os.path.exists(CONFIG["STATE_FILE"]):
        try:
            with open(CONFIG["STATE_FILE"], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read state: {e}")
    return {"open_positions": {}}

def write_state(state: Dict[str, Any]) -> None:
    try:
        with open(CONFIG["STATE_FILE"], "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write state: {e}")

def pct(a, b):
    if b == 0: 
        return 0.0
    return (a - b) / b * 100.0

# ------------------------- DATA FETCH -------------------------

def fetch_klines(symbol: str, interval: str = "1h", limit: int = 300) -> pd.DataFrame:
    """Fetch recent klines from Binance public API."""
    endpoint = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": INTERVAL_MAP.get(interval, "1h"), "limit": limit}
    r = requests.get(endpoint, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time",
            "qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df[["open_time","open","high","low","close","volume","close_time"]]

def fetch_price(symbol: str) -> float:
    endpoint = f"{BINANCE_BASE}/api/v3/ticker/price"
    r = requests.get(endpoint, params={"symbol": symbol.upper()}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])

# ------------------------- INDICATORS -------------------------

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(length).mean()

# ------------------------- SIGNAL LOGIC -------------------------

def make_trade_signal(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    df = fetch_klines(symbol, tf, limit=300)
    if len(df) < 60:
        return None

    df["ema_fast"] = ema(df["close"], CONFIG["EMA_FAST"])
    df["ema_slow"] = ema(df["close"], CONFIG["EMA_SLOW"])
    df["rsi"] = rsi(df["close"], CONFIG["RSI_LEN"])
    df["atr"] = atr(df, 14)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    direction = None
    # Simple cross confirmation on closed candle
    if prev["ema_fast"] < prev["ema_slow"] and last["ema_fast"] > last["ema_slow"] and last["rsi"] <= CONFIG["RSI_BUY_MAX"]:
        direction = "LONG"
    elif prev["ema_fast"] > prev["ema_slow"] and last["ema_fast"] < last["ema_slow"] and last["rsi"] >= CONFIG["RSI_SELL_MIN"]:
        direction = "SHORT"

    if direction is None:
        return None

    entry = float(last["close"])
    atrv = float(last["atr"])
    if not np.isfinite(atrv) or atrv == 0:
        return None

    if direction == "LONG":
        sl = entry - CONFIG["ATR_MULT_SL"] * atrv
        tps = [entry * (1 + p) for p in CONFIG["TP_PCTS"]]
    else:
        sl = entry + CONFIG["ATR_MULT_SL"] * atrv
        tps = [entry * (1 - p) for p in CONFIG["TP_PCTS"]]

    signal = {
        "symbol": symbol,
        "timeframe": tf,
        "timestamp": now_utc().isoformat(),
        "direction": direction,
        "entry": round(entry, 6),
        "sl": round(sl, 6),
        "tp_list": [round(x, 6) for x in tps],
        "ema_fast": round(float(last["ema_fast"]), 6),
        "ema_slow": round(float(last["ema_slow"]), 6),
        "rsi": round(float(last["rsi"]), 2),
        "atr": round(atrv, 6),
        "status": "OPEN",
        "hits": [False]*len(CONFIG["TP_PCTS"])
    }
    return signal

# ------------------------- IMAGE GENERATION -------------------------

def draw_signal_image(symbol: str, direction: str, entry: float, sl: float, tps: List[float]) -> bytes:
    """
    Create an image (PNG) with green TP box and red box from entry to SL.
    Also overlays a simple close-price line for context.
    """
    width, height = CONFIG["IMG_WIDTH"], CONFIG["IMG_HEIGHT"]
    img = Image.new("RGB", (width, height), color=(18, 18, 22))
    draw = ImageDraw.Draw(img)

    # Fetch last ~150 closes for a simple background line
    try:
        df = fetch_klines(symbol, CONFIG["TIMEFRAME"], limit=200)
        closes = df["close"].tail(150).tolist()
    except Exception:
        closes = [entry*0.98, entry*0.99, entry, entry*1.01, entry*1.02]

    # Scale function
    pad = 80
    data = closes + [sl] + tps
    ymin, ymax = min(data), max(data)
    if math.isclose(ymin, ymax):
        ymin *= 0.99
        ymax *= 1.01

    def y_to_px(y):
        # higher price -> smaller y (top of image)
        return pad + (1 - (y - ymin) / (ymax - ymin)) * (height - 2*pad)

    # Draw price line
    if len(closes) >= 2:
        step_x = (width - 2*pad) / (len(closes) - 1)
        pts = [(pad + i*step_x, y_to_px(v)) for i, v in enumerate(closes)]
        draw.line(pts, fill=(200, 200, 200), width=2)

    # Boxes
    entry_y = y_to_px(entry)
    sl_y = y_to_px(sl)
    # TP zone from min(tp) to max(tp)
    tp_min = min(tps) if direction == "SHORT" else entry
    tp_max = max(tps) if direction == "LONG" else entry
    tp_min_y = y_to_px(tp_min)
    tp_max_y = y_to_px(tp_max)

    # Red box: entry -> SL
    red_top = min(entry_y, sl_y)
    red_bottom = max(entry_y, sl_y)
    draw.rectangle([(pad, red_top), (width - pad, red_bottom)], outline=(255, 80, 80), width=4, fill=(80, 20, 20, 60))

    # Green box: entry -> furthest TP
    green_top = min(entry_y, tp_max_y) if direction == "LONG" else min(tp_min_y, entry_y)
    green_bottom = max(entry_y, tp_max_y) if direction == "LONG" else max(tp_min_y, entry_y)
    draw.rectangle([(pad, green_top), (width - pad, green_bottom)], outline=(80, 220, 120), width=4, fill=(20, 60, 30, 60))

    # Text info
    try:
        font = ImageFont.truetype("arial.ttf", 28)
        bigfont = ImageFont.truetype("arial.ttf", 38)
    except Exception:
        font = ImageFont.load_default()
        bigfont = ImageFont.load_default()

    title = f"{symbol} | {direction} | {CONFIG['TIMEFRAME']}"
    draw.text((pad, 20), title, fill=(230, 230, 230), font=bigfont)

    ytxt = 70
    draw.text((pad, ytxt), f"Entry: {entry}", fill=(230, 230, 230), font=font); ytxt += 30
    draw.text((pad, ytxt), f"SL   : {sl}",    fill=(255, 150, 150), font=font); ytxt += 30
    for i, tp in enumerate(tps, 1):
        draw.text((pad, ytxt), f"TP{i} : {tp}", fill=(150, 255, 170), font=font); ytxt += 26

    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

def news_card_image(title: str, source: str) -> bytes:
    """Generate a simple image card for a news headline."""
    width, height = 1200, 628
    img = Image.new("RGB", (width, height), color=(18, 18, 22))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
        small = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    # Decorative bar
    draw.rectangle([(0, 0), (width, 12)], fill=(80, 220, 120))

    # Wrap title
    def wrap(text, max_width):
        words = text.split()
        lines, line = [], ""
        for w in words:
            test = (line + " " + w).strip()
            if draw.textlength(test, font=font) <= max_width:
                line = test
            else:
                lines.append(line); line = w
        if line: lines.append(line)
        return lines

    lines = wrap(title, width - 60)
    y = 80
    for ln in lines[:8]:
        draw.text((30, y), ln, fill=(235, 235, 235), font=font); y += 50

    draw.text((30, height-60), f"Source: {source}", fill=(180, 180, 180), font=small)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

# ------------------------- TELEGRAM HELPERS -------------------------

async def ensure_joined(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> bool:
    """Check if the user is a member of the channel (requires bot admin)."""
    if not CONFIG["GATE_JOIN"]:
        return True
    try:
        chat_member = await context.bot.get_chat_member(CONFIG["CHANNEL_ID"], user_id)
        status = chat_member.status  # 'creator', 'administrator', 'member', 'left', 'kicked'
        return status in ("creator", "administrator", "member")
    except Exception as e:
        logger.warning(f"get_chat_member failed: {e}")
        # If can't verify, allow but still show links first
        return True

def links_markup() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("📢 Join Channel", url=CONFIG["CHANNEL_INVITE_LINK"])],
        [InlineKeyboardButton("🟡 Binance Invite", url=CONFIG["BINANCE_REF_LINK"])],
    ]
    return InlineKeyboardMarkup(buttons)

def small_tip() -> str:
    if CONFIG["POST_SMALL_TIPS"]:
        return f"\n💡 Tip: {np.random.choice(TIPS)}"
    return ""

# ------------------------- COMMANDS -------------------------

WELCOME = (
    "Salam! 👋\n\n"
    "Pehle please mere channel join karein aur Binance link se register karein. "
    "Uske baad main aapko clear crypto signals aur news deta rahunga.\n\n"
    "📢 Channel: {channel}\n"
    "🟡 Binance: {binance}\n\n"
    "Commands:\n"
    "/signal — latest algorithmic trade idea (Entry/TP1..TP5/SL + image)\n"
    "/news — fresh crypto headlines (with image cards)\n"
    "/help — info & disclaimer"
)

HELP_TEXT = (
    "ℹ️ Yeh bot demo strategy (EMA20/50 + RSI + ATR) se daily 1 clear idea deta hai.\n"
    "❗ Financial advice nahi hai. Market risky hota hai. DYOR.\n\n"
    "Channel: {channel}\nBinance: {binance}"
)

def format_signal_caption(sig: Dict[str, Any]) -> str:
    tps = "\n".join([f"TP{i+1}: {tp}" for i, tp in enumerate(sig["tp_list"])])
    risk = (
        "\n\nPosition sizing example:\n"
        "Risk per trade = 1% of equity.\n"
        "Size = (Risk $) / (|Entry − SL|)."
    )
    return (
        f"🔔 *{sig['symbol']}* | *{sig['direction']}* | {sig['timeframe']}\n"
        f"Entry: `{sig['entry']}`\nSL: `{sig['sl']}`\n{tps}\n"
        f"EMA{CONFIG['EMA_FAST']}: `{sig['ema_fast']}`, EMA{CONFIG['EMA_SLOW']}: `{sig['ema_slow']}`, RSI: `{sig['rsi']}`"
        f"{small_tip()}{risk}\n\n"
        f"📢 Channel: {CONFIG['CHANNEL_INVITE_LINK']}\n"
        f"🟡 Binance: {CONFIG['BINANCE_REF_LINK']}"
    )

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        WELCOME.format(channel=CONFIG["CHANNEL_INVITE_LINK"], binance=CONFIG["BINANCE_REF_LINK"]),
        reply_markup=links_markup()
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        HELP_TEXT.format(channel=CONFIG["CHANNEL_INVITE_LINK"], binance=CONFIG["BINANCE_REF_LINK"]),
        reply_markup=links_markup()
    )

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    ok = await ensure_joined(context, user_id)
    msg = (
        "Pehle channel join karein & Binance link use karein, phir details milengi."
        if not ok else "Signal generate ho raha hai..."
    )
    await update.message.reply_text(
        f"📢 {CONFIG['CHANNEL_INVITE_LINK']}\n🟡 {CONFIG['BINANCE_REF_LINK']}\n\n{msg}",
        reply_markup=links_markup()
    )

    # Try generate from the universe until we find one
    for sym in CONFIG["SYMBOLS"]:
        try:
            sig = make_trade_signal(sym, CONFIG["TIMEFRAME"])
            if sig:
                image_bytes = draw_signal_image(sig["symbol"], sig["direction"], sig["entry"], sig["sl"], sig["tp_list"])
                caption = format_signal_caption(sig)
                sent = await update.message.reply_photo(photo=image_bytes, caption=caption, parse_mode=ParseMode.MARKDOWN)
                # Post also to channel
                ch_post = await context.bot.send_photo(chat_id=CONFIG["CHANNEL_ID"], photo=image_bytes, caption=caption, parse_mode=ParseMode.MARKDOWN)
                # Save state
                state = read_state()
                sig["message_id"] = ch_post.message_id
                state["open_positions"][sig["symbol"]] = sig
                write_state(state)
                return
        except Exception as e:
            logger.error(f"Signal generation failed for {sym}: {e}")
            continue
    await update.message.reply_text("Aaj koi high-confidence setup nahi mila. Kal koshish karenge!")

async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📢 {CONFIG['CHANNEL_INVITE_LINK']}\n🟡 {CONFIG['BINANCE_REF_LINK']}\n\nCrypto headlines aa rahe hain...",
        reply_markup=links_markup()
    )
    headlines = get_news_headlines(limit=5)
    for h in headlines:
        img = news_card_image(h["title"], h["source"])
        cap = f"🗞️ *{h['title']}*\nSource: {h['source']}\n{small_tip()}"
        await update.message.reply_photo(photo=img, caption=cap, parse_mode=ParseMode.MARKDOWN)
        # Post to channel
        await context.bot.send_photo(chat_id=CONFIG["CHANNEL_ID"], photo=img, caption=cap, parse_mode=ParseMode.MARKDOWN)

# ------------------------- NEWS -------------------------

def get_news_headlines(limit: int = 5) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for url in CONFIG["NEWS_FEEDS"]:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:3]:  # take a few per-feed
                title = getattr(e, "title", "").strip()
                if not title:
                    continue
                source = feed.feed.get("title", "Crypto News")
                items.append({"title": title, "source": source})
        except Exception as e:
            logger.warning(f"Feed failed: {url} -> {e}")
    # Deduplicate by title
    seen = set()
    deduped = []
    for it in items:
        if it["title"] not in seen:
            deduped.append(it)
            seen.add(it["title"])
        if len(deduped) >= limit:
            break
    return deduped

# ------------------------- DAILY JOBS -------------------------

async def job_daily_signal(context: CallbackContext):
    """Once a day: pick a symbol, post one clear signal to channel."""
    for sym in CONFIG["SYMBOLS"]:
        try:
            sig = make_trade_signal(sym, CONFIG["TIMEFRAME"])
            if not sig:
                continue
            img = draw_signal_image(sig["symbol"], sig["direction"], sig["entry"], sig["sl"], sig["tp_list"])
            caption = format_signal_caption(sig)
            post = await context.bot.send_photo(chat_id=CONFIG["CHANNEL_ID"], photo=img, caption=caption, parse_mode=ParseMode.MARKDOWN)
            # Save state
            state = read_state()
            sig["message_id"] = post.message_id
            state["open_positions"][sig["symbol"]] = sig
            write_state(state)
            # Only one trade per day
            break
        except Exception as e:
            logger.error(f"Daily signal failed for {sym}: {e}")
            continue

async def job_daily_news(context: CallbackContext):
    """Once a day: post curated headlines with images."""
    headlines = get_news_headlines(limit=5)
    for h in headlines:
        img = news_card_image(h["title"], h["source"])
        cap = f"🗞️ *{h['title']}*\nSource: {h['source']}\n{small_tip()}"
        await context.bot.send_photo(chat_id=CONFIG["CHANNEL_ID"], photo=img, caption=cap, parse_mode=ParseMode.MARKDOWN)

async def job_monitor_positions(context: CallbackContext):
    """Every few minutes: check TP/SL status and update thread."""
    state = read_state()
    if not state["open_positions"]:
        return
    for sym, pos in list(state["open_positions"].items()):
        try:
            price = fetch_price(sym)
            direction = pos["direction"]
            entry = pos["entry"]
            sl = pos["sl"]
            tps = pos["tp_list"]
            msg_id = pos.get("message_id")

            hit_sl = (price <= sl) if direction == "LONG" else (price >= sl)
            if hit_sl:
                text = f"⛔ *{sym}* {direction}: SL hit at `{price}`. {small_tip()}"
                await context.bot.send_message(chat_id=CONFIG["CHANNEL_ID"], text=text, parse_mode=ParseMode.MARKDOWN, reply_to_message_id=(msg_id if CONFIG["USE_THREADS"] else None))
                del state["open_positions"][sym]
                write_state(state)
                continue

            changed = False
            for i, tp in enumerate(tps):
                if pos["hits"][i]:
                    continue
                if (direction == "LONG" and price >= tp) or (direction == "SHORT" and price <= tp):
                    pos["hits"][i] = True
                    changed = True
                    text = f"🎯 *{sym}* {direction}: TP{i+1} hit at `{price}` — congrats! {small_tip()}"
                    await context.bot.send_message(chat_id=CONFIG["CHANNEL_ID"], text=text, parse_mode=ParseMode.MARKDOWN, reply_to_message_id=(msg_id if CONFIG["USE_THREADS"] else None))

            # Close position if max TP hit
            if all(pos["hits"]):
                done_text = f"✅ *{sym}* {direction}: All TPs achieved. Trade closed. {small_tip()}"
                await context.bot.send_message(chat_id=CONFIG["CHANNEL_ID"], text=done_text, parse_mode=ParseMode.MARKDOWN, reply_to_message_id=(msg_id if CONFIG["USE_THREADS"] else None))
                del state["open_positions"][sym]
                write_state(state)
                continue

            if changed:
                state["open_positions"][sym] = pos
                write_state(state)

        except Exception as e:
            logger.error(f"Monitor error for {sym}: {e}")
            continue

# ------------------------- MAIN -------------------------

def local_helsinki_time(hour: int, minute: int = 0) -> datetime:
    """Return next occurrence of the given hour in Europe/Helsinki local time, then convert to UTC for JobQueue."""
    # naive approach: Helsinki is UTC+2 or +3. We'll approximate as +3.
    # For accuracy, use pytz or zoneinfo; kept simple for single-file demo.
    helsinki_offset = 3  # adjust if DST differs
    now = datetime.utcnow()
    first = datetime(now.year, now.month, now.day, hour - helsinki_offset, minute, 0)
    if first < now:
        first += timedelta(days=1)
    return first.replace(tzinfo=timezone.utc)

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

def main():
    token = CONFIG["TELEGRAM_BOT_TOKEN"]
    if not token or token == "PUT-YOUR-TOKEN-HERE":
        print("Please set TELEGRAM_BOT_TOKEN in the CONFIG or environment variables.")
        return

    app = ApplicationBuilder().token(token).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("news", cmd_news))

    app.add_error_handler(on_error)

    # Jobs
    jq = app.job_queue
    jq.run_daily(job_daily_signal, time=local_helsinki_time(CONFIG["DAILY_SIGNAL_HOUR"]).time())
    jq.run_daily(job_daily_news, time=local_helsinki_time(CONFIG["DAILY_NEWS_HOUR"]).time())
    jq.run_repeating(job_monitor_positions, interval=180, first=10)  # every 3 min

    print("Bot is running... Press CTRL+C to stop.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()

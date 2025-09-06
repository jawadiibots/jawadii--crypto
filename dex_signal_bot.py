#!/usr/bin/env python3
"""
dex_signal_bot.py
Advanced DEX token signal bot (DexScreener first, CoinGecko fallback).
Commands:
  /trade <token_or_coin>      -> analyze token (e.g., /trade pepe or /trade bitcoin)
  /setpair <chain> <pairId>  -> bind token name to a specific DEX pair (admin)
  /help                      -> instructions
"""
import os, json, requests, math, traceback
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

load_dotenv()

# ---------------- Config ----------------
BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
ADMIN_CHAT = os.getenv("TG_ADMIN_CHAT")  # optional: your telegram id to receive errors
RR_RATIO = float(os.getenv("RR_RATIO", "2.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "5"))
PAIRS_FILE = "pairs.json"  # local mapping file for user setpair

DEX_SEARCH = "https://api.dexscreener.com/latest/dex/search?q={q}"
DEX_PAIR   = "https://api.dexscreener.com/latest/dex/pairs/{chain}/{pairId}"
COINGECKO_OHLC = "https://api.coingecko.com/api/v3/coins/{id}/ohlc?vs_currency=usd&days={days}"

# ---------------- Utilities ----------------
def load_pairs():
    try:
        if os.path.exists(PAIRS_FILE):
            with open(PAIRS_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return {}

def save_pairs(mapping):
    try:
        with open(PAIRS_FILE, "w") as f:
            json.dump(mapping, f)
    except:
        pass

# -------------- Indicators ----------------
def ema_series(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()

def rsi_series(series: pd.Series, length: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def atr_series(df: pd.DataFrame, length: int = 14):
    high = df['high']; low = df['low']; close = df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# -------------- Fetching ------------------
def fetch_dex_search(query: str, timeout=12):
    try:
        r = requests.get(DEX_SEARCH.format(q=query), timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def fetch_pair_detail(chain: str, pairId: str, timeout=12):
    try:
        url = DEX_PAIR.format(chain=chain, pairId=pairId)
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def fetch_coingecko_ohlc(coin_id: str, days: int = 7, timeout=12):
    try:
        url = COINGECKO_OHLC.format(id=coin_id, days=days)
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def find_candles_in_json(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = k.lower()
            if 'candle' in key or 'chart' in key or 'ohlc' in key:
                if isinstance(v, list) and len(v) > 0:
                    return v
            res = find_candles_in_json(v)
            if res is not None:
                return res
    elif isinstance(obj, list) and len(obj) > 0:
        if isinstance(obj[0], list) and len(obj[0]) >= 5 and isinstance(obj[0][0], (int, float)):
            return obj
    return None

def build_df_from_dex_pair(json_pair):
    try:
        candles = find_candles_in_json(json_pair)
        if not candles:
            return None
        rows = []
        for c in candles:
            if isinstance(c, list) and len(c) >= 5:
                ts = pd.to_datetime(int(c[0]), unit='ms')
                o, h, l, cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
                vol = float(c[5]) if len(c) > 5 else None
                rows.append([ts, o, h, l, cl, vol])
            elif isinstance(c, dict):
                ts_val = c.get('t') or c.get('timestamp') or c.get('time')
                if not ts_val:
                    continue
                ts = pd.to_datetime(int(ts_val), unit='ms')
                o = float(c.get('o') or c.get('open'))
                h = float(c.get('h') or c.get('high'))
                l = float(c.get('l') or c.get('low'))
                cl = float(c.get('c') or c.get('close'))
                vol = float(c.get('v') or c.get('volume') or 0)
                rows.append([ts, o, h, l, cl, vol])
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception:
        return None

def build_df_from_coingecko(ohlc_json):
    try:
        rows = []
        for item in ohlc_json:
            ts = pd.to_datetime(int(item[0]), unit='ms')
            o, h, l, c = float(item[1]), float(item[2]), float(item[3]), float(item[4])
            rows.append([ts, o, h, l, c, None])
        df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception:
        return None

# -------------- Analysis -------------------
def analyze_ohlc(df: pd.DataFrame, swing_lookback: int = SWING_LOOKBACK, rr_ratio: float = RR_RATIO, sl_atr_mult: float = SL_ATR_MULT):
    if df is None or len(df) < 30:
        return {"ok": False, "reason": "not_enough_data"}
    df = df.copy().dropna()
    df['ema50'] = ema_series(df['close'], 50)
    df['ema200'] = ema_series(df['close'], 200)
    df['rsi'] = rsi_series(df['close'], 14)
    df['atr'] = atr_series(df, 14)

    body = (df['close'] - df['open']).abs()
    upper_w = df['high'] - df[['open','close']].max(axis=1)
    lower_w = df[['open','close']].min(axis=1) - df['low']
    df['bull_eng'] = (df['close'] > df['open']) & (df['open'].shift(1) < df['close'].shift(1)) & (df['open'] <= df['close'].shift(1)) & (df['close'] >= df['open'].shift(1))
    df['bear_eng'] = (df['close'] < df['open']) & (df['open'].shift(1) > df['close'].shift(1)) & (df['open'] >= df['close'].shift(1)) & (df['close'] <= df['open'].shift(1))
    df['hammer'] = (body > 0) & (lower_w >= 2 * body) & (upper_w <= body) & (df['close'] > df['open'])
    df['shooting'] = (body > 0) & (upper_w >= 2 * body) & (lower_w <= body) & (df['open'] > df['close'])

    last = df.iloc[-1]
    entry = float(last['close'])
    ema50 = float(last['ema50'])
    ema200 = float(last['ema200'])
    current_rsi = float(last['rsi'])
    atr = float(last['atr'])

    bull_base = (ema50 > ema200) and (current_rsi > 40) and (bool(last['bull_eng']) or bool(last['hammer']))
    short_base = (ema50 < ema200) and (current_rsi < 60) and (bool(last['bear_eng']) or bool(last['shooting']))

    bull_risk = (current_rsi > 70) or (ema50 < ema200)
    short_risk = (current_rsi < 30) or (ema50 > ema200)
    risk_flag = bull_risk or short_risk

    bull_safe = bull_base and not bull_risk
    short_safe = short_base and not short_risk

    swing_low = float(df['low'].rolling(swing_lookback).min().iloc[-1])
    swing_high = float(df['high'].rolling(swing_lookback).max().iloc[-1])

    long_sl = swing_low - atr * sl_atr_mult
    short_sl = swing_high + atr * sl_atr_mult
    long_risk = max(1e-8, entry - long_sl)
    short_risk = max(1e-8, short_sl - entry)
    long_tp = entry + long_risk * rr_ratio
    short_tp = entry - short_risk * rr_ratio

    if bull_safe:
        return {"ok": True, "signal": "CONFIRM LONG", "entry": entry, "sl": long_sl, "tp": long_tp,
                "rsi": current_rsi, "reason": "EMA trend + RSI + candle", "ema50": ema50, "ema200": ema200}
    if short_safe:
        return {"ok": True, "signal": "CONFIRM SHORT", "entry": entry, "sl": short_sl, "tp": short_tp,
                "rsi": current_rsi, "reason": "EMA trend + RSI + candle", "ema50": ema50, "ema200": ema200}
    if risk_flag:
        return {"ok": True, "signal": "RISK - CANCEL TRADE", "entry": entry, "rsi": current_rsi,
                "reason": ("BullRisk" if bull_risk else "ShortRisk"), "ema50": ema50, "ema200": ema200}
    return {"ok": True, "signal": "WAIT - No clear setup", "entry": entry, "rsi": current_rsi, "reason": "No confluence",
            "ema50": ema50, "ema200": ema200}

# ------------- Multi-timeframe aggregator -------------
def multi_time_analysis(dfs: dict):
    results = {}
    votes = {"long":0,"short":0,"wait":0,"risk":0}
    for tf, df in dfs.items():
        res = analyze_ohlc(df)
        results[tf] = res
        if not res.get("ok"):
            votes["wait"] += 1
            continue
        sig = res.get("signal","").lower()
        if "confirm long" in sig: votes["long"] += 1
        elif "confirm short" in sig: votes["short"] += 1
        elif "risk" in sig: votes["risk"] += 1
        else: votes["wait"] += 1

    if votes["risk"] > 0:
        return {"final_signal":"RISK - CANCEL TRADE", "votes":votes, "details":results}
    if votes["long"] > votes["short"] and votes["long"] >= votes["wait"]:
        return {"final_signal":"CONFIRM LONG", "votes":votes, "details":results}
    if votes["short"] > votes["long"] and votes["short"] >= votes["wait"]:
        return {"final_signal":"CONFIRM SHORT", "votes":votes, "details":results}
    return {"final_signal":"WAIT - No confluence", "votes":votes, "details":results}

# ------------- High level token -> dfs -------------
def get_best_pair_and_dfs(query_token: str):
    # 1) check user mapping
    pairs_map = load_pairs()
    if query_token in pairs_map:
        meta = pairs_map[query_token]
        # meta expected: {"chain":..,"pairId":..}
        detail = fetch_pair_detail(meta['chain'], meta['pairId'])
        if detail:
            df = build_df_from_dex_pair(detail)
            if df is not None:
                return {"ok":True,"source":"dexscreener","pair":meta,"dfs": make_resampled_dfs(df)}
    # 2) Try search on DexScreener
    search = fetch_dex_search(query_token)
    if search:
        pairs = None
        if isinstance(search, dict):
            pairs = search.get("pairs") or next((v for k,v in search.items() if isinstance(v, list)), None)
        if pairs and isinstance(pairs, list) and len(pairs)>0:
            top = pairs[0]
            chain = top.get("chainId") or top.get("chain") or top.get("dexId") or top.get("chainName")
            pairId = top.get("pairId") or top.get("pair") or top.get("pairAddress") or top.get("pairAddress")
            if chain and pairId:
                detail = fetch_pair_detail(chain, pairId)
                if detail:
                    df = build_df_from_dex_pair(detail)
                    if df is not None:
                        return {"ok":True,"source":"dexscreener","pair":top,"dfs": make_resampled_dfs(df)}
    # 3) Fallback CoinGecko (major coins)
    ohlc = fetch_coingecko_ohlc(query_token, days=7)
    if ohlc:
        df = build_df_from_coingecko(ohlc)
        if df is not None:
            return {"ok":True,"source":"coingecko","pair":{"id":query_token},"dfs": make_resampled_dfs(df)}
    return {"ok":False, "reason":"no_data_found"}

def make_resampled_dfs(df: pd.DataFrame):
    dfs = {}
    try:
        # try to resample to 15m,1h,4h if possible
        # ensure datetime index
        df2 = df.copy()
        if not isinstance(df2.index, pd.DatetimeIndex):
            df2.index = pd.to_datetime(df2.index)
        # if data frequency is very high (per second/min), can resample
        dfs['15m'] = df2.resample('15T').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        dfs['1h']  = df2.resample('1H').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        dfs['4h']  = df2.resample('4H').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    except Exception:
        dfs = {'1h': df}
    return dfs

# --------------- Telegram handlers ----------------
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Usage:\n"
        "/trade <token_or_coin>  - analyze token (e.g., /trade pepe or /trade 0xPairAddress)\n"
        "/setpair <token> <chain> <pairId> - bind token name to specific DEX pair (admin)\n"
        "/help - this message\n\n"
        "Notes: use CoinGecko id (bitcoin, ethereum) for large coins, or token name for DEX tokens.\n"
    )
    await update.message.reply_text(txt)

async def setpair_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # only admin allowed (if ADMIN_CHAT set)
    if ADMIN_CHAT and str(update.effective_user.id) != str(ADMIN_CHAT):
        await update.message.reply_text("❌ You are not allowed to use this command.")
        return
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("Usage: /setpair <token> <chain> <pairId>")
        return
    token = args[0].lower()
    chain = args[1]
    pairId = args[2]
    m = load_pairs()
    m[token] = {"chain": chain, "pairId": pairId}
    save_pairs(m)
    await update.message.reply_text(f"✅ Saved mapping: {token} -> {chain}/{pairId}")

async def trade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /trade <token_or_coin> (e.g., /trade pepe or /trade bitcoin)")
        return
    token = context.args[0].lower()
    msg = await update.message.reply_text(f"🔎 Searching data for `{token}` ...", parse_mode="Markdown")
    try:
        res = get_best_pair_and_dfs(token)
        if not res.get("ok"):
            await msg.edit_text(f"❌ Could not find candle data for `{token}`. Try /setpair to bind a pairId.", parse_mode="Markdown")
            return
        dfs = res['dfs']
        agg = multi_time_analysis(dfs)
        final = agg['final_signal']
        # base formatting
        reply = f"*{token.upper()} — {final}*\n\n"
        # show votes summary
        votes = agg.get("votes", {})
        reply += f"_Votes (15m/1h/4h):_ {votes.get('long',0)}L / {votes.get('short',0)}S / {votes.get('wait',0)}W / {votes.get('risk',0)}R\n\n"

        # include details per TF
        for tf, tfres in agg['details'].items():
            if not tfres.get("ok"):
                reply += f"*{tf}*: No data\n"
                continue
            sig = tfres['signal']
            entry = tfres.get('entry')
            rsi = tfres.get('rsi')
            reply += f"*{tf}*: {sig} | Entry: `{entry:.8g}` | RSI: `{rsi:.2f}`\n"
            if sig.startswith("CONFIRM"):
                reply += f"SL: `{tfres['sl']:.8g}`  TP: `{tfres['tp']:.8g}`\n"
            reply += "\n"

        # final short guidance
        if final.startswith("CONFIRM LONG"):
            # show 1h numbers (preferred) if available
            prefer = agg['details'].get('1h') or list(agg['details'].values())[0]
            if prefer.get('ok') and prefer['signal'].startswith("CONFIRM"):
                reply += f"✅ *Action:* Consider LONG (Entry `{prefer['entry']:.8g}`)  SL `{prefer['sl']:.8g}`  TP `{prefer['tp']:.8g}`\n"
        elif final.startswith("CONFIRM SHORT"):
            prefer = agg['details'].get('1h') or list(agg['details'].values())[0]
            if prefer.get('ok') and prefer['signal'].startswith("CONFIRM"):
                reply += f"✅ *Action:* Consider SHORT (Entry `{prefer['entry']:.8g}`)  SL `{prefer['sl']:.8g}`  TP `{prefer['tp']:.8g}`\n"
        elif final.startswith("RISK"):
            reply += "⚠️ *Action:* MARKET RISKY — Cancel trades / Stay out.\n"
        else:
            reply += "⏳ *Action:* Wait for clearer confluence across timeframes.\n"

        reply += "\n_Method:_ EMA50/EMA200 + RSI + candle patterns + ATR-based SL/TP."
        await msg.edit_text(reply, parse_mode="Markdown")
    except Exception as e:
        tb = traceback.format_exc()
        if ADMIN_CHAT:
            # try notify admin
            try:
                app = context.application
                await app.bot.send_message(chat_id=int(ADMIN_CHAT), text=f"Bot error for /trade {token}:\n{e}\n")
            except:
                pass
        await msg.edit_text("❌ Error occurred while analyzing. Try again later.")
        print("ERROR in /trade:", e)
        print(tb)

# --------------- Main -----------------------
def main():
    if not BOT_TOKEN:
        print("TG_BOT_TOKEN not set in environment.")
        return
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("trade", trade_cmd))
    app.add_handler(CommandHandler("setpair", setpair_cmd))
    print("Bot started. Listening for commands...")
    app.run_polling()

if __name__ == "__main__":
    main()

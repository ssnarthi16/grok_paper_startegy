import copy
import hashlib
import io
import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

IST = "Asia/Kolkata"
MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
PATTERNS = ("Bullish Engulfing", "Bullish Harami", "Tweezer Bottom")
CANDLE_KEYS = ("prev_O", "prev_H", "prev_L", "prev_C", "sig_O", "sig_H", "sig_L", "sig_C")

SL_PCT = 0.0
MAX_HOLD_DAYS = 1
TREND_LOOKBACK = 0
TWEEZER_TOL_PCT = 0.10
FUTURES_MARGIN_PCT = 10.0
BROKERAGE_PCT = 0.0
SLIPPAGE_PCT = 0.0
FRESH_SECONDS = 120
REFRESH_SECONDS = 5

TIMEFRAMES = {
    "1 minute": ("ONE_MINUTE", 1),
    "3 minutes": ("THREE_MINUTE", 3),
    "5 minutes": ("FIVE_MINUTE", 5),
    "10 minutes": ("TEN_MINUTE", 10),
    "15 minutes": ("FIFTEEN_MINUTE", 15),
    "30 minutes": ("THIRTY_MINUTE", 30),
    "1 hour": ("ONE_HOUR", 60),
}

OWNER_ID = "owner"
# Absolute data root so saves survive cwd changes on Streamlit Cloud.
# Override with env DATA_DIR if needed. App reboot on free Cloud can still wipe disk.
_DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(os.path.expanduser("~"), "paper_trading_data")
STRATEGIES_ROOT = os.path.join(_DATA_DIR, "strategies")
os.makedirs(STRATEGIES_ROOT, exist_ok=True)


# ============================== CONFIG ==============================
@dataclass
class Config:
    capital: float = 100000.0
    max_open: int = 5
    target_pct: float = 1.0
    sl_pct: float = SL_PCT
    max_hold: int = MAX_HOLD_DAYS
    tweezer_tol_pct: float = TWEEZER_TOL_PCT
    trend_lookback: int = TREND_LOOKBACK
    brokerage_pct: float = BROKERAGE_PCT
    slippage_pct: float = SLIPPAGE_PCT
    interval: str = "FIVE_MINUTE"
    tf_min: int = 5
    patterns: tuple = PATTERNS
    position_amount: float = 0.0
    no_limit: bool = False
    lots_per_position: int = 0                 # MCX / NFO: fixed lots per position (0 = use the amount instead)

    @property
    def per_position(self):
        return self.position_amount if self.position_amount > 0 else self.capital / self.max_open


# ============================== TIME ==============================
def now_ist():
    return pd.Timestamp.now(tz=IST).tz_localize(None)


def market_open(now, exchange="NSE"):
    start, end = (dtime(9, 0), dtime(23, 30)) if exchange == "MCX" else (dtime(9, 15), dtime(15, 30))
    return now.weekday() < 5 and start <= now.time() <= end


def exit_time(exchange):
    return dtime(23, 15) if exchange == "MCX" else dtime(15, 15)


def session_done(now):
    return now.weekday() < 5 and now.time() > dtime(15, 30)


# ============================== PATTERNS ==============================
def detect_patterns(df, cfg):
    o, l, c = df["open"], df["low"], df["close"]
    po, pl, pc = o.shift(1), l.shift(1), c.shift(1)
    prev_red, cur_green = pc < po, c > o
    sig = pd.DataFrame({
        "Bullish Engulfing": prev_red & cur_green & (o <= pc) & (c >= po),
        "Bullish Harami": prev_red & cur_green & (o > pc) & (c < po),
        "Tweezer Bottom": prev_red & cur_green & ((l - pl).abs() <= pl * cfg.tweezer_tol_pct / 100),
    })
    if cfg.trend_lookback > 0:
        sig = sig.mul(pc < c.shift(cfg.trend_lookback + 1), axis=0).astype(bool)
    return sig.fillna(False)


def classify_candles(r, tol_pct=TWEEZER_TOL_PCT):
    try:
        po, pl, pc = float(r["prev_O"]), float(r["prev_L"]), float(r["prev_C"])
        o, l, c = float(r["sig_O"]), float(r["sig_L"]), float(r["sig_C"])
    except (KeyError, TypeError, ValueError):
        return None
    if any(math.isnan(x) for x in (po, pl, pc, o, l, c)):
        return None
    red, green = pc < po, c > o
    names = []
    if red and green and o <= pc and c >= po:
        names.append("Bullish Engulfing")
    if red and green and o > pc and c < po:
        names.append("Bullish Harami")
    if red and green and abs(l - pl) <= pl * tol_pct / 100:
        names.append("Tweezer Bottom")
    return " + ".join(names) if names else "No pattern"


# ============================== ANGEL ONE ==============================
def angel_login(api_key, client_id, mpin, totp_secret):
    try:
        import pyotp
        from SmartApi import SmartConnect
    except ImportError:
        return None, "Install first:  pip install smartapi-python pyotp"
    try:
        smart = SmartConnect(api_key=api_key.strip())
        otp = pyotp.TOTP(totp_secret.replace(" ", "").strip()).now()
        res = smart.generateSession(client_id.strip(), mpin.strip(), otp)
        if not res or not res.get("status"):
            return None, f"Login failed: {(res or {}).get('message', 'unknown error')}"
        return smart, None
    except Exception as ex:
        return None, f"Login error: {ex}"


def _to_ist_naive(series):
    return pd.to_datetime(series, utc=True).dt.tz_convert(IST).dt.tz_localize(None)


CANDLE_CALL_GAP = 0.4
_RATE_LOCK, _RATE_LAST = threading.Lock(), [0.0]


def _rate_wait():
    with _RATE_LOCK:
        wait = _RATE_LAST[0] + CANDLE_CALL_GAP - time.time()
        if wait > 0:
            time.sleep(wait)
        _RATE_LAST[0] = time.time()


def _candles(smart, item, interval, start, end):
    params = {
        "exchange": item["exchange"], "symboltoken": str(item["token"]), "interval": interval,
        "fromdate": start.strftime("%Y-%m-%d %H:%M"), "todate": end.strftime("%Y-%m-%d %H:%M"),
    }
    for _ in range(4):
        _rate_wait()
        try:
            res = smart.getCandleData(params)
        except Exception:
            res = None
        if res and res.get("status") and res.get("data") is not None:
            if not res["data"]:
                return pd.DataFrame()
            d = pd.DataFrame(res["data"], columns=["date", "open", "high", "low", "close", "volume"])
            d["date"] = _to_ist_naive(d["date"])
            return d
        time.sleep(3)
    return pd.DataFrame()


def fetch_candles(smart, item, interval, now):
    d = _candles(smart, item, interval, now - pd.Timedelta(days=5), now)
    return d[["date", "open", "high", "low", "close"]] if not d.empty else d


def fetch_bars(smart, item, since, now):
    d = _candles(smart, item, "ONE_MINUTE", since, now)
    return d.set_index("date")[["open", "high", "low", "close"]] if not d.empty else d


def fetch_ltp(smart, item):
    try:
        res = smart.ltpData(item["exchange"], item["tradingsymbol"], str(item["token"]))
        return float(res["data"]["ltp"])
    except Exception:
        return None


LAST_API_ERROR = {"quote": "", "bulk": ""}


def _quote_from(d):
    ltp = float(d["ltp"])
    return {
        "open": float(d.get("open", ltp)), "high": float(d.get("high", ltp)),
        "low": float(d.get("low", ltp)), "close": float(d.get("close", ltp)), "ltp": ltp,
    }


def fetch_quote(smart, item):
    try:
        res = smart.ltpData(item["exchange"], item["tradingsymbol"], str(item["token"]))
        d = (res or {}).get("data")
        if not d:
            LAST_API_ERROR["quote"] = str((res or {}).get("message") or res)
            return None
        return _quote_from(d)
    except Exception as ex:
        LAST_API_ERROR["quote"] = f"{type(ex).__name__}: {ex}"
        return None


def fetch_quotes_bulk(smart, items):
    out = {}
    try:
        by_ex = {}
        for it in items:
            by_ex.setdefault(it["exchange"], []).append(str(it["token"]))
        res = smart.getMarketData("OHLC", by_ex)
        fetched = ((res or {}).get("data") or {}).get("fetched") or []
        if not fetched:
            LAST_API_ERROR["bulk"] = str((res or {}).get("message") or res)[:200]
        for r in fetched:
            try:
                out[str(r["symbolToken"])] = _quote_from(r)
            except (KeyError, TypeError, ValueError):
                continue
    except Exception as ex:
        LAST_API_ERROR["bulk"] = f"{type(ex).__name__}: {ex}"
        return {}
    return out


# ============================== INSTRUMENTS ==============================
NIFTY50_FALLBACK = (
    "ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV BEL BHARTIARTL CIPLA "
    "COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HINDUNILVR ICICIBANK "
    "INDUSINDBK INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE "
    "SBIN SHRIRAMFIN SUNPHARMA TATACONSUM TATAMOTORS TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO"
).split()


@st.cache_data(ttl=86400, show_spinner=False)
def nifty50():
    try:
        r = requests.get(
            "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
        )
        r.raise_for_status()
        syms = pd.read_csv(io.StringIO(r.text))["Symbol"].str.strip().tolist()
        if len(syms) >= 40:
            return syms
    except Exception:
        pass
    return list(NIFTY50_FALLBACK)


MASTER_CACHE = "scrip_master_cache.pkl"


@st.cache_resource(ttl=86400, show_spinner=False)
def load_master():
    if os.path.exists(MASTER_CACHE) and pd.Timestamp(os.path.getmtime(MASTER_CACHE), unit="s").date() == pd.Timestamp.now().date():
        try:
            return pd.read_pickle(MASTER_CACHE)
        except Exception:
            pass
    df = pd.DataFrame(requests.get(MASTER_URL, timeout=120).json())
    df.columns = [c.strip().lower() for c in df.columns]
    keep = [c for c in ("exch_seg", "symbol", "name", "token", "lotsize", "instrumenttype", "expiry") if c in df.columns]
    df = df[keep]
    try:
        df.to_pickle(MASTER_CACHE)
    except Exception:
        pass
    return df


def search_item(smart, exchange, text):
    cache = st.session_state.setdefault("search_cache", {})
    key = f"{exchange}:{text}"
    if key not in cache:
        cache[key] = None
        try:
            rows = (smart.searchScrip(exchange, text) or {}).get("data") or []
            rows = [r for r in rows if r["tradingsymbol"].upper() in (text, text + "-EQ")] or rows
            if rows:
                r = rows[0]
                cache[key] = {
                    "symbol": r["tradingsymbol"].replace("-EQ", ""),
                    "tradingsymbol": r["tradingsymbol"],
                    "token": r["symboltoken"], "exchange": exchange, "lot": 1,
                }
        except Exception:
            pass
    return cache[key]


SENSEX30 = set(
    "ADANIPORTS ASIANPAINT AXISBANK BAJFINANCE BAJAJFINSV BHARTIARTL ETERNAL HCLTECH HDFCBANK HINDUNILVR ICICIBANK "
    "INFY ITC KOTAKBANK LT M&M MARUTI NESTLEIND NTPC POWERGRID RELIANCE SBIN SUNPHARMA TATAMOTORS TATASTEEL TCS "
    "TECHM TITAN TRENT ULTRACEMCO".split()
)
MCX_MAIN = set("GOLD GOLDM SILVER SILVERM CRUDEOIL CRUDEOILM NATURALGAS COPPER ZINC ALUMINIUM LEAD NICKEL".split())
NFO_INDEX = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


def exchange_list(df, exchange):
    d = df[df["exch_seg"] == exchange].copy()
    nm = d["name"].fillna("").str.upper()
    sy = d["symbol"].fillna("").str.upper()
    if exchange == "NSE":
        d = d[sy.str.endswith("-EQ") & nm.isin(set(nifty50()))]
        d["label"] = d["name"]
    elif exchange == "BSE":
        d = d[nm.isin(SENSEX30) | sy.isin(SENSEX30)]
        d["label"] = d["name"].where(nm.isin(SENSEX30), d["symbol"])
    else:
        itype = d["instrumenttype"].fillna("") if "instrumenttype" in d else pd.Series("", index=d.index)
        base = MCX_MAIN if exchange == "MCX" else NFO_INDEX | set(nifty50())
        d = d[itype.str.startswith("FUT") & nm.isin(base)]
        d["exp"] = pd.to_datetime(d["expiry"], format="%d%b%Y", errors="coerce")
        d = d[d["exp"] >= pd.Timestamp.now().normalize()].sort_values("exp")
        d = d.drop_duplicates("name")
        d["label"] = d["symbol"]
    return d.drop_duplicates("label").sort_values("label")


@st.cache_data(ttl=3600, show_spinner=False)
def exchange_list_cached(exchange):
    return exchange_list(load_master(), exchange)


def no_dim():
    st.markdown(
        """<style>
        [data-stale="true"], .stale-element, [data-testid="stElementContainer"][data-stale="true"],
        [data-testid="stSidebar"] [data-stale="true"] { opacity: 1 !important; transition: none !important; }
        </style>""",
        unsafe_allow_html=True,
    )


def to_item(row, exchange):
    try:
        lot = max(int(float(row.get("lotsize") or 1)), 1) if exchange in ("NFO", "MCX") else 1
    except ValueError:
        lot = 1
    return {
        "symbol": row["label"], "tradingsymbol": row["symbol"], "token": row["token"],
        "exchange": exchange, "lot": lot,
    }


# ============================== STRATEGY PATHS ==============================
def _safe_name(name: str) -> str:
    """Folder / file name safe for disk (no path tricks)."""
    name = re.sub(r"[^\w\s\-]", "", name.strip())
    name = re.sub(r"\s+", "_", name)
    return name[:80] or "unnamed"


def user_strategies_root(uid: str) -> str:
    """Per-user folder under absolute DATA_DIR (always absolute path)."""
    if uid == OWNER_ID:
        root = STRATEGIES_ROOT
    else:
        root = os.path.join(_DATA_DIR, "users", uid, "strategies")
    os.makedirs(root, exist_ok=True)
    return root


def strategy_dir(uid: str, folder: str, file: str) -> str:
    return os.path.join(user_strategies_root(uid), _safe_name(folder), _safe_name(file))


def list_strategies(uid: str) -> dict:
    """
    Returns {folder: [{file, path, running, exchange, tf_label, capital, ...}, ...]}
    """
    root = user_strategies_root(uid)
    out = {}
    if not os.path.isdir(root):
        return out
    for folder in sorted(os.listdir(root)):
        fp = os.path.join(root, folder)
        if not os.path.isdir(fp) or folder.startswith("."):
            continue
        files = []
        for fname in sorted(os.listdir(fp)):
            fpath = os.path.join(fp, fname)
            if not os.path.isdir(fpath) or fname.startswith("."):
                continue
            cfg_path = os.path.join(fpath, "config.json")
            if not os.path.exists(cfg_path):
                continue
            meta = {"file": fname, "path": fpath, "folder": folder}
            try:
                meta.update(json.load(open(cfg_path)))
            except Exception:
                pass
            eng = _get_engine_by_path(uid, fpath)
            meta["running"] = eng.running if eng else False
            # also check run_config if engine not in memory yet
            if not meta["running"]:
                rf = os.path.join(fpath, "run_config.json")
                if os.path.exists(rf):
                    try:
                        meta["running"] = bool(json.load(open(rf)).get("running"))
                    except Exception:
                        pass
            files.append(meta)
        if files:
            out[folder] = files
    return out


def save_strategy_config(uid: str, folder: str, file: str, data: dict) -> str:
    """Create folder/file and write config.json. Returns absolute strategy path."""
    path = strategy_dir(uid, folder, file)
    os.makedirs(path, exist_ok=True)
    data = dict(data)
    data["folder"] = _safe_name(folder)
    data["file"] = _safe_name(file)
    data["updated"] = str(now_ist())
    cfg_path = os.path.join(path, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(data, f, indent=1, default=str)
    # verify write
    if not os.path.exists(cfg_path):
        raise RuntimeError(f"Failed to save config at {cfg_path}")
    return path


def load_strategy_config(path: str) -> dict:
    cfg_path = os.path.join(path, "config.json")
    if os.path.exists(cfg_path):
        try:
            return json.load(open(cfg_path))
        except Exception:
            pass
    return {}


def delete_strategy(uid: str, folder: str, file: str):
    path = strategy_dir(uid, folder, file)
    eng = _get_engine_by_path(uid, path)
    if eng and eng.running:
        eng.stop()
    import shutil
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    fdir = os.path.dirname(path)
    if os.path.isdir(fdir) and not os.listdir(fdir):
        shutil.rmtree(fdir, ignore_errors=True)


# ============================== STATE ==============================
def load_state(path):
    s = json.load(open(path)) if os.path.exists(path) else {}
    for k in ("open", "closed", "handled", "log", "signals"):
        s.setdefault(k, [])
    s.setdefault("slot_last", {})
    s["_path"] = path
    s.pop("waiting", None)
    for g in s["signals"]:
        if str(g.get("status", "")).startswith(("WAITING", "EXPIRED")):
            g["status"] = "SKIPPED (positions were full)"
    return s


def save_state(state, path=None):
    path = path or state.get("_path")
    if not path:
        return
    with open(path, "w") as f:
        json.dump({k: v for k, v in state.items() if not k.startswith("_")}, f, indent=1, default=str)


def add_log(state, now, msg):
    state["log"].append(f"{pd.Timestamp(now):%Y-%m-%d %H:%M:%S}  {msg}")
    state["log"] = state["log"][-200:]


def _fill_slots(state):
    used = {p["slot"] for p in state["open"] if p.get("slot")}
    for p in sorted([p for p in state["open"] if not p.get("slot")], key=lambda p: str(p.get("entry_time"))):
        n = 1
        while n in used:
            n += 1
        p["slot"] = n
        used.add(n)


def _pick_slot(state, now):
    _fill_slots(state)
    used = {p["slot"] for p in state["open"]}
    today = f"{pd.Timestamp(now):%Y-%m-%d}"
    last = state.setdefault("slot_last", {})
    fresh = [
        (str(v.get("exit_time", "")), int(k))
        for k, v in last.items()
        if int(k) not in used and str(v.get("exit_time", ""))[:10] == today
    ]
    if fresh:
        slot = min(fresh)[1]
        return slot, last[str(slot)]
    n = 1
    while n in used:
        n += 1
    return n, None


# ============================== PATTERN OUTLOOK ==============================
def predict_patterns(done, forming, ltp, cfg):
    lb = cfg.trend_lookback
    if len(done) < lb + 2:
        return "Not enough candles"
    p = done.iloc[-1]
    po, pc, pl = float(p["open"]), float(p["close"]), float(p["low"])
    if not pc < po:
        return "No setup (last candle not red)"
    if lb > 0 and not pc < float(done["close"].iloc[-1 - lb]):
        return "No setup (no downtrend)"
    return (
        f"WATCH: Engulfing if close > {po:.2f} | Harami if open > {pc:.2f} and close < {po:.2f} | "
        f"Tweezer if low ~ {pl:.2f}"
    )


# ============================== EXIT ==============================
def _check_exit(pos, bars):
    sl, tg = pos.get("stop_loss"), pos["target"]
    for ts, r in bars[bars.index >= pd.Timestamp(pos["last_checked"])].iterrows():
        if sl:
            if r["open"] <= sl:
                return float(r["open"]), ts, "SL (gap)"
            if r["low"] <= sl:
                return sl, ts, "SL"
        if r["open"] >= tg:
            return float(r["open"]), ts, "Target (gap)"
        if r["high"] >= tg:
            return tg, ts, "Target"
    return None


def _close(state, cfg, pos, raw_px, when, reason):
    exit_px = raw_px * (1 - cfg.slippage_pct / 100)
    qty = pos["qty"]
    pnl = (exit_px - pos["entry"]) * qty - cfg.brokerage_pct / 100 * (pos["entry"] + exit_px) * qty
    risk = (pos["entry"] - pos["stop_loss"]) * qty if pos.get("stop_loss") else 0
    done = {k: pos[k] for k in ("symbol", "exchange", "pattern", "signal_date", "entry_time",
                                "entry", "stop_loss", "target", "qty")}
    done.update({k: pos[k] for k in CANDLE_KEYS + ("signal_close",) if k in pos})
    done.update({
        "exit_time": str(pd.Timestamp(when).floor("s")), "exit": round(exit_px, 2),
        "reason": reason, "pnl": round(pnl, 2),
        "r_multiple": round(pnl / risk, 2) if risk > 0 else None,
    })
    _fill_slots(state)
    slot = pos.get("slot")
    done["slot"] = slot
    state["closed"].append(done)
    state["open"] = [p for p in state["open"] if p is not pos]
    if slot:
        state.setdefault("slot_last", {})[str(slot)] = {
            "symbol": pos["symbol"], "reason": reason, "exit_time": done["exit_time"],
        }
    add_log(
        state, when,
        f"CLOSED {pos['symbol']} @ {exit_px:.2f}  {reason}  P&L {pnl:,.2f}"
        f"{f'  (position #{slot} is free now)' if slot else ''}",
    )


def check_exits(state, cfg, now, bars_fn, ignore_hours=False):
    for pos in list(state["open"]):
        try:
            bars = bars_fn(pos, pd.Timestamp(pos["last_checked"]))
        except Exception as ex:
            add_log(state, now, f"price error {pos['symbol']}: {ex}")
            continue
        if bars is None or bars.empty:
            continue
        hit = _check_exit(pos, bars)
        if hit:
            _close(state, cfg, pos, hit[0], hit[1], hit[2])
            continue
        pos.setdefault("last_price", float(bars["close"].iloc[-1]))
        pos["last_checked"] = str(bars.index[-1])
        held = int(np.busday_count(pos["entry_time"][:10], str(now.date()))) + 1
        if held >= cfg.max_hold and (ignore_hours or now.time() >= exit_time(pos["exchange"])):
            _close(state, cfg, pos, pos["last_price"], now, "Time exit")


# ============================== ENGINE ==============================
def margin_factor(exchange):
    return FUTURES_MARGIN_PCT / 100 if exchange in ("NFO", "MCX") else 1.0


def candle_boundary(now, exchange, tf_min):
    start = now.normalize() + (pd.Timedelta(hours=9) if exchange == "MCX" else pd.Timedelta(hours=9, minutes=15))
    if now < start:
        return start
    step = pd.Timedelta(minutes=tf_min)
    return start + ((now - start) // step) * step


class Engine:
    GRACE = 4
    EXIT_EVERY = 15
    QUOTE_EVERY = 1.0

    def __init__(self, uid, strategy_path):
        self.uid = uid
        self.strategy_path = strategy_path
        os.makedirs(strategy_path, exist_ok=True)
        self.state_file = os.path.join(strategy_path, "live_state.json")
        self.run_file = os.path.join(strategy_path, "run_config.json")
        self.lock = threading.RLock()
        self.thread = None
        self.stop_evt = threading.Event()
        self.qthread = None
        self.state = load_state(self.state_file)
        self.smart = self.cfg = None
        self.creds, self.client_id, self.login_time = None, "", 0.0
        self.items, self.exchange, self.ignore = [], "NSE", False
        self._clear_runtime()

    def _clear_runtime(self):
        self.cache, self.quotes, self.rows = {}, {}, []
        self.quote_errors, self.cycle_errors = [], []
        self.status, self.last_update, self.last_exit_check = "", None, 0.0

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive() and not self.stop_evt.is_set()

    def set_login(self, smart, creds, client_id):
        self.smart, self.creds, self.client_id, self.login_time = smart, creds, client_id, time.time()

    def _save_run_config(self, running):
        if self.cfg is None:
            return
        try:
            with open(self.run_file, "w") as f:
                json.dump(
                    {
                        "running": running, "exchange": self.exchange, "ignore": self.ignore,
                        "items": self.items, "cfg": asdict(self.cfg),
                    },
                    f, default=str,
                )
        except Exception:
            pass

    def try_autoresume(self, shared_smart=None, shared_creds=None, shared_client_id=""):
        try:
            if not os.path.exists(self.run_file):
                return
            rc = json.load(open(self.run_file))
            if not rc.get("running"):
                return
            smart = shared_smart
            creds = shared_creds
            if smart is None:
                creds = env_creds()
                if not creds:
                    return
                smart, err = angel_login(*creds)
                if err:
                    self.status = f"Auto-resume login failed: {err}"
                    return
            cd = rc["cfg"]
            cd["patterns"] = tuple(cd.get("patterns", PATTERNS))
            self.set_login(smart, creds or shared_creds, (shared_client_id or (creds[1] if creds else "")).strip().upper())
            self.start(smart, Config(**cd), rc["items"], rc["exchange"], rc.get("ignore", False))
            with self.lock:
                add_log(self.state, now_ist(), "Auto-resumed after restart")
        except Exception as ex:
            self.status = f"Auto-resume failed: {ex}"

    def start(self, smart, cfg, items, exchange, ignore_hours=False):
        self.stop()
        with self.lock:
            self.smart, self.cfg, self.items = smart, cfg, list(items)[:50]
            self.exchange, self.ignore = exchange, ignore_hours
            self._clear_runtime()
            self.status = "Starting..."
            self.stop_evt = threading.Event()
            self.thread = threading.Thread(target=self._loop, args=(self.stop_evt,), daemon=True)
            self.qthread = threading.Thread(target=self._quote_loop, args=(self.stop_evt,), daemon=True)
            self.thread.start()
            self.qthread.start()
        self._save_run_config(True)

    def stop(self):
        self.stop_evt.set()
        for t in (self.thread, self.qthread):
            if t is not None and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=2)
        self.thread = self.qthread = None
        self.status = "Stopped"
        self._save_run_config(False)

    def reset_all(self):
        self.stop()
        with self.lock:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
            self.state = load_state(self.state_file)
            self._clear_runtime()

    def close_now(self, key):
        with self.lock:
            for p in self.state["open"]:
                if f"{p['exchange']}:{p['symbol']}" == key:
                    _close(self.state, self.cfg or Config(), p, p["last_price"], now_ist(), "Manual exit")
                    save_state(self.state)
                    return

    def delete_position(self, key):
        with self.lock:
            self.state["open"] = [p for p in self.state["open"] if f"{p['exchange']}:{p['symbol']}" != key]
            save_state(self.state)

    def snapshot(self):
        with self.lock:
            return {
                "state": copy.deepcopy(self.state), "rows": list(self.rows),
                "errors": list(self.quote_errors) + list(self.cycle_errors),
                "status": self.status, "last_update": self.last_update, "running": self.running,
                "cfg": self.cfg, "exchange": self.exchange,
            }

    def _loop(self, evt):
        while not evt.is_set():
            t0 = time.time()
            try:
                self._cycle(evt)
            except Exception as ex:
                with self.lock:
                    self.cycle_errors = [f"engine error: {ex}"]
            evt.wait(max(1.0, 3.0 - (time.time() - t0)))

    def _quote_loop(self, evt):
        while not evt.is_set():
            t0 = time.time()
            try:
                cfg, now = self.cfg, now_ist()
                if cfg is None:
                    evt.wait(1)
                    continue
                live_ok = self.ignore or market_open(now, self.exchange)
                B = candle_boundary(now, self.exchange, cfg.tf_min)
                errs = []
                self._refresh_quotes(self.items, live_ok, errs, evt, B)
                if evt.is_set():
                    return
                with self.lock:
                    self.quote_errors = errs
                self._build_rows(None, B, pd.Timedelta(minutes=cfg.tf_min), live_ok, final=False, persist=False)
            except Exception as ex:
                with self.lock:
                    self.quote_errors = [f"price error: {ex}"]
            evt.wait(max(0.2, self.QUOTE_EVERY - (time.time() - t0)))

    def _candle_due(self, key, B, now, live_ok):
        c = self.cache.get(key)
        if c is None:
            return True
        if c.get("retry_at") is not None:
            return time.time() >= c["retry_at"]
        if c["b"] == B or not live_ok:
            return False
        return now >= B + pd.Timedelta(seconds=self.GRACE)

    def _refresh_quotes(self, items, live_ok, errors, evt, B):
        got = fetch_quotes_bulk(self.smart, items)
        for it in items:
            if evt.is_set():
                return
            key = (it["exchange"], it["symbol"])
            q = got.get(str(it["token"]))
            if q is None:
                q = fetch_quote(self.smart, it)
                time.sleep(0.1)
            if not q:
                errors.append(
                    f"{it['symbol']}: no live price "
                    f"({LAST_API_ERROR['quote'] or LAST_API_ERROR['bulk'] or 'no reply'})"
                )
                continue
            with self.lock:
                self.quotes[key] = q
                c = self.cache.get(key)
                if c:
                    f = c.get("forming")
                    if live_ok and (f is None or f.get("b") != B):
                        if f is not None and f.get("close") is not None:
                            c["roll"] = {"b": B, "close": f["close"]}
                        f = c["forming"] = {
                            "open": q["ltp"], "high": q["ltp"], "low": q["ltp"],
                            "close": q["ltp"], "b": B,
                        }
                    if f:
                        f["high"], f["low"] = max(f["high"], q["ltp"]), min(f["low"], q["ltp"])
                        f["close"] = q["ltp"]
        with self.lock:
            for p in self.state["open"]:
                q = self.quotes.get((p["exchange"], p["symbol"]))
                if q:
                    p["last_price"] = q["ltp"]

    def _build_rows(self, errors, B, tf, live_ok, final=True, persist=True):
        cfg = self.cfg
        with self.lock:
            open_keys = {(p["exchange"], p["symbol"]) for p in self.state["open"]}
            rows = []
            for it in self.items:
                key = (it["exchange"], it["symbol"])
                q, c = self.quotes.get(key), self.cache.get(key)
                row = {
                    "symbol": it["symbol"], "open": None, "high": None, "low": None, "ltp": None,
                    "prev_close": None, "change_%": None, "updated": f"{now_ist():%H:%M:%S}",
                    "position": "IN POSITION" if key in open_keys else "-",
                    "pattern_outlook": "Loading candles...",
                }
                if q:
                    row["ltp"] = q["ltp"]
                if c and c.get("done") is not None and len(c["done"]):
                    d, f = c["done"], c.get("forming")
                    if live_ok and f:
                        o, h, l, pc = f["open"], f["high"], f["low"], float(d["close"].iat[-1])
                        r = c.get("roll")
                        if c.get("b") != B and r and r["b"] == B:
                            pc = r["close"]
                    elif len(d) > 1:
                        o, h, l = (float(d[k].iat[-1]) for k in ("open", "high", "low"))
                        pc = float(d["close"].iat[-2])
                    else:
                        o = h = l = pc = None
                    if o is not None:
                        row.update({
                            "open": round(o, 2), "high": round(h, 2), "low": round(l, 2),
                            "prev_close": round(pc, 2),
                            "change_%": round((q["ltp"] / pc - 1) * 100, 2) if q and pc else None,
                        })
                if c and c.get("done") is not None:
                    row["pattern_outlook"] = predict_patterns(
                        c["done"], c.get("forming"), q["ltp"] if q else None, cfg
                    )
                rows.append(row)
            self.rows, self.last_update = rows, now_ist()
            if errors is not None:
                self.cycle_errors = list(errors)
            if final:
                self.status = (
                    "Waiting for the next candle (" + f"{(B + tf):%H:%M}" + ")"
                    if live_ok
                    else "Market closed - prices only, no entry / exit"
                )
            if persist:
                save_state(self.state)

    def _cycle(self, evt):
        if self.creds and time.time() - self.login_time > 8 * 3600:
            sm, err = angel_login(*self.creds)
            if sm is not None:
                self.smart, self.login_time = sm, time.time()
        cfg, items, smart, exch = self.cfg, self.items, self.smart, self.exchange
        if cfg is None or smart is None:
            return
        now = now_ist()
        live_ok = self.ignore or market_open(now, exch)
        tf = pd.Timedelta(minutes=cfg.tf_min)
        B = candle_boundary(now, exch, cfg.tf_min)
        errors = []

        due = [it for it in items if self._candle_due((it["exchange"], it["symbol"]), B, now, live_ok)]
        if due:
            from concurrent.futures import ThreadPoolExecutor
            counter = [0]

            def load_one(it):
                if evt.is_set():
                    return
                try:
                    self._load_candles(it, B, tf, live_ok)
                except Exception as ex:
                    errors.append(f"{it['symbol']}: {ex}")
                counter[0] += 1
                self.status = f"Loading candles {counter[0]}/{len(due)}"

            self.status = f"Loading candles 0/{len(due)}"
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(load_one, due))
            if evt.is_set():
                return

        if live_ok and time.time() - self.last_exit_check >= self.EXIT_EVERY:
            with self.lock:
                snap = [dict(p) for p in self.state["open"]]
            bars_map = {}
            for p in snap:
                if evt.is_set():
                    return
                self.status = f"Checking exit {p['symbol']}"
                try:
                    bars_map[(p["exchange"], p["symbol"])] = fetch_bars(
                        smart, p, pd.Timestamp(p["last_checked"]), now_ist()
                    )
                except Exception as ex:
                    errors.append(f"{p['symbol']}: exit check failed ({ex})")
            with self.lock:
                check_exits(
                    self.state, cfg, now_ist(),
                    lambda it, since: bars_map.get((it["exchange"], it["symbol"])),
                    self.ignore,
                )
                save_state(self.state)
            self.last_exit_check = time.time()

        self._build_rows(errors, B, tf, live_ok, final=True)

    def _load_candles(self, it, B, tf, live_ok):
        cfg = self.cfg
        key = (it["exchange"], it["symbol"])
        df = fetch_candles(self.smart, it, cfg.interval, now_ist())
        if df is None or df.empty:
            fails = self.cache.get(key, {}).get("fails", 0) + 1
            with self.lock:
                self.cache[key] = {
                    "b": None if fails < 3 else B, "done": None, "forming": None, "tries": 0,
                    "fails": fails if fails < 3 else 0,
                    "retry_at": time.time() + 20 if fails < 3 else None,
                }
            raise RuntimeError(f"no candle data (try {fails})")
        nowf = now_ist()
        mask = (df["date"] + tf) <= nowf
        done, run = df[mask].reset_index(drop=True), df[~mask]
        forming = None if run.empty else {**{k: float(run.iloc[-1][k]) for k in ("open", "high", "low")}, "b": B}
        tries = self.cache.get(key, {}).get("tries", 0) + 1
        lag = live_ok and not done.empty and done["date"].iat[-1] < B - tf and tries < 3
        with self.lock:
            self.cache[key] = {
                "b": None if lag else B, "done": done, "forming": forming,
                "tries": tries if lag else 0, "roll": self.cache.get(key, {}).get("roll"),
            }
        if not lag:
            self._try_entry(it, done)

    def _skip(self, rec, now, key, why):
        state = self.state
        state["handled"] = (state["handled"] + [key])[-2000:]
        rec["status"] = f"SKIPPED ({why})"
        state["signals"] = (state["signals"] + [rec])[-200:]
        add_log(state, now, f"SKIPPED {rec['symbol']}: {why}  ({rec['pattern']})")
        save_state(state)

    def _enter(self, it, rec, ctime, price, now):
        cfg, state = self.cfg, self.state
        ex, sym = it["exchange"], it["symbol"]
        tf = pd.Timedelta(minutes=cfg.tf_min)
        entry = price * (1 + cfg.slippage_pct / 100)
        lot = int(it.get("lot", 1))
        mf = margin_factor(ex)
        cash = (
            cfg.capital + sum(t["pnl"] for t in state["closed"])
            - sum(p["entry"] * p["qty"] * p.get("mf", 1.0) for p in state["open"])
        )
        one_lot = entry * lot * mf
        by_lots = cfg.lots_per_position > 0 and ex in ("MCX", "NFO")
        if by_lots:                                         # you chose the number of lots
            n_lots = cfg.lots_per_position
            qty = n_lots * lot if one_lot * n_lots <= cash else 0
        else:                                               # amount per position
            qty = math.floor(min(cfg.per_position, cash) / one_lot) * lot
        if qty < 1 and by_lots:
            need = one_lot * cfg.lots_per_position
            rec["status"] = f"SKIPPED ({cfg.lots_per_position} lot(s) need {need:,.0f})"
            add_log(
                state, now,
                f"SKIPPED {sym}: {cfg.lots_per_position} lot(s) = {cfg.lots_per_position * lot} units need "
                f"{need:,.0f} margin (price {entry:.2f}{f' x margin {mf * 100:g}%' if mf < 1 else ''}) "
                f"but free cash is {cash:,.0f}",
            )
        elif qty < 1:
            rec["status"] = f"SKIPPED (1 lot needs {one_lot:,.0f})"
            add_log(
                state, now,
                f"SKIPPED {sym}: 1 lot needs {one_lot:,.0f} (price {entry:.2f} x lot {lot}"
                f"{f' x margin {mf * 100:g}%' if mf < 1 else ''}) but amount per position is "
                f"{cfg.per_position:,.0f}, free cash {cash:,.0f}",
            )
        else:
            slot, replaced = _pick_slot(state, now)
            state["open"].append({
                "slot": slot,
                "symbol": sym, "tradingsymbol": it["tradingsymbol"], "token": it["token"], "exchange": ex,
                "pattern": rec["pattern"], "signal_date": str(ctime), "signal_close": str(ctime + tf),
                "entry_time": str(now.floor("s")),
                "entry": round(entry, 2),
                "stop_loss": round(entry * (1 - cfg.sl_pct / 100), 2) if cfg.sl_pct > 0 else None,
                "target": round(entry * (1 + cfg.target_pct / 100), 2), "qty": int(qty), "mf": mf,
                "last_price": price, "last_checked": str(now.floor("min")),
                **{k: rec[k] for k in CANDLE_KEYS},
            })
            rec["status"] = "ENTERED"
            where = f"position #{slot}" + ("" if cfg.no_limit else f"/{cfg.max_open}")
            if replaced:
                where += (
                    f", took the place of {replaced['symbol']} "
                    f"({replaced['reason']} {str(replaced['exit_time'])[11:]})"
                )
            add_log(
                state, now,
                f"ENTRY {sym} @ {entry:.2f}  qty {qty}  TGT {entry * (1 + cfg.target_pct / 100):.2f}  "
                f"({rec['pattern']})  -> {where}",
            )
        state["signals"] = (state["signals"] + [rec])[-200:]
        save_state(state)

    def _try_entry(self, it, done):
        cfg, smart = self.cfg, self.smart
        now = now_ist()
        ex, sym = it["exchange"], it["symbol"]
        if not (self.ignore or market_open(now, ex)):
            return
        if len(done) < cfg.trend_lookback + 3:
            return
        sig = detect_patterns(done, cfg)
        names = [p for p in cfg.patterns if bool(sig[p].iat[-1])]
        if not names:
            return
        tf = pd.Timedelta(minutes=cfg.tf_min)
        ctime = done["date"].iat[-1]
        if not self.ignore and now - (ctime + tf) > min(2 * tf, pd.Timedelta(seconds=FRESH_SECONDS)):
            return
        key = f"{ex}:{sym}|{ctime}"
        with self.lock:
            if key in self.state["handled"]:
                return
        pv, sg = done.iloc[-2], done.iloc[-1]
        candles = {
            f"{p}_{k}": round(float(c[n]), 2)
            for p, c in (("prev", pv), ("sig", sg))
            for k, n in (("O", "open"), ("H", "high"), ("L", "low"), ("C", "close"))
        }
        rec = {
            "time": str(now.floor("s")), "symbol": sym, "exchange": ex,
            "pattern": "+".join(names), "candle": str(ctime), **candles, "status": "ENTERED",
        }

        def blocked():
            st_ = self.state
            if not self.ignore and now.time() >= exit_time(ex):
                return "skip", f"no new entries after {exit_time(ex):%H:%M}"
            if any(p["exchange"] == ex and p["symbol"] == sym for p in st_["open"]):
                return "skip", "already in position"
            if len(st_["open"]) >= cfg.max_open:
                return "skip", f"positions full {len(st_['open'])}/{cfg.max_open}"
            return None

        def handle(b):
            self._skip(rec, now, key, b[1])

        with self.lock:
            b = blocked()
            if b:
                handle(b)
                return
        price = fetch_ltp(smart, it) or float(done["close"].iat[-1])
        with self.lock:
            b = blocked()
            if b:
                handle(b)
                return
            self.state["handled"] = (self.state["handled"] + [key])[-2000:]
            self._enter(it, rec, ctime, price, now)


# ============================== ENGINE REGISTRY ==============================
def env_creds():
    c = tuple(os.getenv(k, "").strip() for k in ("ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_MPIN", "ANGEL_TOTP_SECRET"))
    if all(c):
        return c
    try:
        a = st.secrets["angel"]
        c = tuple(str(a.get(k, "")).strip() for k in ("api_key", "client_id", "mpin", "totp_secret"))
        if all(c):
            return c
    except Exception:
        pass
    return None


def owner_email():
    v = os.getenv("OWNER_EMAIL", "").strip().lower()
    if not v:
        try:
            v = str(st.secrets["access_control"]["owner_email"]).strip().lower()
        except Exception:
            v = ""
    return v


def current_uid():
    em = (st.session_state.get("access_email") or "").strip().lower()
    if not em:
        return None
    if em == owner_email():
        return OWNER_ID
    return "u_" + hashlib.sha256(em.encode()).hexdigest()[:16]


@st.cache_resource
def _engine_registry():
    return {"lock": threading.Lock(), "map": {}}


@st.cache_resource
def _login_cache():
    """Server-memory Angel sessions per uid.
    Survives page refresh / laptop sleep while the Streamlit process is alive.
    Cleared only on explicit Logout or full app restart."""
    return {"lock": threading.Lock(), "map": {}}


def save_login(uid, smart, creds, client_id):
    if not uid or smart is None:
        return
    cache = _login_cache()
    with cache["lock"]:
        cache["map"][uid] = {
            "smart": smart,
            "creds": creds,
            "client_id": client_id or "",
            "login_time": time.time(),
        }
        cache.setdefault("out", set()).discard(uid)


def load_login(uid):
    if not uid:
        return None
    cache = _login_cache()
    with cache["lock"]:
        return cache["map"].get(uid)


def clear_login(uid):
    if not uid:
        return
    cache = _login_cache()
    with cache["lock"]:
        cache["map"].pop(uid, None)


def mark_logged_out(uid, flag=True):
    """Logout = the BROWSER leaves. Engines keep running; this flag only stops a refresh from re-attaching silently."""
    if not uid:
        return
    cache = _login_cache()
    with cache["lock"]:
        out = cache.setdefault("out", set())
        (out.add if flag else out.discard)(uid)


def is_logged_out(uid):
    cache = _login_cache()
    with cache["lock"]:
        return uid in cache.setdefault("out", set())


def get_strategy_engine(uid, strategy_path):
    """One Engine per (user, strategy folder/file)."""
    reg = _engine_registry()
    key = f"{uid}::{strategy_path}"
    with reg["lock"]:
        eng = reg["map"].get(key)
        if eng is None:
            eng = reg["map"][key] = Engine(uid, strategy_path)
        return eng


def _get_engine_by_path(uid, strategy_path):
    reg = _engine_registry()
    key = f"{uid}::{strategy_path}"
    with reg["lock"]:
        return reg["map"].get(key)


def share_login_to_engines(uid, smart, creds, client_id):
    """Push Angel login to all engines of this user (so any file can Start)."""
    reg = _engine_registry()
    with reg["lock"]:
        for k, eng in reg["map"].items():
            if k.startswith(f"{uid}::"):
                eng.set_login(smart, creds, client_id)


def autoresume_all(uid, smart, creds, client_id):
    """After login / restart: resume any file that was RUNNING."""
    root = user_strategies_root(uid)
    if not os.path.isdir(root):
        return
    for folder in os.listdir(root):
        fp = os.path.join(root, folder)
        if not os.path.isdir(fp):
            continue
        for fname in os.listdir(fp):
            fpath = os.path.join(fp, fname)
            if not os.path.isdir(fpath):
                continue
            run_file = os.path.join(fpath, "run_config.json")
            if not os.path.exists(run_file):
                continue
            try:
                rc = json.load(open(run_file))
                if not rc.get("running"):
                    continue
            except Exception:
                continue
            eng = get_strategy_engine(uid, fpath)
            if not eng.running:
                threading.Thread(
                    target=eng.try_autoresume,
                    args=(smart, creds, client_id),
                    daemon=True,
                ).start()


# ============================== ACCESS CONTROL ==============================
ACCESS_APPS_SCRIPT_URL = (
    "https://script.google.com/macros/s/AKfycbw5oSPyD4bjUN6bL8YqsWH94kHo_HvRWszcLecpLDm1wZDO7p6F34nmDSwN-w5SUWo63g/exec"
)


def _get_access_secret():
    try:
        return str(st.secrets["access_control"]["request_secret"]).strip()
    except Exception:
        return ""


def _access_call(action, name="", email=""):
    secret = _get_access_secret()
    if not secret:
        return {
            "status": "ERROR",
            "message": "Access secret is missing. Add [access_control] request_secret in Streamlit Secrets.",
        }
    payload = {
        "action": action, "name": name.strip(), "email": email.strip().lower(),
        "secret": secret, "requested_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        response = requests.post(ACCESS_APPS_SCRIPT_URL, json=payload, timeout=15)
        if response.status_code != 200:
            return {"status": "ERROR", "message": f"Access server returned HTTP {response.status_code}."}
        try:
            return response.json()
        except Exception:
            return {"status": "ERROR", "message": "Invalid response received from access server."}
    except requests.RequestException as exc:
        return {"status": "ERROR", "message": f"Could not contact access server: {exc}"}


def _valid_access_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))


def _saved_email():
    return (st.session_state.get("access_email") or str(st.query_params.get("e", ""))).strip().lower()


def access_request_page():
    st.title("🔐 Request Access")
    st.caption("Administrator approval is required before using the Live Paper Trading application.")
    saved_email = _saved_email()
    saved_name = st.session_state.get("access_name", "")
    if saved_email:
        result = _access_call("check_access", email=saved_email)
        status = str(result.get("status", "")).upper()
        if status == "APPROVED":
            st.session_state["access_email"] = saved_email
            st.session_state["access_granted"] = True
            st.rerun()
        elif status == "PENDING":
            st.info("⏳ Your access request is pending admin approval.")
        elif status == "REJECTED":
            st.error("❌ Your access request was rejected by the administrator.")
        elif status == "ERROR":
            st.warning(result.get("message", "Unable to check access status."))

    with st.form("access_request_form"):
        name = st.text_input("Name", value=saved_name, placeholder="Enter your name")
        email = st.text_input("Email ID", value=saved_email, placeholder="Enter your email address")
        submitted = st.form_submit_button("Request Access", type="primary", use_container_width=True)
    if not submitted:
        return
    name, email = name.strip(), email.strip().lower()
    if not name:
        st.error("Please enter your name.")
        return
    if not _valid_access_email(email):
        st.error("Please enter a valid email address.")
        return
    st.session_state["access_name"] = name
    st.session_state["access_email"] = email
    st.query_params["e"] = email
    check = _access_call("check_access", email=email)
    status = str(check.get("status", "NEW")).upper()
    if status == "APPROVED":
        st.session_state["access_granted"] = True
        st.rerun()
    if status == "PENDING":
        st.info("⏳ This email already has a pending access request.")
        return
    if status == "REJECTED":
        st.error("❌ This email was rejected by the administrator.")
        return
    if status == "ERROR":
        st.error(check.get("message", "Could not check access status."))
        return
    result = _access_call("request_access", name=name, email=email)
    result_status = str(result.get("status", "ERROR")).upper()
    if result_status == "APPROVED":
        st.session_state["access_granted"] = True
        st.rerun()
    elif result_status == "PENDING":
        st.success("✅ Access request submitted successfully. Please wait for administrator approval.")
    elif result_status == "REJECTED":
        st.error("❌ Access was rejected by the administrator.")
    else:
        st.error(result.get("message", "Could not submit the access request."))
    st.caption("After approval, refresh this page. The Angel One login will then appear.")


def access_gate():
    ss = st.session_state
    if ss.get("access_granted", False):
        em = ss.get("access_email", "")
        if em and st.query_params.get("e") != em:
            st.query_params["e"] = em
        return True
    email = _saved_email()
    if email:
        result = _access_call("check_access", email=email)
        if str(result.get("status", "")).upper() == "APPROVED":
            ss["access_email"] = email
            ss["access_granted"] = True
            st.query_params["e"] = email
            return True
    access_request_page()
    return False


# ============================== UI HELPERS ==============================
def excel_check_frame(entries):
    rows = []
    rec = sorted([e for e in entries if all(k in e for k in CANDLE_KEYS)], key=lambda e: str(e.get("entry_time")))
    for k, e in enumerate(rec):
        r0, r1 = 2 + 2 * k, 3 + 2 * k
        rows.append([
            f"{e['symbol']} PREVIOUS candle", e["prev_O"], e["prev_H"], e["prev_L"], e["prev_C"], "", "", "", "",
        ])
        rows.append([
            f"{e['symbol']} SIGNAL candle {e.get('signal_date', '')} to {e.get('signal_close', '?')} "
            f"(entry {e.get('entry_time', '')})",
            e["sig_O"], e["sig_H"], e["sig_L"], e["sig_C"],
            f'=IF(AND(B{r0}>E{r0},B{r1}<E{r1},B{r1}<=E{r0},E{r1}>=B{r0}),"BULLISH","")',
            f'=IF(AND(B{r0}>E{r0},B{r1}<E{r1},B{r1}>E{r0},E{r1}<B{r0}),"BULLISH harami","")',
            f'=IF(AND(E{r0}<B{r0},E{r1}>B{r1},ABS(D{r1}-D{r0})<=0.001*D{r0}),"TWEEZER BOTTOM","")',
            e.get("pattern", ""),
        ])
    return pd.DataFrame(
        rows,
        columns=["candle", "Open", "High", "Low", "Close", "Engulfing (F)", "Harami (G)", "Tweezer (H)", "App pattern (I)"],
    )


def log_frame(lines, n=100):
    rows = []
    for ln in reversed(lines[-n:]):
        t, _, msg = ln.partition("  ")
        parts = msg.split(" ", 2)
        if parts[0] in ("ENTRY", "CLOSED", "SKIPPED") and len(parts) >= 2:
            ev, sym, rest = parts[0], parts[1].rstrip(":"), (parts[2] if len(parts) > 2 else "")
        else:
            ev, sym, rest = "INFO", "", msg
        rows.append({"time": t, "event": ev, "symbol": sym, "details": rest})
    return pd.DataFrame(rows, columns=["time", "event", "symbol", "details"])


def _frames(engine):
    snap = engine.snapshot()
    state, cfg = snap["state"], snap["cfg"] or Config()
    open_df, closed_df = pd.DataFrame(state["open"]), pd.DataFrame(state["closed"])
    if not closed_df.empty:
        old_pat = closed_df["pattern"] if "pattern" in closed_df.columns else [None] * len(closed_df)
        closed_df["pattern"] = [classify_candles(r) or p for r, p in zip(closed_df.to_dict("records"), old_pat)]
        order = [
            "slot", "symbol", "exchange", "signal_date", "signal_close", *CANDLE_KEYS, "pattern",
            "entry_time", "entry", "qty", "exit_time", "exit", "reason", "pnl", "r_multiple",
        ]
        closed_df = closed_df[[c for c in order if c in closed_df.columns]].rename(
            columns={"slot": "position", "signal_date": "signal_candle_start", "signal_close": "signal_candle_close"}
        )
        closed_df["pnl_%"] = ((closed_df["exit"] / closed_df["entry"] - 1) * 100).round(2)
    if not open_df.empty:
        _fill_slots(state)
        open_df = pd.DataFrame(state["open"]).rename(columns={"slot": "position"}).sort_values("position")
        open_df["live_price"] = open_df["last_price"].round(2)
        open_df["PnL"] = ((open_df["last_price"] - open_df["entry"]) * open_df["qty"]).round(2)
        open_df["PnL_%"] = ((open_df["last_price"] / open_df["entry"] - 1) * 100).round(2)
    return snap, state, cfg, open_df, closed_df


def _csv_button(label, df, name, key):
    empty = df is None or df.empty
    st.download_button(
        label, b"" if empty else df.to_csv(index=False).encode(),
        f"{name}_{now_ist():%Y%m%d}.csv", "text/csv", key=key, disabled=empty,
    )


# ============================== LIVE SECTIONS (per engine) ==============================
def live_top(engine, exchange, tf_label):
    snap, state, cfg, open_df, closed_df = _frames(engine)
    if snap["running"]:
        exchange = snap["exchange"]
        tf_label = next((k for k, v in TIMEFRAMES.items() if v[1] == cfg.tf_min), tf_label)
    now = now_ist()
    hrs = "9:00-23:30" if exchange == "MCX" else "9:15-15:30"
    mkt = (
        f"{exchange} market OPEN" if market_open(now, exchange)
        else f"{exchange} market CLOSED (entry/exit run {hrs} IST, Mon-Fri)"
    )
    upd = f"{snap['last_update']:%H:%M:%S}" if snap["last_update"] is not None else "-"
    pp_txt = f"{cfg.lots_per_position} lot(s)" if cfg.lots_per_position else f"{cfg.per_position:,.0f}"
    st.write(
        f"**{'RUNNING' if snap['running'] else 'STOPPED'}**  |  {mkt}  |  now {now:%H:%M:%S} IST  |  "
        f"prices updated {upd}  |  timeframe: {tf_label or '-'}  |  per position: {pp_txt}"
    )
    if snap["running"]:
        st.caption(f"Engine: {snap['status']}")

    realized = float(closed_df["pnl"].sum()) if not closed_df.empty else 0.0
    unreal = float(open_df["PnL"].sum()) if not open_df.empty else 0.0
    win = (closed_df["pnl"] > 0).mean() * 100 if not closed_df.empty else 0.0
    m = st.columns(5)
    m[0].metric("Open positions", f"{len(open_df)} / {'no limit' if cfg.no_limit else cfg.max_open}")
    m[1].metric("Closed trades", len(closed_df))
    m[2].metric("Realized P&L", f"{realized:,.0f}")
    invested = float((open_df["entry"] * open_df["qty"]).sum()) if not open_df.empty else 0.0
    m[3].metric("Unrealized P&L", f"{unreal:,.0f}", f"{unreal / invested * 100:.2f}%" if invested else None)
    m[4].metric("Win rate (closed)", f"{win:.0f}%")

    st.subheader("1. Monitoring - live data + pattern outlook")
    mon_df = pd.DataFrame(snap["rows"])
    if mon_df.empty:
        st.write("Start pannina, live data + pattern outlook inge varum.")
    else:
        st.caption(
            f"ltp = live price (every second) | open / high / low = running {tf_label or 'timeframe'} candle | "
            "prev_close = previous candle close | change_% = ltp vs previous candle close | "
            "pattern_outlook changes only when a candle closes"
        )
        if mon_df["ltp"].isna().all() and snap["errors"]:
            st.warning("Live price varala: " + " | ".join(snap["errors"][:3]))
        st.dataframe(
            mon_df, use_container_width=True, hide_index=True,
            column_config={"pattern_outlook": st.column_config.TextColumn("pattern_outlook", width="large")},
        )


def live_mid(engine):
    snap, state, cfg, open_df, closed_df = _frames(engine)
    _csv_button("Download monitoring CSV", pd.DataFrame(snap["rows"]), "monitoring", "dl_monitor")
    st.subheader("2. Signals")
    sig_df = pd.DataFrame(state["signals"])
    if sig_df.empty:
        st.write("Signals illa (candle mudiyum bodhu pattern vandha inge varum).")
    else:
        st.dataframe(sig_df.iloc[::-1], use_container_width=True, hide_index=True)
    _csv_button("Download signals CSV", sig_df, "signals", "dl_signals")
    if snap["errors"]:
        with st.expander(f"{len(snap['errors'])} warning(s)"):
            st.write("\n".join(snap["errors"]))


def live_positions(engine):
    snap, state, cfg, open_df, closed_df = _frames(engine)
    st.subheader("3. Live positions (entry aana, exit-ku wait)")
    cols = [c for c in ["position", "symbol", "exchange", "pattern", "entry_time", "entry", "live_price", "PnL", "PnL_%"] if c in open_df.columns]
    if open_df.empty:
        st.write("Open positions illa.")
    else:
        st.dataframe(open_df[cols], use_container_width=True, hide_index=True)


def live_bottom(engine):
    snap, state, cfg, open_df, closed_df = _frames(engine)
    cols = [c for c in ["position", "symbol", "exchange", "pattern", "entry_time", "entry", "live_price", "PnL", "PnL_%"] if c in open_df.columns]
    _csv_button("Download live positions CSV", open_df[cols] if not open_df.empty else open_df, "live_positions", "dl_open")

    st.subheader("4. Closed trades (exit aanadhu)")
    show = ["position", "symbol", "exchange", "pattern", "entry_time", "entry", "qty", "exit_time", "exit", "reason", "pnl", "pnl_%", "r_multiple"]
    if closed_df.empty:
        st.write("Closed trades illa.")
    else:
        st.dataframe(
            closed_df[[c for c in show if c in closed_df.columns]].rename(columns={"pnl": "PnL"}).iloc[::-1],
            use_container_width=True, hide_index=True,
        )
    _csv_button("Download closed trades CSV", closed_df, "closed_trades", "dl_closed")

    st.subheader("5. Logs")
    log_df = log_frame(state["log"])
    if log_df.empty:
        st.write("No events yet.")
    else:
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    _csv_button("Download logs CSV", log_df, "logs", "dl_log")

    st.subheader("6. Excel check - every entry's 2 candles + your 3 formulas")
    check_df = excel_check_frame(state["open"] + state["closed"])
    if check_df.empty:
        st.write("Entry aana piragu inge varum (pazhaiya entries-ku candle data illa).")
    else:
        st.caption(
            "CSV-a Excel-la open pannunga. B=Open, C=High, D=Low, E=Close. F, G, H = unga formulas, I = app sonna pattern."
        )
        st.dataframe(check_df.iloc[:, :5], use_container_width=True, hide_index=True)
    _csv_button("Download Excel check CSV (open + closed entries)", check_df, "excel_check", "dl_check")


# ============================== PAGES ==============================
def login_page():
    no_dim()
    if env_creds() and not owner_email():
        st.warning(
            "Server-la Angel login (ANGEL_*) save aagi irukku, aanaa OWNER_EMAIL set pannala, so auto login "
            "yaarukum work aagaadhu. OWNER_EMAIL (illa secrets [access_control] owner_email) set pannunga."
        )
    page = st.empty()
    with page.container():
        st.title("Angel One Login")
        st.caption("Credentials are used only for this session. Nothing is saved to disk.")
        with st.form("login"):
            mine = current_uid() == OWNER_ID
            api_key = st.text_input("API Key", value=os.getenv("ANGEL_API_KEY", "") if mine else "")
            client_id = st.text_input("Client ID", value=os.getenv("ANGEL_CLIENT_ID", "") if mine else "")
            mpin = st.text_input("MPIN", type="password")
            totp_secret = st.text_input("TOTP Secret Key", type="password")
            go = st.form_submit_button("Connect", type="primary")
    if not go:
        return
    if not all([api_key, client_id, mpin, totp_secret]):
        st.error("Ella 4 fields-um fill pannunga.")
        return
    with st.spinner("Connecting to Angel One..."):
        smart, err = angel_login(api_key, client_id, mpin, totp_secret)
    if err:
        st.error(err)
        return
    page.empty()
    st.title("Connected")
    with st.spinner("Angel One stock list load aagudhu..."):
        try:
            load_master()
        except Exception:
            pass
    uid = current_uid()
    creds = (api_key.strip(), client_id.strip(), mpin.strip(), totp_secret.strip())
    cid = client_id.strip().upper()
    save_login(uid, smart, creds, cid)          # survives refresh (server memory)
    share_login_to_engines(uid, smart, creds, cid)
    st.session_state.smart = smart
    st.session_state.client_id = cid
    st.session_state.creds = creds
    st.session_state.pop("no_autologin", None)
    autoresume_all(uid, smart, creds, cid)
    st.rerun()


def _set_nav(page="home", folder="", file=""):
    """Keep page in query params so refresh / new tab with same URL stays on page.
       Email (access) is already in ?e= — that alone prevents access re-request."""
    st.session_state["page"] = page
    qp = dict(st.query_params)
    if st.session_state.get("access_email"):
        qp["e"] = st.session_state["access_email"]
    qp["p"] = page
    if page == "file" and folder and file:
        qp["folder"] = folder
        qp["file"] = file
    else:
        qp.pop("folder", None)
        qp.pop("file", None)
    st.query_params.from_dict(qp)


def strategies_home():
    """Folder / File list + New + open file."""
    uid = current_uid()
    no_dim()
    st.title("Strategies")
    st.caption("Folder → File. Each file has its own Exchange, Stocks, Capital, Timeframe, Start / Stop.")

    with st.sidebar:
        st.write(f"Connected: **{st.session_state.get('client_id', '-')}**")
        st.caption(f"Save path: `{user_strategies_root(uid)}`")
        _reg = _engine_registry()
        with _reg["lock"]:
            n_running = sum(1 for k, e in _reg["map"].items() if k.startswith(f"{uid}::") and e.running)
        if n_running:
            st.caption(f"{n_running} strategy RUNNING - Logout / browser close pannaalum background-la odum. "
                       "Nirutha file-oda **Stop** click pannunga.")
        if st.button("Logout (strategies keep running)"):
            # Only this browser logs out. Engines are NOT stopped and keep their Angel login in server memory
            # (they need it for the daily re-login). Re-login with the API key -> same running strategies.
            mark_logged_out(uid, True)
            clear_login(uid)                     # next visit must enter API key again
            for k in ("smart", "client_id", "creds", "active_strategy"):
                st.session_state.pop(k, None)
            st.session_state["no_autologin"] = True
            # do NOT clear access_granted / access_email
            st.rerun()
        if st.button("+ New strategy / file", type="primary", use_container_width=True):
            st.session_state.pop("create_step", None)
            st.session_state.pop("create_folder", None)
            st.session_state.pop("create_file", None)
            _set_nav("create")
            st.rerun()

    root = user_strategies_root(uid)
    strategies = list_strategies(uid)
    if not strategies:
        st.info("No strategies yet. Sidebar-la **+ New strategy / file** click pannunga.")
        # Show raw disk contents so we can see if files exist but failed to load
        try:
            raw = []
            if os.path.isdir(root):
                for fd in sorted(os.listdir(root)):
                    fp = os.path.join(root, fd)
                    if os.path.isdir(fp):
                        kids = [x for x in os.listdir(fp) if os.path.isdir(os.path.join(fp, x))]
                        raw.append(f"{fd}/ → {kids or '(empty)'}")
            if raw:
                st.warning("Folders on disk but no valid config.json found:\n" + "\n".join(raw))
        except Exception as ex:
            st.caption(f"Could not list disk: {ex}")
        return

    for folder, files in strategies.items():
        st.subheader(f"📁 {folder}")
        for meta in files:
            running = meta.get("running", False)
            badge = "🟢 RUNNING" if running else "⚪ STOPPED"
            exchange = meta.get("exchange", "-")
            tf = meta.get("tf_label", "-")
            capital = meta.get("capital", "-")
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.write(f"**{meta['file']}**  ·  {badge}  ·  {exchange} · {tf} · capital {capital}")
            with col2:
                if st.button("Open", key=f"open_{folder}_{meta['file']}", use_container_width=True):
                    st.session_state["active_strategy"] = {
                        "folder": folder, "file": meta["file"], "path": meta["path"],
                    }
                    _set_nav("file", folder, meta["file"])
                    st.rerun()
            with col3:
                if st.button("Delete", key=f"del_{folder}_{meta['file']}", use_container_width=True):
                    if running:
                        st.warning("Stop first, then delete.")
                    else:
                        delete_strategy(uid, folder, meta["file"])
                        st.rerun()


def create_strategy_page():
    """Step 1: folder + file name. Step 2: settings. Then open file page."""
    uid = current_uid()
    no_dim()
    step = st.session_state.get("create_step", 1)

    if st.button("← Back to Strategies"):
        _set_nav("home")
        st.rerun()

    st.title("New strategy / file")
    st.caption(f"Will save under: `{user_strategies_root(uid)}`")

    if step == 1:
        st.subheader("Step 1 · Folder & File name")
        existing = list(list_strategies(uid).keys())
        folder_mode = st.radio(
            "Folder",
            ["New folder", "Existing folder"] if not existing else ["Existing folder", "New folder"],
            horizontal=True,
        )
        if folder_mode == "Existing folder" and existing:
            folder = st.selectbox("Folder", existing)
        else:
            folder = st.text_input("Folder / Strategy name", placeholder="e.g. Pattern Scalping")
        file_name = st.text_input("File name", placeholder="e.g. NSE morning")
        if st.button("Next", type="primary"):
            if not folder or not folder.strip():
                st.error("Folder name enter pannunga.")
            elif not file_name or not file_name.strip():
                st.error("File name enter pannunga.")
            else:
                path = strategy_dir(uid, folder, file_name)
                if os.path.exists(os.path.join(path, "config.json")):
                    st.error("This folder/file already exists. Choose another name.")
                else:
                    st.session_state["create_folder"] = folder.strip()
                    st.session_state["create_file"] = file_name.strip()
                    st.session_state["create_step"] = 2
                    st.rerun()
        return

    # Step 2
    folder = st.session_state.get("create_folder", "")
    file_name = st.session_state.get("create_file", "")
    st.subheader(f"Step 2 · Settings  ·  {folder} / {file_name}")
    if st.button("← Back to Step 1"):
        st.session_state["create_step"] = 1
        st.rerun()

    smart = st.session_state.smart
    try:
        master = load_master()
    except Exception as ex:
        master = None
        st.warning(f"Stock list load aagala ({ex}).")

    exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "MCX"])
    items = []
    if master is not None:
        inst = exchange_list_cached(exchange)
        labels = inst["label"].tolist()
        by_label = inst.set_index("label")
        pick_all = st.checkbox(f"Select all {len(labels)}", key=f"create_all_{exchange}")
        picked = labels if pick_all else st.multiselect(
            f"{exchange} stocks ({len(labels)})", labels, key=f"create_stocks_{exchange}",
        )
        typed = st.text_input("Or type names (comma separated)", key="create_typed")
        names, miss = list(picked), []
        for t in [x.strip().upper() for x in typed.split(",") if x.strip()]:
            hit = inst[
                (inst["label"].str.upper() == t)
                | (inst["symbol"].str.upper() == t)
                | (inst["name"].str.upper() == t)
            ]
            if hit.empty:
                miss.append(t)
            elif hit.iloc[0]["label"] not in names:
                names.append(hit.iloc[0]["label"])
        if miss:
            st.warning("Not found in " + exchange + ": " + ", ".join(miss))
        items = [to_item({**by_label.loc[n].to_dict(), "label": n}, exchange) for n in names]
    else:
        typed = st.text_input("Type names (comma separated)", key="create_typed2")
        miss = []
        for t in [x.strip().upper() for x in typed.split(",") if x.strip()]:
            it = search_item(smart, exchange, t)
            if it:
                items.append(it)
            else:
                miss.append(t)
        if miss:
            st.warning("Not found: " + ", ".join(miss))

    if len(items) > 50:
        st.caption("Angel API limit — first 50 stocks only.")

    capital = st.number_input("Capital", min_value=10000.0, value=None, step=10000.0, placeholder="Capital")
    tf_label = st.selectbox("Candle timeframe", list(TIMEFRAMES), index=None, placeholder="Select timeframe")
    NO_LIMIT = "Select max (no limit)"
    max_pick = st.selectbox("Max open positions", [NO_LIMIT] + list(range(1, 26)) + [30, 40, 50], index=0)
    no_limit = max_pick == NO_LIMIT
    max_open = 10 ** 6 if no_limit else int(max_pick)
    use_lots, lots_per_pos = False, 0
    if exchange in ("MCX", "NFO"):
        use_lots = st.radio("Position size", ["Lots", "Amount"], horizontal=True, key="create_size_mode") == "Lots"
    if use_lots:
        lots_val = st.number_input("Lots per position", min_value=1, value=None, step=1, placeholder="Lots", key="create_lots")
        lots_per_pos = int(lots_val) if lots_val else 0
        info = ", ".join(f"{it['symbol']}: 1 lot = {it['lot']} units" for it in items[:6])
        if info:
            st.caption(info + (" ..." if len(items) > 6 else ""))
        position_amount = 0.0
    elif no_limit:
        position_amount = st.number_input(
            "Amount per position", min_value=1.0, value=None, step=1000.0, placeholder="Amount",
        )
    else:
        position_amount = st.number_input(
            "Amount per position",
            min_value=1.0,
            value=float(round(capital / max_open, 2)) if capital else None,
            step=1000.0, placeholder="Amount",
        )
    target_pct = st.number_input("Target (%)", min_value=0.1, max_value=20.0, value=None, step=0.1, placeholder="Target %")

    if st.button("Save", type="primary"):
        missing = [
            n for n, v in (
                ("Capital", capital), ("Candle timeframe", tf_label),
                (("Lots per position", lots_per_pos) if use_lots else ("Amount per position", position_amount)),
                ("Target %", target_pct),
            ) if not v
        ]
        if missing:
            st.error("Fill: " + ", ".join(missing))
        elif not items:
            st.error("Stocks select pannunga.")
        else:
            interval, tf_min = TIMEFRAMES[tf_label]
            data = {
                "exchange": exchange,
                "items": items,
                "capital": float(capital),
                "tf_label": tf_label,
                "interval": interval,
                "tf_min": tf_min,
                "max_open": int(max_open),
                "no_limit": no_limit,
                "position_amount": float(position_amount or 0),
                "lots_per_position": lots_per_pos if use_lots else 0,
                "target_pct": float(target_pct),
            }
            path = save_strategy_config(uid, folder, file_name, data)
            sf, sn = _safe_name(folder), _safe_name(file_name)
            st.session_state["active_strategy"] = {
                "folder": sf, "file": sn, "path": path,
            }
            for k in ("create_step", "create_folder", "create_file"):
                st.session_state.pop(k, None)
            _set_nav("file", sf, sn)
            st.success(f"Saved at `{path}`")
            st.rerun()


def file_dashboard():
    """One strategy file: Start/Stop + live tables. Edit settings only when STOPPED."""
    uid = current_uid()
    active = st.session_state.get("active_strategy")
    # restore from URL if session lost (refresh / other tab)
    if not active:
        qf = str(st.query_params.get("folder", "")).strip()
        qn = str(st.query_params.get("file", "")).strip()
        if qf and qn:
            path = strategy_dir(uid, qf, qn)
            if os.path.exists(os.path.join(path, "config.json")):
                active = {"folder": _safe_name(qf), "file": _safe_name(qn), "path": path}
                st.session_state["active_strategy"] = active
    if not active:
        _set_nav("home")
        st.rerun()
        return

    path = active["path"]
    folder, file_name = active["folder"], active["file"]
    cfg_data = load_strategy_config(path)
    if not cfg_data:
        st.error(f"Config not found at `{path}`.")
        if st.button("Back"):
            _set_nav("home")
            st.rerun()
        return

    engine = get_strategy_engine(uid, path)
    if st.session_state.get("smart") and engine.smart is None:
        engine.set_login(
            st.session_state.smart,
            st.session_state.get("creds"),
            st.session_state.get("client_id", ""),
        )

    no_dim()
    if st.button("← Back to Strategies"):
        st.session_state.pop("active_strategy", None)
        _set_nav("home")
        st.rerun()

    st.title(f"{folder} / {file_name}")

    # Start / Stop
    c1, c2, c3 = st.columns([1, 1, 3])
    start_clicked = c1.button("Start", type="primary", use_container_width=True, disabled=engine.running)
    if c2.button("Stop", use_container_width=True, disabled=not engine.running):
        engine.stop()
        st.rerun()
    c3.write(f"Status: **{'RUNNING' if engine.running else 'STOPPED'}**")

    # Summary bar
    exchange = cfg_data.get("exchange", "NSE")
    tf_label = cfg_data.get("tf_label", "-")
    capital = cfg_data.get("capital", 0)
    max_open = cfg_data.get("max_open", 5)
    no_limit = cfg_data.get("no_limit", False)
    target_pct = cfg_data.get("target_pct", 1.0)
    items = cfg_data.get("items", [])
    position_amount = cfg_data.get("position_amount", 0)
    lots_cfg = int(cfg_data.get("lots_per_position", 0) or 0)
    size_txt = f"{lots_cfg} lot(s)/pos" if lots_cfg else f"amount/pos {position_amount:,.0f}"
    st.info(
        f"**{exchange}**  |  {tf_label}  |  capital {capital:,.0f}  |  "
        f"max positions {'no limit' if no_limit else max_open}  |  "
        f"{size_txt}  |  target {target_pct}%  |  stocks {len(items)}"
    )

    if start_clicked:
        if not items:
            st.error("No stocks in this file. Edit settings and add stocks.")
        elif engine.smart is None and st.session_state.get("smart") is None:
            st.error("Angel One login illa.")
        else:
            interval = cfg_data.get("interval", "FIVE_MINUTE")
            tf_min = cfg_data.get("tf_min", 5)
            cfg = Config(
                capital=float(capital),
                max_open=int(max_open),
                target_pct=float(target_pct),
                interval=interval,
                tf_min=int(tf_min),
                position_amount=float(position_amount or 0),
                no_limit=bool(no_limit),
                lots_per_position=lots_cfg,
            )
            smart = engine.smart or st.session_state.smart
            engine.start(smart, cfg, items, exchange, False)
            st.rerun()

    # Edit settings (only when stopped)
    if not engine.running:
        with st.expander("Edit settings (Stop-la irukkumbodhu mattum)"):
            smart = st.session_state.smart
            try:
                master = load_master()
            except Exception:
                master = None
            new_ex = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "MCX"], index=["NSE", "BSE", "NFO", "MCX"].index(exchange) if exchange in ("NSE", "BSE", "NFO", "MCX") else 0)
            new_items = []
            if master is not None:
                inst = exchange_list_cached(new_ex)
                labels = inst["label"].tolist()
                by_label = inst.set_index("label")
                current_syms = [it["symbol"] for it in items if it.get("exchange") == new_ex]
                default_pick = [x for x in current_syms if x in labels]
                pick_all = st.checkbox(f"Select all {len(labels)}", key=f"edit_all_{new_ex}")
                picked = labels if pick_all else st.multiselect(
                    f"{new_ex} stocks", labels, default=default_pick, key=f"edit_stocks_{new_ex}",
                )
                new_items = [to_item({**by_label.loc[n].to_dict(), "label": n}, new_ex) for n in picked]
            new_capital = st.number_input("Capital", min_value=10000.0, value=float(capital), step=10000.0)
            tf_keys = list(TIMEFRAMES)
            tf_idx = tf_keys.index(tf_label) if tf_label in tf_keys else 2
            new_tf = st.selectbox("Candle timeframe", tf_keys, index=tf_idx)
            NO_LIMIT = "Select max (no limit)"
            max_opts = [NO_LIMIT] + list(range(1, 26)) + [30, 40, 50]
            cur_max = NO_LIMIT if no_limit else (max_open if max_open in max_opts else 5)
            max_pick = st.selectbox("Max open positions", max_opts, index=max_opts.index(cur_max) if cur_max in max_opts else 0)
            new_no_limit = max_pick == NO_LIMIT
            new_max = 10 ** 6 if new_no_limit else int(max_pick)
            cur_lots = int(cfg_data.get("lots_per_position", 0) or 0)
            new_use_lots, new_lots = False, 0
            if new_ex in ("MCX", "NFO"):
                new_use_lots = st.radio(
                    "Position size", ["Lots", "Amount"], index=0 if cur_lots > 0 else 1,
                    horizontal=True, key="edit_size_mode",
                ) == "Lots"
            if new_use_lots:
                new_lots = int(st.number_input("Lots per position", min_value=1, value=max(cur_lots, 1), step=1, key="edit_lots"))
                new_amt = 0.0
            else:
                new_amt = st.number_input(
                    "Amount per position", min_value=1.0,
                    value=float(position_amount) if position_amount else float(round(new_capital / max(new_max, 1), 2)),
                    step=1000.0,
                )
            new_tgt = st.number_input("Target (%)", min_value=0.1, max_value=20.0, value=float(target_pct), step=0.1)
            if st.button("Save settings"):
                interval, tf_min = TIMEFRAMES[new_tf]
                data = {
                    "exchange": new_ex,
                    "items": new_items or items,
                    "capital": float(new_capital),
                    "tf_label": new_tf,
                    "interval": interval,
                    "tf_min": tf_min,
                    "max_open": int(new_max),
                    "no_limit": new_no_limit,
                    "position_amount": float(new_amt),
                    "lots_per_position": new_lots if new_use_lots else 0,
                    "target_pct": float(new_tgt),
                }
                save_strategy_config(uid, folder, file_name, data)
                st.success("Settings saved.")
                st.rerun()

    # Manage positions
    open_now = engine.snapshot()["state"]["open"]
    if open_now:
        with st.expander("Manage open positions"):
            keys = [f"{p['exchange']}:{p['symbol']}" for p in open_now]
            sel = st.selectbox("Open position", keys)
            m1, m2 = st.columns(2)
            if m1.button("Close now", use_container_width=True):
                engine.close_now(sel)
                st.rerun()
            if m2.button("Delete (no record)", use_container_width=True):
                engine.delete_position(sel)
                st.rerun()

    if st.button("Reset all trades", disabled=not st.checkbox("Delete ALL open + closed trades for this file")):
        engine.reset_all()
        st.rerun()

    # Live fragments
    frag = getattr(st, "fragment", None) or st.experimental_fragment
    fast = 1 if engine.running else None
    slow = 3 if engine.running else None

    @frag(run_every=fast)
    def _top():
        live_top(engine, exchange, tf_label)

    @frag(run_every=slow)
    def _mid():
        live_mid(engine)

    @frag(run_every=fast)
    def _pos():
        live_positions(engine)

    @frag(run_every=slow)
    def _bot():
        live_bottom(engine)

    _top()
    _mid()
    _pos()
    _bot()


def main_router():
    """After login: home / create / file. Restore page from URL if session empty."""
    page = st.session_state.get("page") or str(st.query_params.get("p", "home"))
    if page == "create":
        create_strategy_page()
    elif page == "file":
        file_dashboard()
    else:
        strategies_home()


# ============================== ENTRY ==============================
st.set_page_config(page_title="Live Paper Trading", layout="wide")

if not access_gate():
    st.stop()

uid = current_uid()
_ss = st.session_state

# Keep email in URL so refresh / laptop wake does not ask access again
_em = (_ss.get("access_email") or str(st.query_params.get("e", ""))).strip().lower()
if _em:
    _ss["access_email"] = _em
    _ss["access_granted"] = True
    if st.query_params.get("e") != _em:
        st.query_params["e"] = _em

# Restore Angel session on refresh — do NOT ask API key again.
# 1) dedicated login cache  2) fallback: any engine still holding smart
# Cleared only by Logout button or full app/server restart.
if "smart" not in _ss and not _ss.get("no_autologin") and uid and not is_logged_out(uid):
    cached = load_login(uid)
    if cached and cached.get("smart") is not None:
        _ss.smart = cached["smart"]
        _ss.client_id = cached.get("client_id") or "connected"
        if cached.get("creds"):
            _ss.creds = cached["creds"]
        share_login_to_engines(uid, _ss.smart, _ss.get("creds"), _ss.client_id)
    else:
        reg = _engine_registry()
        with reg["lock"]:
            for k, eng in reg["map"].items():
                if k.startswith(f"{uid}::") and eng.smart is not None:
                    _ss.smart = eng.smart
                    _ss.client_id = eng.client_id or "connected"
                    if eng.creds:
                        _ss.creds = eng.creds
                    save_login(uid, eng.smart, eng.creds, eng.client_id)
                    break

# Restore active strategy + page from URL (refresh keeps you on same file page)
if "active_strategy" not in _ss or not _ss.get("page"):
    _p = str(st.query_params.get("p", "") or "")
    _f = str(st.query_params.get("folder", "") or "")
    _n = str(st.query_params.get("file", "") or "")
    if _p:
        _ss["page"] = _p
    if _p == "file" and _f and _n and uid:
        _path = strategy_dir(uid, _f, _n)
        if os.path.exists(os.path.join(_path, "config.json")):
            _ss["active_strategy"] = {"folder": _f, "file": _n, "path": _path}

# auto-login for owner only (server-saved ANGEL_* secrets)
if "smart" not in _ss and not _ss.get("no_autologin") and uid == OWNER_ID:
    _creds = env_creds()
    if _creds:
        with st.spinner("Angel One login..."):
            _sm, _err = angel_login(*_creds)
        if _sm is not None:
            cid = _creds[1].strip().upper()
            save_login(uid, _sm, _creds, cid)
            share_login_to_engines(uid, _sm, _creds, cid)
            _ss.smart = _sm
            _ss.client_id = cid
            _ss.creds = _creds
            autoresume_all(uid, _sm, _creds, cid)
        else:
            st.warning(f"Auto login failed: {_err}")

if "smart" not in st.session_state:
    login_page()
else:
    # Push login onto any new engines created after login
    if uid and _ss.get("smart"):
        share_login_to_engines(uid, _ss.smart, _ss.get("creds"), _ss.get("client_id", ""))
    main_router()

# scripts/demo_data.py
# 作用：从 Binance 现货 REST /api/v3/klines 下载 1m 历史K线
#      1) 可落盘到 ./data/binance/*.csv
#      2) 可一键导入 vn.py 数据库（等价于 DataManager.import_data_from_csv）
#
# 用法示例：
#   下载到 CSV（断点续传）：
#     python demo_data.py --symbol BTCUSDT --start 2020-01-01 --end 2025-08-25 --to-db
#
# 备注：
# - 数据库导入逻辑对齐 vnpy_datamanager/engine.py::import_data_from_csv
# - CSV 列：symbol,exchange,datetime,open,high,low,close,turnover,volume,open_interest
# - datetime 为 UTC 文本 "YYYY-MM-DD HH:MM:SS"

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.utility import ZoneInfo
from vnpy.trader.object import BarData
from vnpy.trader.database import get_database
from vnpy.trader.setting import SETTINGS

BINANCE_SPOT_BASE = "https://api.binance.com"
DATA_DIR = os.path.join(".", "data", "binance")


# ----------------------------- Utils -----------------------------

def parse_dt(s: Optional[str], tz=ZoneInfo("UTC")) -> Optional[datetime]:
    if not s:
        return None
    if "T" in s:
        dt = datetime.fromisoformat(s)
    else:
        dt = datetime.strptime(s, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt


def to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return int(dt.timestamp() * 1000)


def from_ms(ms: int) -> datetime:
    """Binance 时间是 UTC 毫秒，这里转回 tz-aware UTC"""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def csv_path(market: str, symbol: str, interval: str) -> str:
    fname = f"{market.lower()}_{symbol.upper()}_{interval}.csv"
    return os.path.join(DATA_DIR, fname)


def read_last_dt_from_csv(path: str) -> Optional[datetime]:
    """
    读取 CSV 最后一行的 datetime（UTC）用于断点续传。
    若文件不存在/为空，返回 None。
    """
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            last = row
    if not last:
        return None
    # CSV 中 datetime 用 "YYYY-MM-DD HH:MM:SS"（UTC）
    dt = datetime.strptime(last["datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return dt


def resolve_exchange(name: str) -> Exchange:
    """
    将字符串解析为 Exchange 枚举；若不识别，回退 LOCAL。
    """
    name = (name or "").strip().upper()
    for ex in Exchange:
        if ex.value == name:
            return ex
    try:
        return Exchange[name]  # 允许传入 "BINANCE" 这种枚举名
    except Exception:
        return Exchange.LOCAL


def interval_str_to_enum(interval: str) -> Interval:
    """
    仅支持 1m（演示脚本），如需拓展在此处加映射。
    """
    if interval != "1m":
        raise ValueError("This demo only supports 1m interval for now.")
    return Interval.MINUTE


# ----------------------------- Binance REST -----------------------------
def fetch_klines_spot(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    session: Optional[requests.Session] = None,
    base_url: str = BINANCE_SPOT_BASE,
    limit: int = 1000,
    sleep_ms: int = 200,
) -> List[list]:
    """
    分批拉取 Binance 现货 /api/v3/klines
    返回原始 kline 列表（每行是长度为12的数组）
    """
    s = session or requests.Session()
    start_ms = to_ms(start.astimezone(timezone.utc))
    end_ms = to_ms(end.astimezone(timezone.utc))

    interval_ms_map = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
        "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
        "1w": 604_800_000, "1M": 2_592_000_000
    }
    if interval not in interval_ms_map:
        raise ValueError(f"Unsupported interval: {interval}")
    step = interval_ms_map[interval] * limit

    out: List[list] = []
    cursor = start_ms
    while cursor <= end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": cursor,
            "endTime": min(cursor + step - 1, end_ms),
            "limit": limit,
        }
        url = f"{base_url}/api/v3/klines"
        resp = s.get(url, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            cursor = params["endTime"] + 1
            continue

        out.extend(batch)

        # 推进：最后一根 openTime + 一个 interval
        last_open = batch[-1][0]
        cursor = last_open + interval_ms_map[interval]

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000)

    return out


# ----------------------------- CSV 写入 -----------------------------
FIELDNAMES = [
    "symbol",
    "exchange",
    "datetime",     # UTC: YYYY-MM-DD HH:MM:SS
    "open",
    "high",
    "low",
    "close",
    "turnover",     # 这里用 quoteAssetVolume
    "volume",       # base asset volume
    "open_interest" # 现货为 0
]

def kline_to_row(
    k: list,
    symbol: str,
    exchange_str: str
) -> dict:
    """
    kline -> CSV 行字典（UTC）
    kline: [0]openTime, [1]open, [2]high, [3]low, [4]close,
           [5]volume(base), [6]closeTime, [7]quoteVolume, ...
    """
    dt = from_ms(k[0])  # UTC
    return {
        "symbol": symbol.upper(),
        "exchange": exchange_str,
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open": float(k[1]),
        "high": float(k[2]),
        "low": float(k[3]),
        "close": float(k[4]),
        "turnover": float(k[7]),
        "volume": float(k[5]),
        "open_interest": 0.0,
    }


def append_klines_to_csv(
    path: str,
    klines: List[list],
    symbol: str,
    exchange_enum: Exchange,
    dedup_after_dt: Optional[datetime] = None
) -> int:
    """
    将 klines 追加写入 CSV。
    - 若提供 dedup_after_dt，则只写 openTime > dedup_after_dt 的数据（UTC 比较）
    - 自动写 header（若新文件）
    返回成功写入的行数
    """
    ensure_dir(os.path.dirname(path))
    mode = "a" if os.path.exists(path) else "w"
    written = 0
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        if mode == "w":
            w.writeheader()
        for k in klines:
            if dedup_after_dt is not None:
                open_dt_utc = from_ms(k[0])  # UTC
                if open_dt_utc <= dedup_after_dt:
                    continue
            row = kline_to_row(k, symbol, exchange_enum.value)
            w.writerow(row)
            written += 1
    return written


# ----------------------------- 导入数据库（参照 vnpy_datamanager/engine.py） -----------------------------
def import_csv_to_db(
    file_path: str,
    symbol: str,
    exchange_enum: Exchange,
    interval_enum: Interval,
    tz_name: str = "UTC",
    # 列名（与本脚本写出的 CSV 完全一致）
    datetime_head: str = "datetime",
    open_head: str = "open",
    high_head: str = "high",
    low_head: str = "low",
    close_head: str = "close",
    volume_head: str = "volume",
    turnover_head: str = "turnover",
    open_interest_head: str = "open_interest",
    datetime_format: str = "%Y-%m-%d %H:%M:%S",
) -> tuple[datetime, datetime, int]:
    """
    读取 CSV → 构造 BarData → get_database().save_bar_data(bars)
    返回：(start_dt, end_dt, count)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        buf = [line.replace("\0", "") for line in f]

    reader = csv.DictReader(buf, delimiter=",")
    bars: list[BarData] = []
    start_dt: Optional[datetime] = None
    last_bar: Optional[BarData] = None
    tz = ZoneInfo(tz_name)

    for item in reader:
        if datetime_format:
            dt = datetime.strptime(item[datetime_head], datetime_format)
        else:
            dt = datetime.fromisoformat(item[datetime_head])
        dt = dt.replace(tzinfo=tz)

        turnover = float(item.get(turnover_head, 0) or 0)
        open_interest = float(item.get(open_interest_head, 0) or 0)

        bar = BarData(
            symbol=symbol,
            exchange=exchange_enum,
            datetime=dt,
            interval=interval_enum,
            volume=float(item[volume_head]),
            open_price=float(item[open_head]),
            high_price=float(item[high_head]),
            low_price=float(item[low_head]),
            close_price=float(item[close_head]),
            turnover=turnover,
            open_interest=open_interest,
            gateway_name="DB",
        )
        bars.append(bar)
        if start_dt is None:
            start_dt = dt
        last_bar = bar

    if not bars:
        raise RuntimeError(f"No rows in CSV: {file_path}")

    db = get_database()
    db.save_bar_data(bars)

    return start_dt, last_bar.datetime, len(bars)


# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Download Binance spot klines (1m) to CSV and optionally import to vn.py database."
    )
    p.add_argument("--symbol", required=True, help="交易对，如 BTCUSDT")
    p.add_argument("--start", required=True, help="开始日期，如 2020-01-01 或 2020-01-01T00:00:00")
    p.add_argument("--end", default=None, help="结束日期（默认=当前UTC）")
    p.add_argument("--interval", default="1m", help="K线周期，默认 1m（本脚本仅示范 1m）")
    p.add_argument("--exchange", default="BINANCE", help="交易所字段（CSV与DB中均使用），默认 BINANCE；不识别则回退 LOCAL")
    p.add_argument("--market", default="spot", choices=["spot"], help="市场类型（先支持 spot）")
    p.add_argument("--sleep-ms", type=int, default=200, help="REST 每批后 sleep 毫秒，默认 200")
    p.add_argument("--overwrite", action="store_true", help="若已有 CSV，覆盖重写；否则断点续传（append）")
    p.add_argument("--base-url", default=BINANCE_SPOT_BASE, help="Binance REST 基地址（现货）")
    # 导库相关
    p.add_argument("--to-db", action="store_true", help="下载/写入 CSV 后，将同一份 CSV 导入 vn.py 数据库")
    p.add_argument("--tz", default="UTC", help="CSV datetime 的时区名（导库时使用），默认 UTC")
    return p.parse_args()


def main():
    args = parse_args()

    symbol = args.symbol.upper()
    interval = args.interval
    if interval != "1m":
        print(f"[WARN] 当前 DEMO 仅演示 1m，收到 {interval}，将继续以 1m 拉取。")
        interval = "1m"

    start = parse_dt(args.start, tz=ZoneInfo("UTC"))
    end = parse_dt(args.end, tz=ZoneInfo("UTC")) or datetime.now(tz=ZoneInfo("UTC"))

    if start >= end:
        raise SystemExit("start must be earlier than end")

    exchange_enum = resolve_exchange(args.exchange)
    interval_enum = interval_str_to_enum(interval)

    out_path = csv_path(args.market, symbol, interval)

    # 断点续传处理
    last_dt = None
    if os.path.exists(out_path) and not args.overwrite:
        last_dt = read_last_dt_from_csv(out_path)
        if last_dt:
            # 从上次最后一根之后继续（注意：写入时会跳过 <= last_dt）
            start = max(start, last_dt)
            print(f"[INFO] Resume from {start.isoformat()} (last CSV dt: {last_dt.isoformat()})")
        else:
            print("[INFO] Existing CSV has no rows, will write header and start fresh.")
    elif args.overwrite and os.path.exists(out_path):
        os.remove(out_path)
        print(f"[INFO] Overwrite enabled, removed existing file: {out_path}")

    print(f"[INFO] Fetching {symbol} 1m from {start.isoformat()} to {end.isoformat()} ...")

    klines = fetch_klines_spot(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        base_url=args.base_url,
        sleep_ms=args.sleep_ms,
    )
    if not klines:
        print("[INFO] No data returned.")
        return

    # 写 CSV（若断点续传，则只写 > last_dt 的数据）
    written = append_klines_to_csv(
        path=out_path,
        klines=klines,
        symbol=symbol,
        exchange_enum=exchange_enum,
        dedup_after_dt=last_dt
    )
    print(f"[OK] Wrote {written} rows to {out_path}")

    # 可选：导入数据库
    if args.to_db:
        print(f"[INFO] Importing to database as {symbol}.{exchange_enum.value}, interval={interval_enum.name}, tz={args.tz} ...")
        start_dt, end_dt, cnt = import_csv_to_db(
            file_path=out_path,
            symbol=symbol,
            exchange_enum=exchange_enum,
            interval_enum=interval_enum,
            tz_name=args.tz,
        )
        print(f"[OK] Imported {cnt} bars to DB, range: {start_dt.isoformat()} ~ {end_dt.isoformat()}")
        print(f"[HINT] 回测请使用 vt_symbol='{symbol}.{exchange_enum.value}', interval=Interval.MINUTE")

if __name__ == "__main__":
    main()

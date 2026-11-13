"""Single-instrument inventory mean-reversion + 1D isothermal Navier-Stokes analogue.

Educational PAPER BACKTEST only: never places orders or uses API keys.

Math (discrete-time signals):
    X_t = log(mid_t) - EMA(log(mid))
    E[delta log(S)] = - (1 - exp(-kappa * horizon_steps)) * X_t
    u = expected return in bps + beta * predicted near-book imbalance
    w* = clip(u / (gamma * realized_variance_bps2), -1, 1)
    target USD = max_notional * w*  [after entry, exit, cost and inventory gates]

OPTIONAL reaction-diffusion benchmark on distance-from-best grid x (bps):
    d rho_side / d tau = D * d2 rho_side / dx2 - c * (rho_side - rho_reference_side)
    side in {bid, ask}; explicit finite differences, reflecting boundaries,
    nonnegative density, and a checked CFL condition.

1D barotropic / isothermal COMPRESSIBLE NAVIER-STOKES analogue (each book side):
    d rho / dt + d (rho*u) / dx = lambda*(rho_ref-rho)
    d (rho*u)/dt + d (rho*u*u + c*c*rho)/dx
        = d[nu*rho*(du/dx)]/dx - drag*(rho*u)
    p = c*c*rho. Uses a Rusanov numerical flux and adaptive CFL steps.
    rho = L2 depth per bps, x = distance from the best bid/ask in bps.
    u = proxy inferred from changes in L2 snapshots, NOT observed true velocity.
    Order placement/cancellation makes book volume NON-CONSERVED, hence source.
    This is a stylized FINANCIAL ANALOGUE, not established market physics.
    Requires snapshots and out-of-sample validation. Not a proof of arbitrage.

Modes:
    python inventory_mr_navier_stokes.py --demo
    python inventory_mr_navier_stokes.py --fetch --hours 72
    python inventory_mr_navier_stokes.py --csv my_ohlcv.csv  # OU baseline only
    python inventory_mr_navier_stokes.py --collect --seconds 120 --interval 5
    python inventory_mr_navier_stokes.py --snapshots snapshots.jsonl
    python inventory_mr_navier_stokes.py --self-test

Dependencies: numpy pandas, optional matplotlib (plots), optional ccxt (API).
A CSV of OHLCV cannot reconstruct historical L2: both depth models OFF for --fetch/--csv.
Historical snapshots collected by --collect can be replayed using --snapshots.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Config:
    instrument: str = "BTC/USDC:USDC"
    max_notional: float = 1000.0  # USD not BTC; no leverage in simulation
    starting_cash: float = 10000.0
    fair_span: int = 80  # observations: e.g. minutes in --fetch mode
    risk_window: int = 40
    kappa_per_step: float = 0.35  # hypothetical mean reversion speed per observation
    horizon_steps: int = 2
    z_entry: float = 1.35
    z_exit: float = 0.40
    min_alpha_bps: float = 2.0
    risk_aversion: float = 0.22
    min_trade_usd: float = 125.0
    fee_bps_per_side: float = 2.0
    slippage_bps_per_side: float = 1.0
    fallback_spread_bps: float = 2.0  # OHLCV-only assumption, NOT observed
    max_gap_factor: float = 3.0
    fluid_beta_bps: float = 3.0  # estimated bps per normalized depth imbalance
    pde_D: float = 1.5  # bps^2 / second
    pde_cancel_rate: float = 0.012  # 1/second, pulling toward historical depth
    pde_horizon_sec: float = 12.0
    pde_step_sec: float = 0.5
    book_bins: int = 8
    book_max_bps: float = 32.0
    depth_ewm_alpha: float = 0.04
    min_depth_ratio: float = 0.35
    demo_step_sec: float = 5.0
    ns_sound_speed: float = 0.6       # analogue units: bps / s
    ns_kinematic_viscosity: float = 0.65  # bps^2 / s
    ns_drag_rate: float = 0.02        # 1 / s
    ns_source_rate: float = 0.01      # 1 / s
    ns_horizon_sec: float = 5.0       # should match sampling/holding horizon
    ns_nominal_step_sec: float = 0.5  # upper bound; adaptive CFL
    ns_velocity_gain: float = 1.0     # fit ONLY on training samples
    ns_max_velocity: float = 1.0      # bps/s, numerical regularizer
    ns_cfl_safety: float = 0.25
    ns_density_floor: float = 1e-8


def _timestamp_series(v: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(v):
        num = pd.to_numeric(v, errors="coerce")
        unit = "ms" if num.dropna().median() > 1e11 else "s"
        ts = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(v, utc=True, errors="coerce")
    return pd.DatetimeIndex(ts)


def from_ohlcv(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """End-of-bar close generates signal, NEXT bar open executes (causal)."""
    d = df.copy()
    if "time" in d.columns and "timestamp" not in d.columns:
        d = d.rename(columns={"time": "timestamp"})
    if "timestamp" not in d:
        raise ValueError("CSV needs timestamp or time column (ISO UTC / Unix ms)")
    d.index = _timestamp_series(d["timestamp"])
    d.index.name = "time"
    for col in ["open", "close"]:
        if col not in d:
            raise ValueError(f"Missing {col}")
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["open", "close"])
    d = d[(d.open > 0) & (d.close > 0)]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    if len(d) < max(config.fair_span, config.risk_window) + 20:
        raise ValueError("Need more closed bars for rolling indicators")
    # Last still-open candle must be filtered at fetch time by the caller.
    # Each row's signal_mid is known only at that bar CLOSE; trading that
    # signal at the NEXT row's open avoids same-bar lookahead.
    result = pd.DataFrame(index=d.index)
    result["signal_mid"] = d["close"].astype(float)
    result["mark_exec"] = d["open"].astype(float)
    result["buy_exec"] = result["mark_exec"] * (1 + config.fallback_spread_bps / 20000)
    result["sell_exec"] = result["mark_exec"] * (1 - config.fallback_spread_bps / 20000)
    result.attrs["source"] = "ohlcv_proxy"
    return result


def fetch_ohlcv(hours: int, cfg: Config) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError as exc:
        raise RuntimeError("Install ccxt: python -m pip install ccxt") from exc
    exchange = ccxt.hyperliquid({"enableRateLimit": True, "timeout": 25000})
    exchange.load_markets()
    if cfg.instrument not in exchange.markets:
        raise ValueError(f"Symbol unavailable: {cfg.instrument}; check ccxt symbols")
    end_ms = int(time.time() * 1000) // 60000 * 60000
    start_ms = end_ms - hours * 3600000
    cursor = start_ms
    data: list[list[float]] = []
    while cursor < end_ms:
        batch = exchange.fetch_ohlcv(cfg.instrument, timeframe="1m", since=cursor, limit=500)
        batch = [b for b in batch if start_ms <= int(b[0]) < end_ms] if batch else []
        if not batch:
            break
        data.extend(batch)
        next_cursor = int(max(b[0] for b in batch)) + 60000
        if next_cursor <= cursor:
            raise RuntimeError("Exchange pagination stalled")
        cursor = next_cursor
    if not data:
        raise RuntimeError("Exchange returned no closed historical candles")
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.drop_duplicates("timestamp").sort_values("timestamp")


def collect_snapshots(path: Path, seconds: float, interval: float, cfg: Config) -> None:
    """Forward-only PUBLIC top-of-book capture; never place orders."""
    if seconds <= 0 or interval < 1:
        raise ValueError("seconds > 0, interval >= 1 required")
    try:
        import ccxt
    except ImportError as exc:
        raise RuntimeError("Install ccxt: python -m pip install ccxt") from exc
    ex = ccxt.hyperliquid({"enableRateLimit": True, "timeout": 25000})
    ex.load_markets()
    if cfg.instrument not in ex.markets:
        raise ValueError(f"Unknown market {cfg.instrument}")
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = time.monotonic() + seconds
    count = 0
    with path.open("a", encoding="utf-8") as f:
        while time.monotonic() < stop:
            start = time.monotonic()
            ob = ex.fetch_order_book(cfg.instrument, limit=20)
            if not ob.get("bids") or not ob.get("asks"):
                continue
            record = {
                "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
                "bids": [[float(p), float(q)] for p, q in ob["bids"][:20]],
                "asks": [[float(p), float(q)] for p, q in ob["asks"][:20]],
            }
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()
            count += 1
            time.sleep(max(0.0, interval - (time.monotonic() - start)))
    print(f"Captured {count} snapshots into {path}. No orders placed.")


def book_to_bins(levels: list[list[float]], mid: float, side: str, cfg: Config) -> np.ndarray:
    """Depth in BTC units per distance-from-mid price bucket."""
    out = np.zeros(cfg.book_bins, dtype=float)
    for px, quantity in levels:
        if px <= 0 or quantity <= 0:
            continue
        dist_bps = (mid - px) / mid * 10000 if side == "bid" else (px - mid) / mid * 10000
        if not 0 <= dist_bps < cfg.book_max_bps:
            continue
        idx = min(int(dist_bps / cfg.book_max_bps * cfg.book_bins), cfg.book_bins - 1)
        out[idx] += quantity
    return out


def from_snapshots(path: Path, cfg: Config) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            bids, asks = raw["bids"], raw["asks"]
            if not bids or not asks:
                continue
            bid, ask = float(bids[0][0]), float(asks[0][0])
            if not (0 < bid < ask):
                continue
            mid = (bid + ask) / 2
            rows.append({"timestamp": raw["timestamp"], "signal_mid": mid,
                         "mark_exec": mid, "buy_exec": ask, "sell_exec": bid,
                         "depth_bid": book_to_bins(bids, mid, "bid", cfg),
                         "depth_ask": book_to_bins(asks, mid, "ask", cfg)})
    if not rows:
        raise ValueError("No valid order books in JSONL")
    d = pd.DataFrame(rows)
    d.index = _timestamp_series(d.timestamp)
    d.index.name = "time"
    d = d.drop(columns="timestamp").sort_index()
    return d[~d.index.duplicated(keep="last")]


def simulate_data(n: int, seed: int, cfg: Config) -> pd.DataFrame:
    """OU-style synthetic mid and synthetic L2: NOT real exchange evidence."""
    rng = np.random.default_rng(seed)
    anchor = np.zeros(n)
    residual = np.zeros(n)
    for t in range(1, n):
        anchor[t] = anchor[t - 1] + rng.normal(0, 0.00006)
        residual[t] = 0.78 * residual[t - 1] + rng.normal(0, 0.00075)
    mid = 75000 * np.exp(anchor + residual)
    ts = pd.date_range("2026-01-01", periods=n, freq=f"{int(cfg.demo_step_sec)}s", tz="UTC", name="time")
    rows = []
    for i, (stamp, price) in enumerate(zip(ts, mid)):
        spread_bps = 1.3 + 1.2 * rng.random()
        bid, ask = price * (1 - spread_bps / 20000), price * (1 + spread_bps / 20000)
        bias = np.clip(-residual[i] * 200, -0.65, 0.65)
        levels_b, levels_a = [], []
        for k in range(1, cfg.book_bins * 2 + 1):
            distance_bps = k * cfg.book_max_bps / (cfg.book_bins * 2)
            bid_q = max(0.001, 0.06 * math.exp(-k / 11) * (1 + bias) * rng.lognormal(0, 0.23))
            ask_q = max(0.001, 0.06 * math.exp(-k / 11) * (1 - bias) * rng.lognormal(0, 0.23))
            levels_b.append([bid * (1 - distance_bps / 10000), bid_q])
            levels_a.append([ask * (1 + distance_bps / 10000), ask_q])
        # Actual best quotes MUST be in the level arrays.
        levels_b.insert(0, [bid, 0.1 * rng.lognormal(0, 0.2)])
        levels_a.insert(0, [ask, 0.1 * rng.lognormal(0, 0.2)])
        rows.append({"signal_mid": price, "mark_exec": price, "buy_exec": ask, "sell_exec": bid,
                     "depth_bid": book_to_bins(levels_b, price, "bid", cfg),
                     "depth_ask": book_to_bins(levels_a, price, "ask", cfg)})
    return pd.DataFrame(rows, index=ts)


def pde_forward(depth: np.ndarray, reference: np.ndarray, cfg: Config) -> np.ndarray:
    """Predict one future profile using an explicit Neumann-boundary PDE solver."""
    rho = np.asarray(depth, dtype=float).copy()
    ref = np.asarray(reference, dtype=float)
    dx = cfg.book_max_bps / cfg.book_bins
    dt = cfg.pde_step_sec
    if cfg.pde_D * dt / dx**2 > 0.5:
        raise ValueError("Unstable PDE step: D * dt / dx**2 must <= 0.5")
    if cfg.pde_cancel_rate * dt > 1:
        raise ValueError("Unstable reaction step: cancel_rate * dt must <= 1")
    if cfg.pde_horizon_sec <= 0 or cfg.pde_step_sec <= 0:
        raise ValueError("PDE horizon and step must be positive")
    steps = int(math.ceil(cfg.pde_horizon_sec / dt))
    for j in range(steps):
        actual_dt = min(dt, cfg.pde_horizon_sec - j * dt)
        # Zero spatial derivative at both edges (reflecting boundary).
        padded = np.pad(rho, (1, 1), mode="edge")
        lap = (padded[:-2] - 2 * rho + padded[2:]) / dx**2
        rho = rho + actual_dt * (cfg.pde_D * lap - cfg.pde_cancel_rate * (rho - ref))
        rho = np.maximum(0.0, rho)
    return rho



def ns_forward(depth: np.ndarray, ref_depth: np.ndarray,
               velocity: np.ndarray, cfg: Config) -> np.ndarray:
    """Finite-volume 1D compressible isothermal Navier-Stokes analogue.

    U=(rho,m=rho*u); F=(m,m*m/rho+c^2*rho).
    Rusanov interface flux, reflecting end walls, explicit viscous momentum
    flux and order-flow source term. x is distance from best (bps).
    Does NOT forecast actual order placements; parameters are hypotheses.
    """
    rho = np.asarray(depth, dtype=float).copy() / (cfg.book_max_bps / cfg.book_bins)
    ref = np.asarray(ref_depth, dtype=float).copy() / (cfg.book_max_bps / cfg.book_bins)
    vel = np.asarray(velocity, dtype=float).copy()
    n = cfg.book_bins
    if any(v.shape != (n,) for v in (rho, ref, vel)):
        raise ValueError("NS arrays must have book_bins cells")
    if not np.isfinite(rho).all() or not np.isfinite(ref).all() or np.any(rho < 0) or np.any(ref < 0):
        raise ValueError("NS requires finite nonnegative book density")
    if cfg.ns_horizon_sec <= 0 or cfg.ns_nominal_step_sec <= 0 or cfg.ns_sound_speed <= 0:
        raise ValueError("Invalid NS time/sound-speed setting")
    dx = cfg.book_max_bps / n
    floor = cfg.ns_density_floor
    rho = np.maximum(rho, floor)
    ref = np.maximum(ref, floor)
    m = rho * np.clip(vel, -cfg.ns_max_velocity, cfg.ns_max_velocity)
    c = cfg.ns_sound_speed
    nu = cfg.ns_kinematic_viscosity
    lam = cfg.ns_source_rate
    drag = cfg.ns_drag_rate
    t = 0.0
    while t < cfg.ns_horizon_sec - 1e-11:
        # Numerical regularization; prevents division by tiny density.
        u = np.clip(m / np.maximum(rho, floor), -cfg.ns_max_velocity, cfg.ns_max_velocity)
        m = rho * u
        max_speed = max(float(np.max(np.abs(u))) + c, c)
        dt_stability = cfg.ns_cfl_safety / (max_speed / dx + 2 * nu / dx**2 + lam + drag)
        dt = min(cfg.ns_nominal_step_sec, dt_stability, cfg.ns_horizon_sec - t)
        if dt <= 0 or not math.isfinite(dt):
            raise FloatingPointError("Invalid NS CFL step")
        # Reflective boundaries (no density flux at x=0 or x=max).
        rho_ext = np.pad(rho, (1, 1), mode="edge")
        m_ext = np.r_[-m[0], m, -m[-1]]
        left_rho, right_rho = rho_ext[:-1], rho_ext[1:]
        left_m, right_m = m_ext[:-1], m_ext[1:]
        u_l = left_m / np.maximum(left_rho, floor)
        u_r = right_m / np.maximum(right_rho, floor)
        speed = np.maximum(np.abs(u_l) + c, np.abs(u_r) + c)
        # Face flux F=(rho*u, rho*u^2 + pressure), pressure=c^2*rho.
        flux_rho = 0.5 * (left_m + right_m) - 0.5 * speed * (right_rho - left_rho)
        f_m_l = left_m * u_l + c**2 * left_rho
        f_m_r = right_m * u_r + c**2 * right_rho
        flux_m = 0.5 * (f_m_l + f_m_r) - 0.5 * speed * (right_m - left_m)
        # Enforce exactly zero density flux at both walls.
        flux_rho[0] = 0.0
        flux_rho[-1] = 0.0
        # Dynamic viscosity mu=nu*rho; no-slip gradient is set to zero at walls.
        viscous_flux = np.zeros(n + 1)
        viscous_flux[1:-1] = nu * 0.5 * (rho[:-1] + rho[1:]) * np.diff(u) / dx
        new_rho = rho - dt / dx * np.diff(flux_rho) + dt * lam * (ref - rho)
        new_m = (m - dt / dx * np.diff(flux_m) + dt / dx * np.diff(viscous_flux)
                 - dt * drag * m)
        if not np.isfinite(new_rho).all() or not np.isfinite(new_m).all():
            raise FloatingPointError("NS numerical blow-up")
        if np.min(new_rho) < -1e-8:
            raise FloatingPointError("NS negative density: reduce CFL")
        rho = np.maximum(new_rho, floor)
        m = new_m
        t += dt
    return rho * dx   # Convert density BTC/bps back into per-bin BTC.


def infer_book_velocity(current: np.ndarray, previous: np.ndarray | None,
                        elapsed_sec: float, cfg: Config) -> np.ndarray:
    """A PROXY for unobservable quote-volume flow, not measured order speed.

    If quote depth vanishes at x, assume outward velocity (toward larger x).
    Not identifiable from snapshots due to creates/cancels/trades; calibrate.
    """
    if previous is None or elapsed_sec <= 0 or not np.isfinite(elapsed_sec):
        return np.zeros(cfg.book_bins)
    a = np.asarray(current, float)
    b = np.asarray(previous, float)
    dx = cfg.book_max_bps / cfg.book_bins
    relative_change = (b - a) / np.maximum(0.5 * (a + b), 1e-7)
    u = cfg.ns_velocity_gain * relative_change * dx / elapsed_sec
    # Smooth cell-level snapshot micro-noise without seeing future snapshots.
    if cfg.book_bins >= 3:
        u = np.convolve(np.pad(u, (1, 1), mode="edge"),
                        [0.25, 0.50, 0.25], mode="valid")
    return np.clip(u, -cfg.ns_max_velocity, cfg.ns_max_velocity)

def make_features(df: pd.DataFrame, cfg: Config, model: str = "baseline") -> pd.DataFrame:
    """Causal feature calculation. model: baseline, reaction_diffusion, navier_stokes."""
    if model not in ("baseline", "reaction_diffusion", "navier_stokes"):
        raise ValueError("Unknown model")
    d = df.copy()
    if not d.index.is_monotonic_increasing or d.index.has_duplicates:
        raise ValueError("Unique sorted timestamps required")
    log_price = np.log(d.signal_mid.astype(float))
    fair = log_price.ewm(span=cfg.fair_span, adjust=False).mean()
    residual = log_price - fair
    res_std = residual.rolling(cfg.risk_window, min_periods=cfg.risk_window).std(ddof=0)
    ret_bps = log_price.diff() * 10000
    vol_bps = ret_bps.rolling(cfg.risk_window, min_periods=cfg.risk_window).std(ddof=0)
    d["z"] = residual / res_std.replace(0, np.nan)
    d["mr_alpha_bps"] = -10000 * (1 - np.exp(-cfg.kappa_per_step * cfg.horizon_steps)) * residual
    d["vol_bps"] = vol_bps
    d["gap_from_previous_sec"] = d.index.to_series().diff().dt.total_seconds().to_numpy()
    d["fluid_imbalance"] = 0.0
    d["fluid_depth_ratio"] = 1.0
    has_books = "depth_bid" in d.columns and "depth_ask" in d.columns
    if model != "baseline" and not has_books:
        raise ValueError("Cannot use liquidity model without historical L2 depth snapshots")
    if model != "baseline":
        ref_b = ref_a = prev_b = prev_a = None
        imb_vals, depth_ratios = [], []
        for bid_raw, ask_raw, delta_t in zip(d.depth_bid, d.depth_ask,
                                             d.gap_from_previous_sec):
            bid = np.asarray(bid_raw, dtype=float)
            ask = np.asarray(ask_raw, dtype=float)
            if bid.shape != (cfg.book_bins,) or ask.shape != (cfg.book_bins,):
                raise ValueError("Order book bin count mismatch")
            if ref_b is None:
                ref_b, ref_a = bid.copy(), ask.copy()
            if model == "reaction_diffusion":
                bid_f = pde_forward(bid, ref_b, cfg)
                ask_f = pde_forward(ask, ref_a, cfg)
            else:
                v_bid = infer_book_velocity(bid, prev_b, float(delta_t), cfg)
                v_ask = infer_book_velocity(ask, prev_a, float(delta_t), cfg)
                bid_f = ns_forward(bid, ref_b, v_bid, cfg)
                ask_f = ns_forward(ask, ref_a, v_ask, cfg)
            near = max(1, cfg.book_bins // 4)
            bnear, anear = float(bid_f[:near].sum()), float(ask_f[:near].sum())
            base_near = max(float(ref_b[:near].sum() + ref_a[:near].sum()), 1e-12)
            imbalance = (bnear - anear) / max(bnear + anear, 1e-12)
            depth_ratio = (bnear + anear) / base_near
            imb_vals.append(float(np.clip(imbalance, -1, 1)))
            depth_ratios.append(float(np.clip(depth_ratio, 0, 2)))
            # Model only sees the current/past: update reference AFTER prediction.
            ref_b = (1 - cfg.depth_ewm_alpha) * ref_b + cfg.depth_ewm_alpha * bid
            ref_a = (1 - cfg.depth_ewm_alpha) * ref_a + cfg.depth_ewm_alpha * ask
            prev_b, prev_a = bid, ask
        d["fluid_imbalance"] = imb_vals
        d["fluid_depth_ratio"] = depth_ratios
    d["alpha_bps"] = d["mr_alpha_bps"] + (cfg.fluid_beta_bps * d["fluid_imbalance"] if model != "baseline" else 0)
    d["pde_active"] = model != "baseline"  # Existing backtest liquidity scaling flag.
    d["liquidity_model"] = model
    return d

def backtest(d: pd.DataFrame, cfg: Config, *, strategy_label: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Causal target: bar/snapshot i-1 signal, i open/quote execution.

    Takable next-observation quotes + added slippage and fees. This is still
    optimistic about size, market impact and quote persistence; paper only.
    """
    cash = cfg.starting_cash
    qty = 0.0
    old_executed = 0
    trades, equity_rows = [], []
    min_history = max(cfg.fair_span, cfg.risk_window) + 3
    median_dt = d.index.to_series().diff().dt.total_seconds().median()
    if not np.isfinite(median_dt) or median_dt <= 0:
        raise ValueError("Need a positive time interval")
    normal_gap = median_dt * cfg.max_gap_factor
    turnover = 0.0
    # At observation i, execute decisions made at observation i-1.
    for i in range(len(d)):
        row = d.iloc[i]
        mark = float(row.mark_exec)
        if not np.isfinite(mark) or mark <= 0:
            raise ValueError("Invalid execution mark")
        wanted = qty
        reason = "hold"
        if i >= min_history:
            prior = d.iloc[i - 1]
            gap = (d.index[i] - d.index[i - 1]).total_seconds()
            valid_signal = np.isfinite([prior.z, prior.alpha_bps, prior.vol_bps]).all()
            if (gap > normal_gap or float(prior.gap_from_previous_sec) > normal_gap
                    or not valid_signal):
                wanted = 0.0  # move to flat on next available observation
                reason = "gap_or_invalid"
            else:
                z = float(prior.z)
                alpha = float(prior.alpha_bps)
                vol = max(float(prior.vol_bps), 0.5)
                # Compare expected alpha with rough round-trip observed spread+fees/slippage.
                prev_spread = (float(prior.buy_exec) - float(prior.sell_exec)) / float(prior.mark_exec) * 10000
                round_trip = prev_spread + 2 * (cfg.fee_bps_per_side + cfg.slippage_bps_per_side)
                significant = abs(alpha) > max(cfg.min_alpha_bps, round_trip)
                # Inventory optimizer: max over w in [-1, 1] of
                #      w * expected_bps - 0.5 * gamma * vol_bps**2 * w**2.
                fraction = np.clip(alpha / (cfg.risk_aversion * vol**2), -1, 1)
                depth_scale = 1.0
                if bool(prior.pde_active):
                    # Near-book projected depth constrains maximum inventory.
                    depth_scale = float(np.clip(prior.fluid_depth_ratio, cfg.min_depth_ratio, 1.0))
                proposed_usd = cfg.max_notional * float(fraction) * depth_scale
                if abs(z) <= cfg.z_exit or np.sign(alpha) == np.sign(z):
                    wanted = 0.0
                    reason = "exit_reversion"
                elif abs(z) >= cfg.z_entry and significant:
                    wanted = proposed_usd / mark
                    reason = "entry_or_rebalance"
                # Within hysteresis band: do not change inventory.
                if (reason not in ("gap_or_invalid", "exit_reversion")
                        and abs(wanted - qty) * mark < cfg.min_trade_usd):
                    wanted = qty
                    reason = "deadband"
        # Net exposure constraint, notional measured using executable mid.
        wanted = float(np.clip(wanted, -cfg.max_notional / mark, cfg.max_notional / mark))
        delta = wanted - qty
        if abs(delta * mark) > 1e-9:
            quoted_exec = float(row.buy_exec if delta > 0 else row.sell_exec)
            if quoted_exec <= 0:
                raise ValueError("Bad executable quote")
            trade_px = quoted_exec * (1 + (1 if delta > 0 else -1) * cfg.slippage_bps_per_side / 10000)
            notional = abs(delta) * trade_px
            fee = notional * cfg.fee_bps_per_side / 10000
            cash -= delta * trade_px + fee
            qty = wanted
            turnover += notional
            trades.append({"time": d.index[i], "strategy": strategy_label,
                           "reason": reason, "side": "BUY" if delta > 0 else "SELL",
                           "price": trade_px, "delta_qty": delta,
                           "inventory_qty": qty, "notional_usd": notional,
                           "fee_usd": fee, "alpha_bps_prev": (float(d.alpha_bps.iloc[i - 1]) if i else np.nan),
                           "z_prev": (float(d.z.iloc[i - 1]) if i else np.nan)})
        equity_rows.append({"time": d.index[i], "strategy": strategy_label, "equity": cash + qty * mark,
                            "cash": cash, "qty": qty, "notional_usd": qty * mark})
    # Explicitly close final remaining inventory at the LAST AVAILABLE next-observation quote.
    if abs(qty) > 1e-12:
        last = d.iloc[-1]
        delta = -qty
        quote = float(last.buy_exec if delta > 0 else last.sell_exec)
        px = quote * (1 + (1 if delta > 0 else -1) * cfg.slippage_bps_per_side / 10000)
        fee = abs(delta) * px * cfg.fee_bps_per_side / 10000
        cash -= delta * px + fee
        turnover += abs(delta) * px
        trades.append({"time": d.index[-1], "strategy": strategy_label, "reason": "end_flatten",
                       "side": "BUY" if delta > 0 else "SELL", "price": px,
                       "delta_qty": delta, "inventory_qty": 0.0,
                       "notional_usd": abs(delta) * px, "fee_usd": fee,
                       "alpha_bps_prev": np.nan, "z_prev": np.nan})
        qty = 0.0
        equity_rows[-1].update({"equity": cash, "cash": cash, "qty": 0.0, "notional_usd": 0.0})
    eq = pd.DataFrame(equity_rows).set_index("time")
    tr = pd.DataFrame(trades)
    peak = eq.equity.cummax()
    max_dd = float(((eq.equity / peak) - 1).min() * 100)
    summary = {"strategy": strategy_label, "snapshots_or_bars": len(d),
               "executions": len(tr), "net_pnl_usd": float(eq.equity.iloc[-1] - cfg.starting_cash),
               "return_pct": float((eq.equity.iloc[-1] / cfg.starting_cash - 1) * 100),
               "max_drawdown_pct": max_dd, "turnover_usd": turnover,
               "ending_qty": qty, "liquidity_model": str(d.liquidity_model.iloc[0])}
    return eq, tr, summary


def run_comparison(df: pd.DataFrame, cfg: Config, output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    active = "depth_bid" in df and "depth_ask" in df
    modes = [("OU_inventory_baseline", "baseline")]
    if active:
        modes += [("OU_inventory_reaction_diffusion", "reaction_diffusion"),
                  ("OU_inventory_Navier_Stokes", "navier_stokes")]
    all_eq, all_tr, results = [], [], []
    for name, model in modes:
        features = make_features(df, cfg, model)
        eq, tr, summary = backtest(features, cfg, strategy_label=name)
        features.drop(columns=[c for c in ["depth_bid", "depth_ask"] if c in features]).to_csv(output / f"features_{name}.csv")
        eq.to_csv(output / f"equity_{name}.csv")
        tr.to_csv(output / f"trades_{name}.csv", index=False)
        all_eq.append(eq.equity.rename(name))
        all_tr.append(tr)
        results.append(summary)
    pd.DataFrame(results).to_csv(output / "summary.csv", index=False)
    with (output / "config.json").open("w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    try:
        import matplotlib.pyplot as plt
        curves = pd.concat(all_eq, axis=1)
        ax = curves.plot(title="Single-asset inventory MR: OU / reaction-diffusion / Navier-Stokes analogue", figsize=(10, 5))
        ax.set_ylabel("Marked-to-market equity (USD)")
        ax.grid(alpha=0.2)
        ax.figure.tight_layout()
        ax.figure.savefig(output / "equity_comparison.png", dpi=140)
        plt.close(ax.figure)
    except ImportError:
        pass
    return results


def self_test() -> None:
    cfg = Config(fair_span=25, risk_window=20, z_entry=1.0,
                 fee_bps_per_side=1, slippage_bps_per_side=0.5)
    const = np.full(cfg.book_bins, 0.25)
    # Equilibrium remains stationary with constant pressure and zero velocity.
    stationary = ns_forward(const, const, np.zeros_like(const), cfg)
    np.testing.assert_allclose(stationary, const, rtol=1e-8, atol=1e-8)
    # With zero volume sources and reflecting walls, mass is conserved.
    conserved = Config(ns_source_rate=0.0)
    varying = np.array([0.1, 0.2, 0.4, 0.5, 0.3, 0.2, 0.1, 0.1])
    pred = ns_forward(varying, varying, np.zeros_like(varying), conserved)
    assert np.all(pred >= 0)
    np.testing.assert_allclose(pred.sum(), varying.sum(), rtol=2e-6)
    assert (pde_forward(varying, varying, cfg) >= 0).all()
    d = simulate_data(350, 123, cfg)
    feature_sets = {m: make_features(d, cfg, m)
                    for m in ("baseline", "reaction_diffusion", "navier_stokes")}
    for model, f in feature_sets.items():
        assert np.isfinite(f.fluid_imbalance).all()
        assert (f.fluid_imbalance.abs() <= 1 + 1e-12).all()
        eq, tr, summ = backtest(f, cfg, strategy_label=model)
        assert (eq.notional_usd.abs() <= cfg.max_notional + 1e-6).all()
        assert abs(summ["ending_qty"]) < 1e-10
        assert eq.equity.notna().all()
    altered = d.copy()
    cut = 280
    altered.loc[altered.index[cut]:, "signal_mid"] *= 1.15
    altered.loc[altered.index[cut]:, "depth_bid"] = pd.Series(
        [2 * x for x in altered.depth_bid.iloc[cut:]],
        index=altered.index[cut:]).astype(object)
    for model, f in feature_sets.items():
        f2 = make_features(altered, cfg, model)
        cols = ["z", "mr_alpha_bps", "vol_bps", "fluid_imbalance", "alpha_bps"]
        pd.testing.assert_frame_equal(f[cols].iloc[:cut], f2[cols].iloc[:cut])
    tmp = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=100, freq="min", tz="UTC"),
                        "open": np.linspace(10, 12, 100), "close": np.linspace(10, 12, 100)})
    d2 = from_ohlcv(tmp, cfg)
    try:
        make_features(d2, cfg, "navier_stokes")
    except ValueError:
        pass
    else:
        raise AssertionError("OHLCV-only incorrectly activated Navier-Stokes")
    print("PASS: NS stationary solution, volume conservation, positivity,"
          " causality, inventory cap, OHLCV guard, liquidation")

def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = pa.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true", help="synthetic price + synthetic L2")
    group.add_argument("--fetch", action="store_true", help="get closed Hyperliquid 1-minute candles; OU only")
    group.add_argument("--csv", type=Path, help="local OHLCV file; OU only")
    group.add_argument("--collect", action="store_true", help="capture PUBLIC current L2 snapshots to JSONL")
    group.add_argument("--snapshots", type=Path, help="replay JSONL L2 snapshots")
    group.add_argument("--self-test", action="store_true")
    pa.add_argument("--hours", type=int, default=72)
    pa.add_argument("--seconds", type=float, default=120)
    pa.add_argument("--interval", type=float, default=5)
    pa.add_argument("--out", type=Path, default=Path("inventory_mr_results"))
    pa.add_argument("--seed", type=int, default=24)
    pa.add_argument("--n", type=int, default=1200)
    pa.add_argument("--ns-horizon", type=float, default=5.0,
                    help="NS forward horizon in seconds, ideally same as L2 observation cadence")
    args = pa.parse_args()
    if args.self_test:
        self_test()
        return
    cfg = Config(ns_horizon_sec=args.ns_horizon)
    if args.collect:
        collect_snapshots(args.out / "orderbook_snapshots.jsonl", args.seconds, args.interval, cfg)
        print("Collect for at least a few hundred snapshots before replaying; this is forward-only data.")
        return
    if args.demo:
        data = simulate_data(args.n, args.seed, cfg)
        label = "SYNTHETIC L2 (NOT VALIDATED MARKET RESULTS)"
    elif args.snapshots:
        data = from_snapshots(args.snapshots, cfg)
        label = "REPLAYED PUBLIC L2 SNAPSHOTS; IDEALIZED NEXT-SNAPSHOT FILLS"
    elif args.csv:
        data = from_ohlcv(pd.read_csv(args.csv), cfg)
        label = "OHLCV ONLY: PDE INACTIVE; ASSUMED SPREAD/EXECUTION"
    else:
        data = from_ohlcv(fetch_ohlcv(args.hours, cfg), cfg)
        label = "REAL OHLCV ONLY: PDE INACTIVE; ASSUMED SPREAD/EXECUTION"
    if len(data) < max(cfg.fair_span, cfg.risk_window) + 20:
        raise ValueError("Not enough snapshots: require at least 100, preferably > 1000")
    if "depth_bid" in data:
        sample_sec = data.index.to_series().diff().dt.total_seconds().median()
        if sample_sec > cfg.ns_horizon_sec * 3:
            print(f"WARNING: snapshot interval {sample_sec:g}s >> NS horizon {cfg.ns_horizon_sec:g}s;"
                  " signal forecasting horizon is not aligned with execution sampling.")
    print("\nINPUT:", label, "| rows:", len(data))
    results = run_comparison(data, cfg, args.out)
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print("\nSaved CSV, summary, and optionally PNG under:", args.out.resolve())
    print("PAPER TEST ONLY: optimistic next-snapshot quote execution; NO funding, queue simulation, latency, full impact.")
    print("Navier-Stokes is a stylized L2 analogue, not a verified physical law of markets.")


if __name__ == "__main__":
    main()

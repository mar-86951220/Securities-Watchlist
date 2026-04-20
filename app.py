import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import psycopg2
from datetime import date, timedelta

st.set_page_config(page_title="Watchlist Peer Analysis", layout="wide", page_icon="📈")

DARK_BG = "#0e1117"
PAPER_BG = "#1a1f2e"
GRID_COLOR = "#2a2f3e"
POSITIVE_FILL = "rgba(100, 180, 255, 0.5)"
NEGATIVE_FILL = "rgba(20, 20, 50, 0.8)"

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.5rem;
    }
    .card-label { color: #8b95a8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .card-ticker { font-size: 2rem; font-weight: 700; color: #ffffff; }
    .card-name { color: #8b95a8; font-size: 0.8rem; margin-bottom: 0.3rem; }
    .badge-pos { background: #1a4a2e; color: #4ade80; border-radius: 5px; padding: 2px 8px; font-size: 0.8rem; font-weight: 600; }
    .badge-neg { background: #4a1a1a; color: #f87171; border-radius: 5px; padding: 2px 8px; font-size: 0.8rem; font-weight: 600; }
    .section-title { font-size: 1.4rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.2rem; }
    .section-subtitle { color: #8b95a8; font-size: 0.85rem; margin-bottom: 1rem; }
    div[data-baseweb="tag"] { background-color: #3a4060 !important; }
</style>
""", unsafe_allow_html=True)


def get_db_url() -> str:
    url = st.secrets["DATABASE_URL"]
    # Strip channel_binding param — not supported by psycopg2
    if "channel_binding" in url:
        parts = url.split("?")
        params = [p for p in parts[1].split("&") if not p.startswith("channel_binding")]
        url = parts[0] + "?" + "&".join(params)
    return url


@st.cache_data(ttl=300)
def load_watchlist() -> pd.DataFrame:
    conn = psycopg2.connect(get_db_url())
    df = pd.read_sql("SELECT ticker, name, security_type AS \"SecurityType\" FROM watchlist ORDER BY ticker", conn)
    conn.close()
    df.columns = ["Ticker", "Name", "SecurityType"]
    return df


@st.cache_data(ttl=3600)
def fetch_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = [tickers[0]]
    return prices.dropna(how="all")


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices / prices.bfill().iloc[0]


def compute_returns(normalized: pd.DataFrame) -> pd.Series:
    return (normalized.ffill().iloc[-1] - normalized.bfill().iloc[0]) / normalized.bfill().iloc[0]


def compute_peer_avg(normalized: pd.DataFrame, ticker: str) -> pd.Series:
    others = [c for c in normalized.columns if c != ticker]
    return normalized[others].mean(axis=1) if others else pd.Series(np.nan, index=normalized.index)


def period_to_dates(label: str):
    today = date.today()
    mapping = {
        "1M": 30, "3M": 91, "6M": 182,
        "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5, "10Y": 365 * 10,
    }
    return (today - timedelta(days=mapping.get(label, 182))).isoformat(), today.isoformat()


def base_layout(title="", height=300):
    return dict(
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color="#c0c8d8", size=11),
        title=dict(text=title, font=dict(size=13, color="#c0c8d8"), x=0),
        height=height,
        margin=dict(l=40, r=20, t=40 if title else 20, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )


def plot_normalized_chart(normalized: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in normalized.columns:
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col, line=dict(width=1.5)))
    layout = base_layout(height=380)
    layout["yaxis"]["title"] = "Normalized price"
    layout["xaxis"]["title"] = "Date"
    layout["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="v", x=1.01)
    fig.update_layout(**layout)
    return fig


def plot_peer_comparison(normalized: pd.DataFrame, ticker: str):
    peer = compute_peer_avg(normalized, ticker)
    stock = normalized[ticker]

    fig_overlay = go.Figure([
        go.Scatter(x=stock.index, y=peer.values, mode="lines", name="Peer average",
                   line=dict(color="#8b95a8", width=1.2)),
        go.Scatter(x=stock.index, y=stock.values, mode="lines", name=ticker,
                   line=dict(color="#ef4444", width=1.5)),
    ])
    ov = base_layout(title=f"{ticker} vs peer average", height=230)
    ov["yaxis"]["title"] = "Price"
    ov["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9), orientation="h", x=0, y=-0.25)
    fig_overlay.update_layout(**ov)

    delta = stock - peer
    fig_delta = go.Figure([
        go.Scatter(x=delta.index, y=delta.values, mode="lines",
                   fill="tozeroy", line=dict(color="rgba(100,180,255,0.8)", width=0.5),
                   fillcolor=POSITIVE_FILL, name="Delta"),
    ])
    dl = base_layout(title=f"{ticker} minus peer average", height=230)
    dl["yaxis"]["title"] = "Delta"
    dl["showlegend"] = False
    fig_delta.update_layout(**dl)
    return fig_overlay, fig_delta


def plot_type_performance(normalized: pd.DataFrame, watchlist: pd.DataFrame) -> go.Figure:
    colors = {"Common Stock": "#60a5fa", "ETF": "#34d399", "Commodity ETF/Trust": "#fbbf24"}
    ticker_type = watchlist.set_index("Ticker")["SecurityType"].to_dict()
    fig = go.Figure()
    seen_types = list(dict.fromkeys(ticker_type.get(t, "") for t in normalized.columns))
    for stype in seen_types:
        tickers_in = [t for t in normalized.columns if ticker_type.get(t) == stype]
        if not tickers_in:
            continue
        avg = normalized[tickers_in].mean(axis=1)
        fig.add_trace(go.Scatter(x=avg.index, y=avg.values, mode="lines", name=stype,
                                  line=dict(width=2, color=colors.get(stype, "#a78bfa"))))
    layout = base_layout(height=320)
    layout["yaxis"]["title"] = "Avg normalized price"
    layout["xaxis"]["title"] = "Date"
    layout["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="v", x=1.01)
    fig.update_layout(**layout)
    return fig


def plot_correlation_heatmap(prices: pd.DataFrame) -> go.Figure:
    corr = prices.pct_change().dropna().corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmin=-1, zmax=1,
        colorbar=dict(title="Corr", thickness=12),
        text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=9),
    ))
    fig.update_layout(**base_layout(height=420))
    return fig


def plot_rolling_volatility(prices: pd.DataFrame, window: int = 30) -> go.Figure:
    vol = prices.pct_change().rolling(window).std() * np.sqrt(252) * 100
    fig = go.Figure()
    for col in vol.columns:
        fig.add_trace(go.Scatter(x=vol.index, y=vol[col], mode="lines", name=col, line=dict(width=1.5)))
    layout = base_layout(height=340)
    layout["yaxis"]["title"] = "Annualized volatility (%)"
    layout["xaxis"]["title"] = "Date"
    layout["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="v", x=1.01)
    fig.update_layout(**layout)
    return fig


def plot_drawdown(prices: pd.DataFrame) -> go.Figure:
    norm = normalize_prices(prices)
    drawdown = ((norm - norm.cummax()) / norm.cummax() * 100).min()
    df = drawdown.reset_index()
    df.columns = ["Ticker", "MaxDrawdown"]
    df = df.sort_values("MaxDrawdown")
    colors = ["#ef4444" if v < -20 else "#f97316" if v < -10 else "#facc15" for v in df["MaxDrawdown"]]
    fig = go.Figure(go.Bar(
        x=df["Ticker"], y=df["MaxDrawdown"], marker_color=colors,
        text=df["MaxDrawdown"].round(1).astype(str) + "%", textposition="outside",
    ))
    layout = base_layout(height=320)
    layout["yaxis"]["title"] = "Max drawdown (%)"
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def render_metric_card(label: str, ticker: str, name: str, ret: float):
    sign = "+" if ret >= 0 else ""
    badge = "badge-pos" if ret >= 0 else "badge-neg"
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">{label}</div>
        <div class="card-ticker">{ticker}</div>
        <div class="card-name">{name}</div>
        <span class="{badge}">{sign}{ret*100:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)


def render_ranked_list(label: str, items: list[tuple[str, str, float]], positive: bool):
    """Render a ranked top/bottom list card. items = [(ticker, name, return), ...]"""
    badge = "badge-pos" if positive else "badge-neg"
    rows_html = ""
    for rank, (ticker, name, ret) in enumerate(items, 1):
        sign = "+" if ret >= 0 else ""
        rows_html += f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding: 0.45rem 0; border-bottom: 1px solid #2a2f3e;">
            <div style="display:flex; align-items:center; gap:0.7rem;">
                <span style="color:#8b95a8; font-size:0.75rem; width:1rem;">#{rank}</span>
                <div>
                    <div style="font-weight:700; color:#fff; font-size:0.95rem;">{ticker}</div>
                    <div style="color:#8b95a8; font-size:0.72rem;">{name}</div>
                </div>
            </div>
            <span class="{badge}">{sign}{ret*100:.2f}%</span>
        </div>"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label" style="margin-bottom:0.5rem;">{label}</div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────

watchlist = load_watchlist()
all_tickers = watchlist["Ticker"].tolist()
ticker_name = watchlist.set_index("Ticker")["Name"].to_dict()
type_map = watchlist.set_index("Ticker")["SecurityType"].to_dict()

st.markdown("# 📈 Watchlist peer analysis")
st.markdown('<p class="section-subtitle">Compare securities against others in their peer group.</p>', unsafe_allow_html=True)

col_tickers, col_period = st.columns([3, 1])
with col_tickers:
    selected = st.multiselect(
        "Stock tickers", options=all_tickers, default=all_tickers[:7],
        format_func=lambda t: f"{t} – {ticker_name.get(t, '')}",
    )
with col_period:
    period = st.radio("Time horizon", ["1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y"], index=2, horizontal=True)

if len(selected) < 2:
    st.warning("Select at least 2 tickers to enable peer comparison.")
    st.stop()

start_date, end_date = period_to_dates(period)

with st.spinner("Fetching price data…"):
    prices = fetch_prices(tuple(sorted(selected)), start_date, end_date)

prices = prices.reindex(columns=[t for t in selected if t in prices.columns])
missing = [t for t in selected if t not in prices.columns]
if missing:
    st.warning(f"No data found for: {', '.join(missing)}. They will be excluded.")

normalized = normalize_prices(prices)
returns = compute_returns(normalized)
tickers_list = list(prices.columns)

# ── NORMALIZED PRICE CHART ────────────────────────────────────────────────────
st.plotly_chart(plot_normalized_chart(normalized), use_container_width=True)

# ── BEST / WORST OVERALL ──────────────────────────────────────────────────────
st.markdown('<p class="section-title">Best &amp; worst performers</p>', unsafe_allow_html=True)
col_b, col_w = st.columns(2)
with col_b:
    render_metric_card("Best stock", returns.idxmax(), ticker_name.get(returns.idxmax(), ""), returns.max())
with col_w:
    render_metric_card("Worst stock", returns.idxmin(), ticker_name.get(returns.idxmin(), ""), returns.min())

# ── TOP 3 / BOTTOM 3 BY SECURITY TYPE ────────────────────────────────────────
st.markdown('<p class="section-title">Top &amp; bottom 3 by security type</p>', unsafe_allow_html=True)
N = 3
for stype in list(dict.fromkeys(type_map.get(t, "") for t in tickers_list)):
    tickers_in = [t for t in tickers_list if type_map.get(t) == stype]
    if not tickers_in:
        continue
    tr = returns[tickers_in].sort_values(ascending=False)
    top = [(t, ticker_name.get(t, ""), tr[t]) for t in tr.head(N).index]
    bottom = [(t, ticker_name.get(t, ""), tr[t]) for t in tr.tail(N).index[::-1]]
    st.markdown(f"**{stype}**")
    c1, c2 = st.columns(2)
    with c1:
        render_ranked_list(f"Top {min(N, len(tickers_in))}", top, positive=True)
    with c2:
        render_ranked_list(f"Bottom {min(N, len(tickers_in))}", bottom, positive=False)

# ── GROUPED PERFORMANCE BY TYPE ───────────────────────────────────────────────
st.markdown('<p class="section-title">Cumulative performance by security type</p>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Average normalized price across all selected securities of each type.</p>', unsafe_allow_html=True)
st.plotly_chart(plot_type_performance(normalized, watchlist), use_container_width=True)

# ── INDIVIDUAL STOCK vs PEER AVERAGE ─────────────────────────────────────────
st.markdown('<p class="section-title">Individual stocks vs peer average</p>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">The peer average when analyzing stock X always excludes X itself.</p>', unsafe_allow_html=True)
for i in range(0, len(tickers_list), 2):
    cols = st.columns(4)
    for j, ticker in enumerate(tickers_list[i:i+2]):
        fig_ov, fig_dl = plot_peer_comparison(normalized, ticker)
        with cols[j * 2]:
            st.plotly_chart(fig_ov, use_container_width=True)
        with cols[j * 2 + 1]:
            st.plotly_chart(fig_dl, use_container_width=True)

# ── RETURN RANKINGS TABLE ─────────────────────────────────────────────────────
st.markdown('<p class="section-title">Return rankings</p>', unsafe_allow_html=True)
peer_avg_overall = returns.mean()
rankings = [{
    "Ticker": t, "Name": ticker_name.get(t, ""), "Type": type_map.get(t, ""),
    "Period Return %": round(returns[t] * 100, 2),
    "vs Peer Avg %": round((returns[t] - returns[[x for x in tickers_list if x != t]].mean()) * 100, 2),
} for t in tickers_list]
rank_df = pd.DataFrame(rankings).sort_values("Period Return %", ascending=False).reset_index(drop=True)

def color_return(val):
    return f"color: {'#4ade80' if val > 0 else '#f87171' if val < 0 else '#ffffff'}; font-weight: 600"

st.dataframe(rank_df.style.map(color_return, subset=["Period Return %", "vs Peer Avg %"]),
             use_container_width=True, hide_index=True)

# ── ADDITIONAL ANALYTICS ──────────────────────────────────────────────────────
with st.expander("📊 Correlation heatmap", expanded=False):
    st.markdown("Daily return correlations between selected securities.")
    st.plotly_chart(plot_correlation_heatmap(prices), use_container_width=True)

with st.expander("📉 Rolling 30-day volatility", expanded=False):
    st.markdown("Annualized volatility (30-day rolling window). Higher = riskier.")
    st.plotly_chart(plot_rolling_volatility(prices), use_container_width=True)

with st.expander("🔻 Maximum drawdown", expanded=False):
    st.markdown("Largest peak-to-trough decline over the selected period.")
    st.plotly_chart(plot_drawdown(prices), use_container_width=True)

with st.expander("🗃️ Raw data", expanded=False):
    display = prices.copy()
    display.index = display.index.date
    st.dataframe(display.style.format("{:.4f}"), use_container_width=True)

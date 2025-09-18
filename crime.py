import time
import datetime as dt
import re
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

try:
    from shapely.geometry import MultiPoint, Point
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

# --------------------
# CONFIG
# --------------------
st.set_page_config(page_title="Karachi Crime Intelligence Dashboard", layout="wide")

# --------------------
# STATE
# --------------------
DEFAULT_LAT, DEFAULT_LON = 24.8607, 67.0011
COOLDOWN_SECONDS = 15 * 60  # 15 minutes

if "reports" not in st.session_state:
    st.session_state.reports = []
if "selected_location" not in st.session_state:
    st.session_state.selected_location = {"lat": DEFAULT_LAT, "lon": DEFAULT_LON}
if "last_submit_at" not in st.session_state:
    st.session_state.last_submit_at = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "page" not in st.session_state:
    st.session_state.page = "Report"
if "clear_filters" not in st.session_state:
    st.session_state.clear_filters = False

# --------------------
# THEMES
# --------------------
def apply_theme():
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
        body { background-color: #0e0f12; color: #e5e5e5; font-family: 'Segoe UI', sans-serif; }
        .block-container { padding-top: 0.75rem; }
        .card { background: rgba(30, 30, 32, 0.85); border-radius: 16px;
                padding: 16px; border: 1px solid rgba(255,255,255,0.05);
                box-shadow: 0 8px 24px rgba(0,0,0,0.25); }
        .map-wrap { border-radius: 16px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08);
                    box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
        .nav-title { font-weight: 800; margin-bottom: 8px; }
        .nav-note { color: #9aa0a6; font-size: 12px; margin-bottom: 8px; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        body { background-color: #f5f5f5; color: #222; font-family: 'Segoe UI', sans-serif; }
        .block-container { padding-top: 0.75rem; }
        .card { background: #fff; border-radius: 16px; padding: 16px; border: 1px solid #ddd;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
        .map-wrap { border-radius: 16px; overflow: hidden; border: 1px solid #ddd;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
        .nav-title { font-weight: 800; margin-bottom: 8px; }
        .nav-note { color: #666; font-size: 12px; margin-bottom: 8px; }
        </style>
        """, unsafe_allow_html=True)

apply_theme()

# --------------------
# GEOCODER
# --------------------
@st.cache_resource
def get_geocoder():
    geolocator = Nominatim(user_agent="karachi_crime_dashboard")
    return RateLimiter(geolocator.geocode, min_delay_seconds=1), RateLimiter(geolocator.reverse, min_delay_seconds=1)

geocode, reverse = get_geocoder()

def reverse_geocode(lat, lon):
    try:
        loc = reverse((lat, lon), language="en")
        if not loc:
            return {}
        addr = loc.raw.get("address", {})
        area = addr.get("neighbourhood") or addr.get("suburb") or addr.get("city_district") or ""
        return {"area": area}
    except Exception:
        return {}

def geocode_address(query: str) -> Optional[Tuple[float, float]]:
    if not query or not query.strip():
        return None
    try:
        loc = geocode(query.strip() + ", Karachi, Pakistan")
        if not loc:
            return None
        return float(loc.latitude), float(loc.longitude)
    except Exception:
        return None

# --------------------
# LANGUAGE FILTER
# --------------------
BANNED_WORDS = {"badword1", "badword2", "offensivephrase"}
def contains_profanity(text: str) -> bool:
    if not text:
        return False
    pattern = re.compile(r"(" + "|".join(re.escape(w) for w in BANNED_WORDS) + r")", re.IGNORECASE)
    return bool(pattern.search(text))

# --------------------
# SIDEBAR NAVIGATION
# --------------------
with st.sidebar:
    st.markdown("### Theme")
    if st.button("Toggle Light/Dark"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        apply_theme()
        st.rerun()

    st.markdown("### Navigation")
    # Vertical cylinders via buttons (Streamlit limits deep CSS on widgets; functionally equivalent)
    nav_options = ["Report", "Explore", "AI Analytics", "About"]
    for opt in nav_options:
        if st.button(("● " if st.session_state.page == opt else "○ ") + opt, key=f"nav_{opt}"):
            st.session_state.page = opt
            st.rerun()

    st.markdown("### Quick actions")
    if st.button("Center City"):
        st.session_state.selected_location = {"lat": DEFAULT_LAT, "lon": DEFAULT_LON}
    if st.button("Use Last Pin") and st.session_state.reports:
        last = st.session_state.reports[-1]
        st.session_state.selected_location = {"lat": last["lat"], "lon": last["lon"]}
    if st.button("Clear Filters"):
        st.session_state.clear_filters = True

# --------------------
# PAGE LOGIC
# --------------------
page = st.session_state.page

# =========================
# REPORT PAGE
# =========================
if page == "Report":
    st.title("Report an Incident")
    colL, colR = st.columns([1.1, 0.9], gap="large")

    with colL:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Select location")

        m = folium.Map(
            location=[st.session_state.selected_location["lat"], st.session_state.selected_location["lon"]],
            zoom_start=13, tiles="CartoDB positron", control_scale=True
        )
        folium.CircleMarker(
            location=[st.session_state.selected_location["lat"], st.session_state.selected_location["lon"]],
            radius=9, color="#0078ff", fill=True, fill_color="#0078ff"
        ).add_to(m)

        st.markdown("<div class='map-wrap'>", unsafe_allow_html=True)
        map_event = st_folium(m, height=440, key="report_map")
        st.markdown("</div>", unsafe_allow_html=True)

        if map_event and map_event.get("last_clicked"):
            lat, lon = map_event["last_clicked"]["lat"], map_event["last_clicked"]["lng"]
            st.session_state.selected_location = {"lat": lat, "lon": lon}

        st.markdown("</div>", unsafe_allow_html=True)

    with colR:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Incident details")

        now_ts = time.time()
        cooldown_active = False
        if st.session_state.last_submit_at and now_ts - st.session_state.last_submit_at < COOLDOWN_SECONDS:
            cooldown_active = True
            remaining = int(COOLDOWN_SECONDS - (now_ts - st.session_state.last_submit_at))
            st.warning(f"Please wait {remaining//60}m {remaining%60}s before submitting again.")

        with st.form("incident_form"):
            crime_type = st.selectbox(
                "Type of crime",
                ["Mobile Snatching", "Robbery", "Harassment", "Other"]
            )
            area = st.text_input("Area or landmark", placeholder="DHA Phase 6, Clifton Block 5")
            time_of_incident = st.time_input("Time of incident", value=dt.datetime.now().time())
            description = st.text_area(
                "Brief description",
                placeholder="What happened? Keep it respectful and factual.",
                height=120,
                max_chars=600
            )
            # Address-to-pin helper (optional, no extra empty inputs)
            address_query = st.text_input("Find address (optional)", placeholder="House 10, Street 5, DHA, Karachi")
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                do_find = st.form_submit_button("Find on map")
            with col_btn2:
                submitted = st.form_submit_button("Submit report", disabled=cooldown_active)

        # Address to map sync
        if do_find and address_query.strip():
            coords = geocode_address(address_query)
            if coords:
                lat, lon = coords
                st.session_state.selected_location = {"lat": lat, "lon": lon}
                rg = reverse_geocode(lat, lon)
                if rg.get("area"):
                    area = rg["area"]
                st.success("Address located and pinned.")
                st.rerun()
            else:
                st.warning("Could not locate that address. Try specifying area and city.")

        # Submit handling
        if submitted and not cooldown_active:
            errs = []
            if not description.strip():
                errs.append("Please provide a brief description.")
            if contains_profanity(description):
                errs.append("Please remove inappropriate language from the description.")
            if not area.strip():
                errs.append("Please enter an area or landmark.")

            if errs:
                for e in errs:
                    st.error(e)
            else:
                st.session_state.reports.append({
                    "SubmittedAt": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Crime": crime_type,
                    "Area": area.strip(),
                    "Time": time_of_incident.strftime("%H:%M"),
                    "Description": description.strip(),
                    "lat": st.session_state.selected_location["lat"],
                    "lon": st.session_state.selected_location["lon"]
                })
                st.session_state.last_submit_at = time.time()
                st.success("Incident reported.")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# EXPLORE PAGE
# =========================
if page == "Explore":
    st.title("Explore Reports")

    df_all = pd.DataFrame(st.session_state.reports)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Filters")

    default_window_index = 2  # 30d
    default_days = 30
    if st.session_state.clear_filters:
        default_window_index = 2
        default_days = 30
        st.session_state.clear_filters = False

    colf1, colf2, colf3 = st.columns([1, 1, 2])
    with colf1:
        window = st.radio("Time window", ["24h", "7d", "30d", "90d", "All"], horizontal=True, index=default_window_index)
    with colf2:
        custom_days = st.number_input("Custom days (ignored unless needed)", min_value=1, max_value=365, value=default_days, step=1)
    with colf3:
        query = st.text_input("Search (area or description)", value="")

    def filter_by_window(df: pd.DataFrame, win: str) -> pd.DataFrame:
        if df.empty:
            return df
        def parse_dt(s):
            try:
                return dt.datetime.strptime(s, "%Y-%m-%d %H:%M")
            except Exception:
                return None
        df = df.copy()
        df["SubmittedAt_dt"] = df["SubmittedAt"].apply(parse_dt)
        now_dt = dt.datetime.now()
        if win == "24h":
            min_dt = now_dt - dt.timedelta(days=1)
        elif win == "7d":
            min_dt = now_dt - dt.timedelta(days=7)
        elif win == "30d":
            min_dt = now_dt - dt.timedelta(days=30)
        elif win == "90d":
            min_dt = now_dt - dt.timedelta(days=90)
        elif win == "All":
            return df
        else:
            return df
        return df[df["SubmittedAt_dt"] >= min_dt]

    df = filter_by_window(df_all, window)

    all_types = sorted(df["Crime"].dropna().unique().tolist()) if not df.empty else []
    types_sel = st.multiselect("Crime type", options=all_types, default=all_types)

    if not df.empty:
        mask = pd.Series(True, index=df.index)
        if types_sel:
            mask &= df["Crime"].isin(types_sel)
        if query.strip():
            q = query.strip().lower()
            mask &= (
                df["Description"].str.lower().str.contains(q, na=False) |
                df["Area"].str.lower().str.contains(q, na=False)
            )
        df_view = df[mask].copy()
    else:
        df_view = df

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Recent reports")
    if not df_view.empty:
        cols_show = ["SubmittedAt", "Crime", "Time", "Area", "Description", "lat", "lon"]
        cols_show = [c for c in cols_show if c in df_view.columns]
        st.dataframe(df_view.sort_values("SubmittedAt", ascending=False)[cols_show], use_container_width=True, height=360)
        csv = df_view.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="karachi_crime_reports.csv", mime="text/csv")
    else:
        st.info("No reports match the selected filters.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Heatmap")
    if not df_view.empty and {"lat", "lon"}.issubset(df_view.columns):
        coords = df_view[["lat", "lon"]].dropna()
        if not coords.empty:
            m = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=12, tiles="CartoDB positron", control_scale=True)
            HeatMap(
                data=coords.values.tolist(),
                radius=18,
                blur=25,
                max_zoom=15
            ).add_to(m)
            # Add subtle markers
            for _, r in coords.iterrows():
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=2, color="#0078ff", fill=True, fill_color="#0078ff", fill_opacity=0.7
                ).add_to(m)
            st.markdown("<div class='map-wrap'>", unsafe_allow_html=True)
            st_folium(m, height=520, key="explore_heatmap")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No location-based reports available.")
    else:
        st.info("No reports to visualize yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# AI ANALYTICS PAGE
# =========================
if page == "AI Analytics":
    st.title("AI Analytics")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Potential hotspots (DBSCAN + KDE)")

    df = pd.DataFrame(st.session_state.reports)
    if df.empty or not {"lat", "lon"}.issubset(df.columns) or df[["lat","lon"]].dropna().shape[0] < 3:
        st.info("Not enough data for analytics yet.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            eps_m = st.slider("Cluster radius (m)", 50, 1000, 250, 25)
        with c2:
            min_samples = st.slider("Min points per cluster", 2, 15, 4, 1)
        with c3:
            bw_m = st.slider("KDE bandwidth (m)", 50, 1200, 300, 25)

        meters_per_deg_lat = 111_320.0  # approx near Karachi
        eps_deg = eps_m / meters_per_deg_lat
        bw_deg = bw_m / meters_per_deg_lat

        X = df[["lat", "lon"]].dropna().to_numpy()

        # DBSCAN clusters
        db = DBSCAN(eps=eps_deg, min_samples=min_samples).fit(X)
        labels = db.labels_
        unique_labels = [l for l in np.unique(labels) if l != -1]
        dfc = pd.DataFrame(X, columns=["lat", "lon"])
        dfc["cluster"] = labels

        # KDE risk surface
        kde = KernelDensity(bandwidth=bw_deg, kernel="gaussian")
        kde.fit(X)
        lat_min, lat_max = X[:,0].min()-0.01, X[:,0].max()+0.01
        lon_min, lon_max = X[:,1].min()-0.01, X[:,1].max()+0.01
        grid_n = 60
        g_lats = np.linspace(lat_min, max(lat_max, lat_min+1e-3), grid_n)
        g_lons = np.linspace(lon_min, max(lon_max, lon_min+1e-3), grid_n)
        grid_points = np.array([(la, lo) for la in g_lats for lo in g_lons])
        dens = np.exp(kde.score_samples(grid_points))
        dens_norm = (dens - dens.min()) / (dens.max() - dens.min() + 1e-9)
        heat_data = [[float(la), float(lo), float(w)] for (la, lo), w in zip(grid_points, dens_norm)]

        m = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=12, tiles="CartoDB positron", control_scale=True)
        HeatMap(data=heat_data, radius=18, blur=28, max_zoom=16, min_opacity=0.2).add_to(m)

        colors = ["#0078ff", "#00c2a8", "#ff6b6b", "#fbbf24", "#a78bfa", "#34d399", "#60a5fa"]
        for i, cl in enumerate(unique_labels):
            pts = dfc[dfc["cluster"] == cl][["lat", "lon"]].to_numpy()
            col = colors[i % len(colors)]
            center = pts.mean(axis=0).tolist()
            folium.CircleMarker(location=center, radius=7, color=col, fill=True, fill_color=col, fill_opacity=0.95,
                                tooltip=f"Cluster {cl} (n={len(pts)})").add_to(m)
            if HAS_SHAPELY and len(pts) >= 3:
                hull = MultiPoint([Point(p[1], p[0]) for p in pts]).convex_hull
                if hull.geom_type == "Polygon":
                    coords = [(y, x) for x, y in np.array(hull.exterior.coords)]
                    folium.PolyLine(coords, color=col, weight=3, opacity=0.7).add_to(m)
            else:
                for p in pts:
                    folium.CircleMarker(location=[p[0], p[1]], radius=3, color=col, fill=True, fill_color=col, fill_opacity=0.7).add_to(m)

        st.markdown("<div class='map-wrap'>", unsafe_allow_html=True)
        st_folium(m, height=560, key="ai_map")
        st.markdown("</div>", unsafe_allow_html=True)

        # Simple cluster confidence
        conf = "Low"
        largest = dfc["cluster"].value_counts().drop(labels=-1, errors="ignore").max() if not dfc.empty else 0
        if len(unique_labels) >= 1 and largest >= max(min_samples, 5):
            conf = "Medium"
        if len(unique_labels) >= 2 and largest >= max(min_samples, 8):
            conf = "High"
        st.caption(f"Cluster confidence: {conf}. Statistical estimate; not definitive.")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ABOUT PAGE
# =========================
if page == "About":
    st.title("About")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("""
    This dashboard helps Karachi residents and visitors turn isolated incidents into shared awareness.
    Each report is a signal; signals stack into patterns; patterns inform choices — a safer route, a better pause,
    a quick heads-up to a friend.

    The aim is clarity without noise: clean maps, visible hotspots, and practical tools for everyday vigilance.
    Contribute thoughtfully, avoid personal identifiers, and double-check locations before submitting.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

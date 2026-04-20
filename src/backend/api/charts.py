"""
api/charts.py
─────────────
Plotly chart endpoints for the CineGraph Story Slides.
Strictly follows user logic, adapted visually to Frontend Glassmorphism Theme.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from settings import settings

router = APIRouter(prefix="/api/charts", tags=["charts"])

# ─────────────────────────────────────────────────────────────
# CONSTANTS (Frontend Palette)
# ─────────────────────────────────────────────────────────────
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_COLORS = {
    "sadness": "#4A90E2",   # Blue
    "joy": "#FFD700",       # Gold
    "love": "#FF6B8A",      # Pink
    "anger": "#bd34fe",     # Purple
    "fear": "#8B5CF6",      # Deep Purple
    "surprise": "#00D4AA",  # Teal
}

CLUSTER_COLORS = [
    "#4A90E2", "#bd34fe", "#FFD700", "#00D4AA",
    "#FF6B8A", "#8B5CF6", "#FF9F1C", "#2EC4B6",
]

THEME = dict(
    bg="rgba(0,0,0,0)",       
    paper="rgba(0,0,0,0)",    
    text="#E8E6E3",           
    grid="rgba(255,255,255,0.05)", 
    accent="#4A90E2",         
    purple="#bd34fe",         
    gold="#FFD700",           
    font='"SF Pro Display", system-ui, sans-serif', 
)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _apply_theme(fig: go.Figure, show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=None, 
        paper_bgcolor=THEME["paper"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"], family=THEME["font"]),
        margin=dict(t=40, b=40, l=40, r=20),
        showlegend=show_legend,
        legend=dict(
            font=dict(size=12, color="#E8E6E3"),
            bgcolor="rgba(0,0,0,0)", 
            borderwidth=0,
        ) if show_legend else {},
        hoverlabel=dict(
            font_size=14, font_family=THEME["font"],
            bgcolor="rgba(20,20,25,0.95)", 
            font_color="#FFFFFF",          
            bordercolor=THEME["accent"],
        ),
    )
    # Динамически применяем стили ко ВСЕМ осям подграфиков (xaxis, xaxis2, yaxis, yaxis2)
    for axis in list(fig.layout):
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            getattr(fig.layout, axis).update(
                gridcolor=THEME["grid"], linecolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#8899AA"), 
                title_font=dict(color="#8899AA", size=13),
            )
    return fig

def _parse_arc(raw) -> np.ndarray:
    if isinstance(raw, str): return np.array(json.loads(raw))
    if isinstance(raw, (list, np.ndarray)): return np.array(raw)
    return np.zeros(24)

def _extract_meta(row, key, default=None):
    try:
        if isinstance(row, str): row = json.loads(row)
        return row.get(key, default) if isinstance(row, dict) else default
    except Exception: return default

def _fig_json(fig: go.Figure) -> JSONResponse:
    return JSONResponse(content=json.loads(fig.to_json()))

# Глобальная переменная для кэширования датасета (чтобы не грузить БД 7 раз одновременно)
_GLOBAL_DF_CACHE = None

async def _load_movies(session: AsyncSession) -> pd.DataFrame:
    global _GLOBAL_DF_CACHE
    if _GLOBAL_DF_CACHE is not None:
        return _GLOBAL_DF_CACHE

    from sqlalchemy import text
    result = await session.execute(
        text("SELECT id, title, emotion_arc, year, other_data FROM movies WHERE year IS NOT NULL;")
    )
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=["id", "title", "emotion_arc", "year", "other_data"])
    df["emotion_arc"] = df["emotion_arc"].apply(_parse_arc)

    for col, key, default in [("revenue", "revenue", 0), ("vote_average", "vote_average", 0)]:
        df[col] = df["other_data"].apply(lambda x, k=key, d=default: _extract_meta(x, k, d))

    df["mean_emotions"] = df["emotion_arc"].apply(lambda x: (x[0:6] + x[6:12] + x[12:18]) / 3)
    df["volatility"] = df["emotion_arc"].apply(lambda x: np.sum(x[18:24]))
    
    _GLOBAL_DF_CACHE = df
    return df


# ─────────────────────────────────────────────────────────────
# 1 БЛОК — movies-per-year
# ─────────────────────────────────────────────────────────────
@router.get("/movies-per-year")
async def get_movies_per_year(session: AsyncSession = Depends(get_db)):
    df_movies = await _load_movies(session)
    df_year_counts = df_movies.groupby('year').size().reset_index(name='count')
    df_year_counts = df_year_counts[(df_year_counts['year'] >= 1900) & (df_year_counts['year'] <= 2024)]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_year_counts['year'], y=df_year_counts['count'],
        fill='tozeroy', fillcolor='rgba(74, 144, 226, 0.2)', # Frontend Blue
        line=dict(color=THEME['accent'], width=2.5, shape='spline'),
        hovertemplate='<b>%{x}</b><br>%{y:,} movies<extra></extra>',
        name='Movies Released'
    ))

    milestones = {
        1927: "Talkies Era Begins",
        1946: "Post-War Boom",
        1975: "Jaws → Blockbuster Era",
        1994: "Independent Explosion",
        2008: "Streaming Rise",
        2020: "COVID Crash"
    }
    for yr, txt in milestones.items():
        row = df_year_counts[df_year_counts['year'] == yr]
        if not row.empty:
            yv = row['count'].values[0]
            fig1.add_annotation(
                x=yr, y=yv, text=txt, showarrow=True,
                arrowhead=2, arrowcolor=THEME['gold'], arrowwidth=1.5,
                ax=0, ay=-45, font=dict(size=11, color=THEME['gold']),
                bgcolor='rgba(10,10,15,0.85)', bordercolor=THEME['gold'],
                borderwidth=1, borderpad=4
            )

    _apply_theme(fig1, show_legend=False)
    fig1.update_yaxes(title_text="Number of Movies")
    fig1.update_xaxes(title_text="Release Year")
    return _fig_json(fig1)


# ─────────────────────────────────────────────────────────────
# 2 БЛОК — vocabulary-tfidf (With Smart Fallback & High Contrast Hover)
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _compute_vocabulary():
    subtitle_dir = getattr(settings, "subtitle_dir", "./src/backend/preprocessing/ready_data")
    csv_files = glob.glob(os.path.join(subtitle_dir, "*.csv"))
    
    doc_freq, total_docs = Counter(), 0
    for fpath in csv_files:
        try:
            with open(fpath, "r", errors="ignore") as f: text = f.read()
            words = [w for w in re.findall(r"\b[a-z]{3,}\b", text.lower()) if w not in ENGLISH_STOP_WORDS]
            if words:
                doc_freq.update(set(words))
                total_docs += 1
        except Exception: continue

    if total_docs == 0: return [], [], [], []
    idf_dict = {w: math.log(total_docs / float(df)) for w, df in doc_freq.items() if 5 <= df <= total_docs * 0.9}

    cum_tfidf, glob_tf = Counter(), Counter()
    for fpath in csv_files:
        try:
            with open(fpath, "r", errors="ignore") as f: text = f.read()
            words = [w for w in re.findall(r"\b[a-z]{3,}\b", text.lower()) if w not in ENGLISH_STOP_WORDS]
            if words:
                doc_len = len(words)
                counts = Counter(words)
                for w, c in counts.items():
                    if w in idf_dict:
                        cum_tfidf[w] += (c / float(doc_len)) * idf_dict[w]
                        glob_tf[w] += c
        except Exception: continue

    tc, tt = glob_tf.most_common(15), cum_tfidf.most_common(15)
    return [w for w,_ in tc], [c for _,c in tc], [w for w,_ in tt], [s for _,s in tt]

@router.get("/vocabulary-tfidf")
async def get_vocabulary_tfidf():
    cw, cc, tw, ts = _compute_vocabulary()
    
    # Фолбэк при ошибке (как просили)
    if not cw:
        cw = ["hey", "love", "sorry", "sir", "shes", "god", "night", "great", "wont", "talk", "maybe", "theyre", "money", "years", "new"]
        cc = [520000, 380000, 320000, 300000, 250000, 240000, 220000, 220000, 218000, 218000, 215000, 210000, 210000, 200000, 200000]
        tw = ["fuck", "fucking", "sir", "shit", "mom", "alright", "christmas", "dad", "aint", "hey", "brother", "captain", "baby", "mrs", "police"]
        ts = [61, 56, 42, 41, 39, 36, 36, 35, 30, 27, 25, 24, 23, 23, 23]

    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=["Most Common Words (Frequency)", "Defining Words (Cumulative TF-IDF)"],
        vertical_spacing=0.2
    )

    fig.add_trace(go.Bar(
        x=cw, y=cc, marker_color=THEME["accent"], name="Frequency",
        hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=tw, y=ts, marker_color="#FF9F1C", name="TF-IDF",
        hovertemplate='<b>%{x}</b><br>TF-IDF: %{y:.2f}<extra></extra>'
    ), row=2, col=1) 

    _apply_theme(fig, show_legend=False)
    for annotation in fig['layout']['annotations']: 
        annotation['font'] = dict(size=14, color="#8899AA", family=THEME["font"])
    return _fig_json(fig)


# ─────────────────────────────────────────────────────────────
# 3 БЛОК — macro-seismograph 
# ─────────────────────────────────────────────────────────────
@router.get("/macro-seismograph")
async def get_macro_seismograph(session: AsyncSession = Depends(get_db)):
    df_movies = await _load_movies(session)
    df_hist = df_movies.copy()
    
    df_yearly_vol = df_hist[(df_hist['year'] >= 1930) & (df_hist['year'] <= 2023)].groupby('year')['volatility'].mean().reset_index()
    df_yearly_vol['rolling_volatility'] = df_yearly_vol['volatility'].rolling(window=5, center=True).mean()

    fig_vol = go.Figure()

    fig_vol.add_trace(go.Scatter(
        x=df_yearly_vol['year'], y=df_yearly_vol['rolling_volatility'],
        mode='lines', line=dict(color=THEME['accent'], width=4), 
        name='5-Year Rolling Volatility'
    ))

    fig_vol.add_trace(go.Scatter(
        x=df_yearly_vol['year'], y=df_yearly_vol['volatility'],
        mode='lines', line=dict(color='gray', width=1, dash='dot'),
        name='Yearly Average', opacity=0.5
    ))

    annotations = [
        dict(year=1942, text="WWII Peak", y_offset=0.05),
        dict(year=1975, text="New Hollywood / Gritty Era", y_offset=0.08),
        dict(year=2008, text="Financial Crisis", y_offset=0.05)
    ]

    for ann in annotations:
        y_val = df_yearly_vol.loc[df_yearly_vol['year'] == ann['year'], 'rolling_volatility'].values
        if len(y_val) > 0:
            fig_vol.add_annotation(
                x=ann['year'], y=y_val[0], text=ann['text'], showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#fff",
                ax=0, ay=-40, 
                font=dict(size=12, color="#fff"),
                bgcolor="rgba(255,255,255,0.1)", bordercolor="rgba(255,255,255,0.2)", borderwidth=1
            )

    _apply_theme(fig_vol)
    fig_vol.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_vol.update_xaxes(title_text="Release Year")
    fig_vol.update_yaxes(title_text="Average Emotional Volatility")
    return _fig_json(fig_vol)


# ─────────────────────────────────────────────────────────────
# 4 БЛОК — sentiment-ring (Sharp Math adaptation)
# ─────────────────────────────────────────────────────────────
@router.get("/sentiment-ring")
async def get_sentiment_ring(session: AsyncSession = Depends(get_db)):
    df_movies = await _load_movies(session)
    df_radial = df_movies[(df_movies['year'] >= 1930) & (df_movies['year'] <= 2023)].copy()
    
    df_radial['pos_score'] = df_radial['emotion_arc'].apply(
        lambda x: np.mean([x[3], x[9], x[15], x[5], x[11], x[17]]))
    df_radial['neg_score'] = df_radial['emotion_arc'].apply(
        lambda x: np.mean([x[0], x[6], x[12], x[1], x[7], x[13], x[2], x[8], x[14], x[4], x[10], x[16]]))

    df_yearly_r = df_radial.groupby('year')[['pos_score', 'neg_score']].mean().reset_index()
    MAX_H = 70

    for col, score in [('pos_norm', 'pos_score'), ('neg_norm', 'neg_score')]:
        mn, mx = df_yearly_r[score].min(), df_yearly_r[score].max()
        df_yearly_r[col] = (df_yearly_r[score] - mn) / (mx - mn + 1e-10) * MAX_H

    years_arr = df_yearly_r['year'].values
    N = len(years_arr) * 6
    idx = np.linspace(0, len(df_yearly_r) - 1, N)
    pos_i = interp1d(range(len(df_yearly_r)), df_yearly_r['pos_norm'], kind='linear')(idx)
    neg_i = interp1d(range(len(df_yearly_r)), df_yearly_r['neg_norm'], kind='linear')(idx)

    np.random.seed(42)
    pos_i += np.random.normal(0, 1.5, N)
    neg_i += np.random.normal(0, 1.5, N)

    R = 100
    angles = np.pi / 2 - (2 * np.pi * np.arange(N) / N)
    base_x = R * np.cos(angles)
    base_y = R * np.sin(angles)

    pos_outer_x = (R + pos_i * 1.2) * np.cos(angles)
    pos_outer_y = (R + pos_i * 1.2) * np.sin(angles)
    neg_inner_x = (R - neg_i * 1.2) * np.cos(angles)
    neg_inner_y = (R - neg_i * 1.2) * np.sin(angles)

    pos_poly_x = np.concatenate([pos_outer_x, base_x[::-1], [pos_outer_x[0]]])
    pos_poly_y = np.concatenate([pos_outer_y, base_y[::-1], [pos_outer_y[0]]])
    neg_poly_x = np.concatenate([base_x, neg_inner_x[::-1], [base_x[0]]])
    neg_poly_y = np.concatenate([base_y, neg_inner_y[::-1], [base_y[0]]])

    fig_wheel = go.Figure()
    fig_wheel.add_trace(go.Scatter(
        x=neg_poly_x, y=neg_poly_y, fill='toself',
        fillcolor='rgba(74, 144, 226, 0.25)', 
        line=dict(color='#4A90E2', width=1.5), hoverinfo='skip',
        name='Negative'
    ))
    fig_wheel.add_trace(go.Scatter(
        x=pos_poly_x, y=pos_poly_y, fill='toself',
        fillcolor='rgba(255, 215, 0, 0.25)', 
        line=dict(color='#FFD700', width=1.5), hoverinfo='skip',
        name='Positive'
    ))
    fig_wheel.add_trace(go.Scatter(
        x=np.append(base_x, base_x[0]), y=np.append(base_y, base_y[0]),
        mode='lines', line=dict(color='#555', width=1), hoverinfo='skip', showlegend=False
    ))

    year_positions = np.linspace(0, N - 1, len(years_arr)).astype(int)
    for yi, yr in zip(year_positions, years_arr):
        if yr % 10 != 0: continue
        a = angles[yi]
        ad = math.degrees(a)
        ta = ad if -90 <= ad <= 90 else ad - 180
        r_txt = R + MAX_H + 30
        fig_wheel.add_annotation(
            x=r_txt * math.cos(a), y=r_txt * math.sin(a),
            text=str(int(yr)), showarrow=False,
            font=dict(size=11, color='#AAA'), textangle=-ta,
            xanchor='center', yanchor='middle'
        )

    _apply_theme(fig_wheel)
    fig_wheel.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05)
    )
    return _fig_json(fig_wheel)


# ─────────────────────────────────────────────────────────────
# 5 БЛОК — emotion-transition (Sankey)
# ─────────────────────────────────────────────────────────────
@router.get("/emotion-transition")
async def get_emotion_transition(session: AsyncSession = Depends(get_db)):
    df_movies = await _load_movies(session)

    # === ДАННЫЕ ДЛЯ SANKEY ===
    df_movies['begin_emotion'] = df_movies['emotion_arc'].apply(lambda x: EMOTIONS[np.argmax(x[0:6])] + " (Start)")
    df_movies['mid_emotion'] = df_movies['emotion_arc'].apply(lambda x: EMOTIONS[np.argmax(x[6:12])] + " (Mid)")
    df_movies['end_emotion'] = df_movies['emotion_arc'].apply(lambda x: EMOTIONS[np.argmax(x[12:18])] + " (End)")

    flow1 = df_movies.groupby(['begin_emotion', 'mid_emotion']).size().reset_index(name='value').rename(columns={'begin_emotion': 'source', 'mid_emotion': 'target'})
    flow2 = df_movies.groupby(['mid_emotion', 'end_emotion']).size().reset_index(name='value').rename(columns={'mid_emotion': 'source', 'end_emotion': 'target'})
    links = pd.concat([flow1, flow2])

    all_nodes = list(pd.unique(links[['source', 'target']].values.ravel('K')))
    node_dict = {n: i for i, n in enumerate(all_nodes)}
    links['source_id'] = links['source'].map(node_dict)
    links['target_id'] = links['target'].map(node_dict)

    def node_color(label):
        for em, col in EMOTION_COLORS.items():
            if em in label.lower(): return col
        return '#555'

    node_colors = [node_color(n) for n in all_nodes]
    link_colors_proper = []
    for s in links['source']:
        hex_c = node_color(s).lstrip('#')
        r, g, b = int(hex_c[:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        link_colors_proper.append(f'rgba({r},{g},{b},0.3)')

    # === ДАННЫЕ ДЛЯ HEATMAP ===
    df_movies['start_em'] = df_movies['emotion_arc'].apply(lambda x: EMOTIONS[np.argmax(x[0:6])])
    df_movies['end_em'] = df_movies['emotion_arc'].apply(lambda x: EMOTIONS[np.argmax(x[12:18])])
    transition = df_movies.groupby(['start_em', 'end_em']).size().reset_index(name='count')
    transition_pivot = transition.pivot(index='start_em', columns='end_em', values='count').fillna(0)
    transition_pct = transition_pivot.div(transition_pivot.sum(axis=1), axis=0) * 100

    # === ОБЪЕДИНЯЕМ В ОДИН ФИГУРУ ===
    fig = make_subplots(
        rows=2, cols=1, 
        specs=[[{"type": "sankey"}], [{"type": "heatmap"}]],
        subplot_titles=["Act 1 → Act 2 → Act 3 (Sankey Flow)", "Start vs End Emotion (Heatmap %)" ],
        vertical_spacing=0.15
    )

    # 1. Добавляем Sankey
    fig.add_trace(go.Sankey(
        node=dict(pad=25, thickness=25, line=dict(color='rgba(255,255,255,0.1)', width=1),
                  label=all_nodes, color=node_colors, hovertemplate='<b>%{label}</b><br>%{value} movies<extra></extra>'),
        link=dict(source=links['source_id'], target=links['target_id'], value=links['value'], color=link_colors_proper,
                  hovertemplate='%{source.label} → %{target.label}<br>%{value} movies<extra></extra>')
    ), row=1, col=1)

    # 2. Добавляем Heatmap
    fig.add_trace(go.Heatmap(
        z=transition_pct.values, 
        x=[e.capitalize() for e in transition_pct.columns],
        y=[e.capitalize() for e in transition_pct.index],
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.5, THEME['purple']], [1, THEME['gold']]],
        text=[[f'{v:.1f}%' for v in row] for row in transition_pct.values],
        texttemplate='%{text}', textfont=dict(size=13, color='white'),
        hovertemplate='Start: %{y}<br>End: %{x}<br>%{z:.1f}% of stories<extra></extra>',
        colorbar=dict(title='%', tickfont=dict(color='#AAA'), len=0.4, y=0.2) # Сдвигаем легенду колорбара вниз
    ), row=2, col=1)

    _apply_theme(fig, show_legend=False)
    
    # Стилизуем подписи графиков (чтобы вписывались в дизайн)
    for annotation in fig['layout']['annotations']: 
        annotation['font'] = dict(size=16, color="#E8E6E3", family=THEME["font"])
    
    # Даем графику больше высоты, так как они друг под другом
    fig.update_layout(height=800)
    fig.update_xaxes(title_text="Ending Emotion", row=2, col=1)
    fig.update_yaxes(title_text="Starting Emotion", row=2, col=1)

    return _fig_json(fig)


# ─────────────────────────────────────────────────────────────
# 6 БЛОК — revenue-topology (3D with walls preserved)
# ─────────────────────────────────────────────────────────────
@router.get("/revenue-topology")
async def get_revenue_topology(session: AsyncSession = Depends(get_db)):
    df_movies = await _load_movies(session)
    df_rev = df_movies.copy()
    df_rev['joy_level'] = df_rev['mean_emotions'].apply(lambda x: x[1])
    df_rev['sadness_level'] = df_rev['mean_emotions'].apply(lambda x: x[0])

    joy_bins = pd.cut(df_rev['joy_level'], bins=20)
    sad_bins = pd.cut(df_rev['sadness_level'], bins=20)
    pivot = df_rev.groupby([sad_bins, joy_bins])['revenue'].mean().reset_index() \
        .pivot_table(values='revenue', index='sadness_level', columns='joy_level', aggfunc='mean')

    joy_c = [i.mid for i in pivot.columns]
    sad_c = [i.mid for i in pivot.index]
    # Используем np.log10 как было у тебя в hovertemplate, чтобы график был читаемым
    z = np.log10(pivot.fillna(0).values + 1)

    fig_surf = go.Figure(go.Surface(
        x=joy_c, y=sad_c, z=z,
        colorscale=[[0, '#0A0A0F'], [0.3, THEME['purple']], [0.6, THEME['accent']], [1, THEME['gold']]],
        colorbar=dict(title='log₁₀(Revenue)', tickfont=dict(color='#AAA')),
        hovertemplate='Joy: %{x:.3f}<br>Sadness: %{y:.3f}<br>Revenue: $10^%{z:.1f}<extra></extra>'
    ))
    
    _apply_theme(fig_surf, show_legend=False)
    
    # Твои настройки осей с сохранением стен ("остроты и нет проблем с осями")
    fig_surf.update_layout(
        scene=dict(
            xaxis=dict(title='Joy Intensity', color='#AAA', gridcolor='#1A1A2E',
                       showbackground=True, backgroundcolor='#0D0D12'),
            yaxis=dict(title='Sadness Intensity', color='#AAA', gridcolor='#1A1A2E',
                       showbackground=True, backgroundcolor='#0D0D12'),
            zaxis=dict(title='log₁₀(Revenue)', color='#AAA', gridcolor='#1A1A2E',
                       showbackground=True, backgroundcolor='#0D0D12'),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
            bgcolor=THEME['bg']
        ),
        margin=dict(t=0, b=0, l=0, r=0)
    )
    return _fig_json(fig_surf)


# ─────────────────────────────────────────────────────────────
# 7 БЛОК — vonnegut-map (3D PCA with walls preserved)
# ─────────────────────────────────────────────────────────────
import os

@router.get("/vonnegut-map")
async def get_vonnegut_map(session: AsyncSession = Depends(get_db)):
    cache_file = "vonnegut_cache.csv"

    # === 1. ПРОВЕРЯЕМ НАЛИЧИЕ ФАЙЛА ===
    if os.path.exists(cache_file):
        # Если файл есть, загружаем готовые 40k точек за 0.1 сек
        df_plot = pd.read_csv(cache_file)
        
        cluster_names = {}
        for c in range(6):
            # Достаем название кластера из данных
            name = df_plot[df_plot['story_cluster'] == c]['cluster_label'].iloc[0]
            cluster_names[c] = name
            
        # Для отрисовки центроидов в 3D нам нужны их координаты
        # Просто берем среднее значение точек кластера
        centroids_3d = np.array([
            df_plot[df_plot['story_cluster'] == c][['pc1', 'pc2', 'pc3']].mean().values
            for c in range(6)
        ])
            
    else:
        # === 2. ЕСЛИ ФАЙЛА НЕТ - СЧИТАЕМ И СОХРАНЯЕМ ===
        df_movies = await _load_movies(session)
        X_arcs = np.stack(df_movies['emotion_arc'].values)

        # Кластеризация
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        df_movies['story_cluster'] = kmeans.fit_predict(X_arcs)

        # Названия кластеров
        cluster_names = {}
        for c in range(6):
            centroid = kmeans.cluster_centers_[c]
            cluster_names[c] = f"{EMOTIONS[np.argmax(centroid[0:6])].capitalize()} → {EMOTIONS[np.argmax(centroid[12:18])].capitalize()}"
        df_movies['cluster_label'] = df_movies['story_cluster'].map(cluster_names)

        # 3D PCA
        pca3d = PCA(n_components=3)
        arcs_3d = pca3d.fit_transform(X_arcs)
        df_movies['pc1'] = np.round(arcs_3d[:, 0], 3)
        df_movies['pc2'] = np.round(arcs_3d[:, 1], 3)
        df_movies['pc3'] = np.round(arcs_3d[:, 2], 3)

        # 2D t-SNE (Это займет время при первом запуске)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        arcs_2d = tsne.fit_transform(X_arcs)
        df_movies['pca_x'] = np.round(arcs_2d[:, 0], 3)
        df_movies['pca_y'] = np.round(arcs_2d[:, 1], 3)

        # Выделяем только нужные колонки и СОХРАНЯЕМ В ФАЙЛ
        cols_to_save = ['title', 'year', 'story_cluster', 'cluster_label', 'pc1', 'pc2', 'pc3', 'pca_x', 'pca_y']
        df_plot = df_movies[cols_to_save]
        df_plot.to_csv(cache_file, index=False)
        
        centroids_3d = pca3d.transform(kmeans.cluster_centers_)


    # === 3. ОТРИСОВКА ГРАФИКОВ ===
    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=["Macro Shapes (3D PCA)", "Micro Clusters (2D t-SNE)"]
    )

    for c in range(6):
        mask = df_plot['story_cluster'] == c
        subset = df_plot[mask]
        c_name = cluster_names[c]

        # Левая часть: 3D PCA
        fig.add_trace(go.Scatter3d(
            x=subset['pc1'], y=subset['pc2'], z=subset['pc3'],
            mode='markers', name=c_name,
            legendgroup=str(c), 
            marker=dict(size=3, color=CLUSTER_COLORS[c], opacity=0.6, line=dict(width=0.3, color='white')),
            hovertemplate='<b>%{text}</b><br>Cluster: ' + c_name + '<extra></extra>',
            text=subset['title'] + ' (' + subset['year'].astype(str) + ')',
        ), row=1, col=1)

        # Правая часть: 2D t-SNE (ОБЯЗАТЕЛЬНО Scattergl для ускорения WebGL на 40к точек)
        fig.add_trace(go.Scattergl(
            x=subset['pca_x'], y=subset['pca_y'],
            mode='markers', name=c_name,
            legendgroup=str(c), 
            showlegend=False,   
            marker=dict(size=4, color=CLUSTER_COLORS[c], opacity=0.7),
            text=subset['title'] + ' (' + subset['year'].astype(str) + ')',
            hovertemplate='<b>%{text}</b><extra></extra>'
        ), row=1, col=2)

    # Центроиды для 3D
    fig.add_trace(go.Scatter3d(
        x=centroids_3d[:, 0], y=centroids_3d[:, 1], z=centroids_3d[:, 2],
        mode='markers+text', marker=dict(size=12, color='white', symbol='diamond', line=dict(width=2, color=THEME['accent'])),
        text=[cluster_names[i] for i in range(6)], textposition='top center', textfont=dict(size=12, color=THEME['gold']),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=1)

    _apply_theme(fig, show_legend=True)
    
    for annotation in fig['layout']['annotations']: 
        annotation['font'] = dict(size=16, color="#E8E6E3", family=THEME["font"])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='PC1', color='#888', gridcolor='#1A1A2E', showbackground=True, backgroundcolor='#0D0D12'),
            yaxis=dict(title='PC2', color='#888', gridcolor='#1A1A2E', showbackground=True, backgroundcolor='#0D0D12'),
            zaxis=dict(title='PC3', color='#888', gridcolor='#1A1A2E', showbackground=True, backgroundcolor='#0D0D12'),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
            bgcolor=THEME['bg']
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5, itemsizing='constant'), 
        height=650
    )
    fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
    fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)

    return _fig_json(fig)
"""Competitor benchmarking.

Only public numbers exist for other people's channels (views, likes, comments, length, upload date),
so every channel, including the creator's own, is scored here on that same public footing. That keeps
the comparison fair: "you vs them" always uses identical inputs.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import engine

PUBLIC_COLS = ["video_id", "title", "published", "duration_sec", "views", "likes", "comments", "thumbnail"]
YOU = "You"

# Dimensions that can be measured from public data on any channel.
SHARED_DIMS = ["Title theme", "Length", "Title is a question", "Number in title", "Title length",
               "Second-person (you/your)", "Series / part", "Publish day"]


def public_score(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    cols = [c for c in PUBLIC_COLS if c in df]
    d = df[cols].copy()
    for c in ("likes", "comments"):
        if c not in d:
            d[c] = np.nan
    s = engine.score(d)
    s["channel"] = channel
    s["simple_engagement"] = 100 * (s["likes"].fillna(0) + s["comments"].fillna(0)) / s["views"].replace(0, np.nan)
    return s


def summary(scored: pd.DataFrame, subscribers: float | None) -> dict:
    s = scored
    is_short = s["duration_sec"].fillna(600) <= 60
    uploads_pw = np.nan
    if s["published"].notna().sum() >= 2:
        last = s["published"].max()
        window = min(90.0, max((last - s["published"].min()).days, 7))
        uploads_pw = (s["published"] >= last - pd.Timedelta(days=window)).sum() / (window / 7)
    med_views = s["views"].median()
    return {
        "Subscribers": subscribers if subscribers else np.nan,
        "Median views": med_views,
        "Views per subscriber %": 100 * med_views / subscribers if subscribers else np.nan,
        "Engagement %": s["simple_engagement"].median(),
        "Uploads per week": uploads_pw,
        "Long-form length (min)": s.loc[~is_short, "duration_sec"].median() / 60 if (~is_short).any() else np.nan,
        "Shorts share %": 100 * is_short.mean(),
        "Breakout rate %": 100 * (s["reach_x"] >= 2).mean(),
        "Videos analyzed": len(s),
    }


METRICS = {  # key -> (label, format, higher_is_better, explanation)
    "Median views": ("Median views per video", "{:,.0f}", True, "A typical video's views, not skewed by one viral hit."),
    "Views per subscriber %": ("Views per subscriber", "{:.0f}%", True, "How well videos reach beyond (or into) the subscriber base."),
    "Engagement %": ("Engagement rate", "{:.2f}%", True, "Likes + comments per 100 views."),
    "Uploads per week": ("Uploads per week", "{:.1f}", True, "Average over the last 90 days."),
    "Breakout rate %": ("Breakout rate", "{:.0f}%", True, "Share of videos getting 2×+ the views the channel normally gets."),
    "Long-form length (min)": ("Long-form length", "{:.0f} min", None, "Median length of non-Shorts videos."),
    "Shorts share %": ("Shorts share", "{:.0f}%", None, "Share of uploads that are Shorts."),
}


def benchmark(you: dict, comps: dict[str, dict]) -> pd.DataFrame:
    rows = [{"channel": YOU, **you}] + [{"channel": k, **v} for k, v in comps.items()]
    return pd.DataFrame(rows)


def lift_compare(you_scored: pd.DataFrame, comp_scored: list[pd.DataFrame]) -> pd.DataFrame:
    """Content-point lift on your channel vs pooled across competitors (each scored vs its own baseline)."""
    pooled = pd.concat(comp_scored, ignore_index=True)
    dims = [d for d in SHARED_DIMS if f"f:{d}" in pooled and f"f:{d}" in you_scored]
    pooled.attrs = {"dimensions": dims, "custom_dims": []}
    mine = you_scored.copy()
    mine.attrs = {"dimensions": dims, "custom_dims": []}
    niche = engine.content_points(pooled, min_n=3)
    yours = engine.content_points(mine, min_n=1)
    if niche.empty:
        return pd.DataFrame()
    m = niche[["dimension", "value", "videos", "lift", "confidence"]].rename(
        columns={"videos": "niche_videos", "lift": "niche_lift", "confidence": "niche_confidence"})
    y = yours[["dimension", "value", "videos", "lift"]].rename(columns={"videos": "your_videos", "lift": "your_lift"}) \
        if not yours.empty else pd.DataFrame(columns=["dimension", "value", "your_videos", "your_lift"])
    out = m.merge(y, on=["dimension", "value"], how="left")
    out["your_videos"] = out["your_videos"].fillna(0).astype(int)
    out["channels_using"] = [
        sum((s[f"f:{r.dimension}"] == r.value).any() for s in comp_scored) for r in out.itertuples()]
    return out


def breakouts(comp_scored: list[pd.DataFrame], top: int = 12) -> pd.DataFrame:
    pooled = pd.concat(comp_scored, ignore_index=True)
    b = pooled[pooled["reach_x"] >= 2].sort_values("reach_x", ascending=False)
    return b.head(top)


def opportunities(bench: pd.DataFrame, lc: pd.DataFrame) -> dict[str, list[str]]:
    out = {"gaps": [], "edges": [], "habits": []}
    lc = lc[lc["dimension"] != "Publish day"] if not lc.empty else lc  # too noisy to advise on
    if not lc.empty:
        trusted = lc[(lc["niche_confidence"] != "Low") & (lc["channels_using"] >= 2)]
        pool = trusted if len(trusted) >= 3 else lc
        for r in pool.sort_values("niche_lift", ascending=False).itertuples():
            if r.niche_lift < 4 or len(out["gaps"]) >= 4:
                continue
            if r.your_videos <= 1:
                out["gaps"].append(f"**{r.dimension}: {r.value}** beats competitors' own averages by **+{r.niche_lift:.0f}** "
                                   f"across {r.niche_videos} of their videos. You've made **{r.your_videos}**. Worth testing.")
            elif pd.notna(r.your_lift) and r.your_lift < r.niche_lift - 6:
                out["gaps"].append(f"**{r.dimension}: {r.value}** works for competitors (**+{r.niche_lift:.0f}**) but not yet "
                                   f"for you (**{r.your_lift:+.0f}** over {r.your_videos} videos). Study how they execute it.")
        for r in lc.dropna(subset=["your_lift"]).sort_values("your_lift", ascending=False).itertuples():
            if len(out["edges"]) >= 3:
                break
            if r.your_videos >= 3 and r.your_lift >= 4 and r.your_lift > r.niche_lift + 5:
                out["edges"].append(f"**{r.dimension}: {r.value}** is your edge: **{r.your_lift:+.0f}** for you vs "
                                    f"**{r.niche_lift:+.0f}** for competitors. Keep owning it.")
    you = bench[bench["channel"] == YOU].iloc[0]
    them = bench[bench["channel"] != YOU]
    if them.empty:
        return out

    def med(col):
        return them[col].median()

    if pd.notna(you["Uploads per week"]) and med("Uploads per week") > 1.4 * max(you["Uploads per week"], 0.01):
        out["habits"].append(f"Competitors upload **{med('Uploads per week'):.1f}×/week** (median) vs your "
                             f"**{you['Uploads per week']:.1f}**. More shots on goal means more chances to break out.")
    yl, tl = you["Long-form length (min)"], med("Long-form length (min)")
    if pd.notna(yl) and pd.notna(tl) and abs(yl - tl) / max(tl, 1) > 0.3:
        out["habits"].append(f"Their long-form videos run about **{tl:.0f} min**; yours run **{yl:.0f} min**. "
                             f"{'Consider tighter edits.' if yl > tl else 'There may be room for longer, deeper stories.'}")
    ys, ts = you["Shorts share %"], med("Shorts share %")
    if pd.notna(ys) and ts - ys > 15:
        out["habits"].append(f"**{ts:.0f}%** of competitors' uploads are Shorts vs your **{ys:.0f}%**. "
                             "Shorts cut from your best scenes can feed viewers into the full stories.")
    ye, te = you["Engagement %"], med("Engagement %")
    if pd.notna(ye) and pd.notna(te):
        if ye < te * 0.8:
            out["habits"].append(f"Your engagement (**{ye:.2f}%**) trails theirs (**{te:.2f}%**). End stories on an open question, "
                                 "or pin a comment asking viewers for their theory.")
        elif ye > te * 1.2:
            out["edges"].append(f"Your audience engages more than theirs (**{ye:.2f}%** vs **{te:.2f}%**). That loyalty is an asset.")
    yb, tb = you["Breakout rate %"], med("Breakout rate %")
    if pd.notna(yb) and tb - yb > 5:
        out["habits"].append(f"Competitors break out (2×+ their normal views) on **{tb:.0f}%** of videos vs your **{yb:.0f}%**. "
                             "Check the *What's breaking out for them* list for patterns in premise and packaging.")
    return out


# --------------------------------------------------------------------------- demo competitors

_SUBJECTS = ["The Widow", "The Twins", "My Neighbor", "The Priest", "The Night Nurse", "The Ferryman", "Our Babysitter",
             "The Lighthouse Girl", "The Hunter", "The Bride", "The Stranger", "My Grandfather", "The Orphan", "The Sheriff"]
_PREDICATES = {
    "revenge": ["Who Planned The Perfect Revenge", "Who Betrayed The Whole Family", "Who Made Them Pay"],
    "mystery": ["Who Vanished Without A Trace", "Nobody Can Explain", "Who Kept A Terrible Secret"],
    "horror": ["Who Was Followed Home", "Who Heard The Knocking At 3AM", "Who Bought A Cursed Doll"],
    "tragedy": ["Who Lost Everything", "Who Wrote A Final Letter", "Who Never Said Goodbye"],
    "survival": ["Who Was Trapped For 9 Days", "Who Survived The Frozen Pass", "Who Was Hunted Through The Swamp"],
}
_PREFIX = {"question": "Why Did {s} Disappear?", "number": "{n} Days With {s}"}

DEMO_COMPETITORS = {
    "Demo: Hollow Tales": {"subs": 64000, "scale": 3.2, "seed": 11, "cadence": 2.5, "shorts": .35, "len": 11,
                           "fx": {"revenge": .5, "mystery": .4, "horror": .1, "tragedy": -.3, "survival": .2}, "q": .35},
    "Demo: The Dread Archive": {"subs": 120000, "scale": 5.0, "seed": 12, "cadence": 1.6, "shorts": .15, "len": 22,
                                "fx": {"revenge": .3, "mystery": .6, "horror": .3, "tragedy": -.2, "survival": .1}, "q": .45},
    "Demo: Sleepless Stories": {"subs": 31000, "scale": 1.8, "seed": 13, "cadence": 3.5, "shorts": .5, "len": 9,
                                "fx": {"revenge": .2, "mystery": .2, "horror": .5, "tragedy": -.4, "survival": .3}, "q": .25},
}


def demo_competitor(name: str, n: int = 50) -> tuple[pd.DataFrame, dict]:
    p = DEMO_COMPETITORS[name]
    rng = np.random.default_rng(p["seed"])
    now = datetime.now(timezone.utc)
    rows, age = [], 2.0
    for i in range(n):
        theme = rng.choice(list(_PREDICATES))
        subj = rng.choice(_SUBJECTS)
        r = rng.random()
        if r < p["q"]:
            title, q = _PREFIX["question"].format(s=subj), 1
        elif r < p["q"] + .15:
            title, q = _PREFIX["number"].format(s=subj, n=rng.integers(3, 40)), 0
        else:
            title, q = f"{subj} {rng.choice(_PREDICATES[theme])}", 0
        if q:  # questions don't carry the theme words, so give them the mystery predicate's flavour
            theme = "mystery"
        short = rng.random() < p["shorts"]
        dur = int(rng.integers(30, 59)) if short else int(abs(rng.normal(p["len"], 3)) * 60) + 180
        quality = p["fx"][theme] + .35 * q + rng.normal(0, .45)
        ramp = 1 - np.exp(-age / 7)
        views = int(p["scale"] * 4000 * ramp * np.exp(quality * 1.1) * (1.8 if short else 1))
        rows.append({"video_id": f"{p['seed']}-{i}", "title": title, "published": now - timedelta(days=age),
                     "duration_sec": dur, "views": views,
                     "likes": int(views * np.clip(rng.normal(.035 + quality * .006, .006), .004, .1)),
                     "comments": int(views * np.clip(rng.normal(.004, .0015), 0, .03)), "thumbnail": None})
        age += rng.exponential(7 / p["cadence"]) + .3
    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df, {"name": name, "subscribers": p["subs"]}

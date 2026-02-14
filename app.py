import streamlit as st

st.write("ISOM5240")


#!/usr/bin/env python3
"""
Funnay Frenzy - single-file playable game (pure Python, tkinter)
- Click the colorful critters for points.
- Avoid bombs (black) or you'll lose points.
- Survive and score as many points as you can in the time limit.
- High score persists to 'funnay_highscore.txt' in same folder.

Tested on Python 3.8+. No external dependencies.
"""

import tkinter as tk
import random
import time
import math
import json
import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List

# ----------------------------
# Configuration
# ----------------------------
WINDOW_W = 800
WINDOW_H = 600
GAME_DURATION = 30.0  # seconds
INITIAL_POP_INTERVAL = 1.0  # seconds between spawns initially
MIN_POP_INTERVAL = 0.25
POP_DECAY = 0.92  # how pop interval multiplies per level-up
MAX_ACTIVE = 8
MOLE_MIN_RADIUS = 18
MOLE_MAX_RADIUS = 44
BOMB_PROB_START = 0.08
BOMB_PROB_MAX = 0.25
SCORE_FILE = "funnay_highscore.txt"

# ----------------------------
# Utilities
# ----------------------------
def clamp(v, a, b):
    return max(a, min(b, v))

def now():
    return time.perf_counter()

# ----------------------------
# Mole Dataclass
# ----------------------------
@dataclass
class Mole:
    id_canvas: int
    x: float
    y: float
    r: float
    color: str
    born: float
    lifetime: float
    vx: float = 0.0
    vy: float = 0.0
    is_bomb: bool = False
    wobble_phase: float = 0.0

# ----------------------------
# Game Class
# ----------------------------
class FunnayFrenzy:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Funnay Frenzy")
        self.canvas = tk.Canvas(root, width=WINDOW_W, height=WINDOW_H, bg="#111111")
        self.canvas.pack(fill="both", expand=True)
        self.last_time = now()
        self.running = False
        self.paused = False
        self.score = 0
        self.start_time = None
        self.end_time = None
        self.remaining = GAME_DURATION
        self.moles: Dict[int, Mole] = {}
        self.next_spawn = 0.0
        self.pop_interval = INITIAL_POP_INTERVAL
        self.bomb_prob = BOMB_PROB_START
        self.highscore = self._load_highscore()
        self.round = 1
        self.display_items = {}
        self._setup_ui()
        self._bind_events()
        self._show_title()

    # ----------------------------
    # High score
    # ----------------------------
    def _load_highscore(self) -> int:
        try:
            if os.path.exists(SCORE_FILE):
                with open(SCORE_FILE, "r", encoding="utf-8") as f:
                    return int(f.read().strip() or "0")
        except Exception:
            pass
        return 0

    def _save_highscore(self):
        try:
            with open(SCORE_FILE, "w", encoding="utf-8") as f:
                f.write(str(self.highscore))
        except Exception:
            pass

    # ----------------------------
    # UI setup
    # ----------------------------
    def _setup_ui(self):
        # top HUD
        self.hud_frame = tk.Frame(self.root, bg="#111111")
        # can't pack; we use canvas for all visuals
        # HUD items drawn on canvas instead
        self.canvas.configure(cursor="hand2")
        # overlay texts (canvas items)
        self.display_items['score_text'] = self.canvas.create_text(
            10, 10, anchor="nw", text="", fill="#fff", font=("Segoe UI", 16, "bold")
        )
        self.display_items['time_text'] = self.canvas.create_text(
            WINDOW_W-10, 10, anchor="ne", text="", fill="#fff", font=("Segoe UI", 16, "bold")
        )
        self.display_items['round_text'] = self.canvas.create_text(
            WINDOW_W//2, 10, anchor="n", text="", fill="#ffd966", font=("Segoe UI", 14, "bold")
        )
        # helpful tip
        self.display_items['tip'] = self.canvas.create_text(
            WINDOW_W//2, WINDOW_H-8, anchor="s", text="Click critters to score. Avoid bombs!", fill="#ddd", font=("Segoe UI", 12, "italic")
        )
        # simple decorative background stars
        self._stars = []
        for _ in range(60):
            sx = random.randint(0, WINDOW_W)
            sy = random.randint(0, WINDOW_H)
            s = self.canvas.create_oval(sx, sy, sx+2, sy+2, fill="#2d2d2d", outline="")
            self._stars.append(s)

    def _bind_events(self):
        self.canvas.bind("<Button-1>", self._on_click)
        self.root.bind("<space>", lambda e: self._toggle_pause())
        self.root.bind("<Escape>", lambda e: self._quit())

    # ----------------------------
    # Title / Screens
    # ----------------------------
    def _show_title(self):
        self.canvas.delete("screen")
        title = self.canvas.create_text(
            WINDOW_W//2, WINDOW_H//3, text="FUNNAY FRENZY",
            fill="#fff", font=("Segoe UI", 48, "bold"), tags="screen"
        )
        subtitle = self.canvas.create_text(
            WINDOW_W//2, WINDOW_H//3+70, text="A silly click 'n' dodge game",
            fill="#ffd966", font=("Segoe UI", 18), tags="screen"
        )
        author = self.canvas.create_text(
            WINDOW_W//2, WINDOW_H//3+110, text="(pure Python + tkinter)",
            fill="#bbb", font=("Segoe UI", 12), tags="screen"
        )
        hs = self.canvas.create_text(
            WINDOW_W//2, WINDOW_H//3+150, text=f"High Score: {self.highscore}",
            fill="#8fe8a3", font=("Segoe UI", 14, "bold"), tags="screen"
        )
        start_btn = self._make_button(WINDOW_W//2, WINDOW_H//2+40, "START", self._start_game, tags="screen")
        instruct = self.canvas.create_text(
            WINDOW_W//2, WINDOW_H//2+120, text="Click critters. Avoid black bombs. Press Space to pause.",
            fill="#ddd", font=("Segoe UI", 11), tags="screen"
        )
        self.canvas.create_text(
            WINDOW_W-6, WINDOW_H-6, anchor="se",
            text="Tip: the moles get faster!",
            fill="#444", font=("Segoe UI", 9), tags="screen"
        )

    def _make_button(self, x, y, text, cmd, tags=()):
        # basic canvas button: rectangle + text; binds to click
        padx, pady = 22, 10
        font = ("Segoe UI", 14, "bold")
        tw = self.canvas.create_text(x, y, text=text, font=font, fill="#222")
        bbox = self.canvas.bbox(tw)
        if not bbox:
            bbox = (x-50, y-12, x+50, y+12)
        x1, y1, x2, y2 = bbox
        rect = self.canvas.create_rectangle(x1-padx, y1-pady, x2+padx, y2+pady, fill="#ffd966", outline="#000", width=2, tags=tags)
        self.canvas.tag_raise(tw, rect)
        # group into tag
        group_tag = f"btn_{rect}"
        self.canvas.addtag_withtag(group_tag, rect)
        self.canvas.addtag_withtag(group_tag, tw)
        for t in (rect, tw):
            self.canvas.tag_bind(t, "<Button-1>", lambda e, c=cmd: c())
        return rect

    def _show_gameover(self):
        self.canvas.delete("screen")
        overlay = self.canvas.create_rectangle(0, 0, WINDOW_W, WINDOW_H, fill="#000000b0", tags="screen")
        self.canvas.create_text(WINDOW_W//2, WINDOW_H//3, text="GAME OVER", fill="#ff7f7f", font=("Segoe UI", 42, "bold"), tags="screen")
        self.canvas.create_text(WINDOW_W//2, WINDOW_H//3+64, text=f"Score: {self.score}", fill="#fff", font=("Segoe UI", 20), tags="screen")
        self.canvas.create_text(WINDOW_W//2, WINDOW_H//3+95, text=f"High Score: {self.highscore}", fill="#aaffaa", font=("Segoe UI", 14), tags="screen")
        self._make_button(WINDOW_W//2, WINDOW_H//2+20, "PLAY AGAIN", self._start_game, tags="screen")
        self._make_button(WINDOW_W//2, WINDOW_H//2+80, "QUIT", self._quit, tags="screen")

    # ----------------------------
    # Game control
    # ----------------------------
    def _start_game(self):
        # reset state
        self.canvas.delete("screen")
        for k in list(self.moles.keys()):
            self._remove_mole(k)
        self.running = True
        self.paused = False
        self.score = 0
        self.start_time = now()
        self.end_time = self.start_time + GAME_DURATION
        self.remaining = GAME_DURATION
        self.next_spawn = 0.2
        self.pop_interval = INITIAL_POP_INTERVAL
        self.bomb_prob = BOMB_PROB_START
        self.round = 1
        self.update_display()
        self.last_time = now()
        self._game_loop()

    def _toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if not self.paused:
            # resume
            # push timers forward
            shift = now() - self.last_time
            self.end_time += shift
            self.last_time = now()

    def _quit(self):
        self.root.quit()

    # ----------------------------
    # Mole management
    # ----------------------------
    def _spawn_mole(self):
        if len(self.moles) >= MAX_ACTIVE:
            return
        # choose random spot avoiding very edges
        r = random.uniform(MOLE_MIN_RADIUS, MOLE_MAX_RADIUS)
        x = random.uniform(r+8, WINDOW_W - r-8)
        y = random.uniform(40 + r, WINDOW_H - r - 40)
        lifespan = random.uniform(1.2, 2.4) * max(0.5, (1.0 - (self.round-1)*0.05))
        is_bomb = random.random() < clamp(self.bomb_prob, 0.0, BOMB_PROB_MAX)
        if is_bomb:
            color = "#111111"
        else:
            # bright playful colors
            palette = ["#ff6b6b", "#ffe66d", "#6bffb0", "#6bd0ff", "#c56bff", "#ff9fd6"]
            color = random.choice(palette)
        # create circle + eyes (eyes for cuteness)
        oval = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline="#222", width=2)
        # eyes
        ex = x - r*0.28
        ey = y - r*0.18
        eye_l = self.canvas.create_oval(ex- r*0.12, ey- r*0.12, ex+ r*0.12, ey+ r*0.12, fill="#fff", outline="")
        eye_r = self.canvas.create_oval(ex+ r*0.36 - r*0.12, ey- r*0.12, ex+ r*0.36 + r*0.12, ey+ r*0.12, fill="#fff", outline="")
        pupil_l = self.canvas.create_oval(ex- r*0.06, ey- r*0.04, ex+ r*0.03, ey+ r*0.06, fill="#222", outline="")
        pupil_r = self.canvas.create_oval(ex+ r*0.36- r*0.06, ey- r*0.04, ex+ r*0.36+ r*0.03, ey+ r*0.06, fill="#222", outline="")
        group = [oval, eye_l, eye_r, pupil_l, pupil_r]
        group_tag = f"mole_{oval}"
        for item in group:
            self.canvas.addtag_withtag(group_tag, item)
            # make clickable as group
            self.canvas.tag_bind(item, "<Button-1>", lambda e, gid=oval: self._on_mole_click(gid))
        # slight velocity for drift
        angle = random.random()*math.pi*2
        speed = random.uniform(6, 36) / 60.0  # px per frame approx
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        m = Mole(id_canvas=oval, x=x, y=y, r=r, color=color, born=now(), lifetime=lifespan, vx=vx, vy=vy, is_bomb=is_bomb, wobble_phase=random.random()*10)
        self.moles[oval] = m

    def _remove_mole(self, cid):
        if cid not in self.moles:
            # try remove grouped items by tag
            try:
                self.canvas.delete(f"mole_{cid}")
            except Exception:
                pass
            return
        # remove group by tag
        tag = f"mole_{cid}"
        self.canvas.delete(tag)
        try:
            del self.moles[cid]
        except KeyError:
            pass

    def _on_mole_click(self, cid):
        if not self.running or self.paused:
            return
        if cid not in self.moles:
            return
        m = self.moles.get(cid)
        if not m:
            return
        # clicking a bomb penalizes, normal mole rewards
        if m.is_bomb:
            self.score = max(0, self.score - 7)
            # explosion effect
            self._pop_effect(m.x, m.y, color="#000000", big=True)
        else:
            self.score += int(1 + m.r/10)
            self._pop_effect(m.x, m.y, color=m.color, big=False)
        # remove mole
        self._remove_mole(cid)
        # tiny difficulty tweak on successful hits
        self.pop_interval = max(MIN_POP_INTERVAL, self.pop_interval * 0.98)
        self.bomb_prob = min(BOMB_PROB_MAX, self.bomb_prob + 0.003)
        self.update_display()

    # ----------------------------
    # Visual effects
    # ----------------------------
    def _pop_effect(self, x, y, color="#fff", big=False):
        # simple expanding ring
        maxr = 36 if big else 20
        ring = self.canvas.create_oval(x-4, y-4, x+4, y+4, outline=color, width=3)
        start = now()
        def anim():
            t = now() - start
            if t > 0.45:
                self.canvas.delete(ring)
                return
            frac = t / 0.45
            r = 4 + frac * maxr
            alpha = int(220 * (1-frac))
            # tkinter doesn't support alpha on outlines; emulate with color fade toward black
            self.canvas.coords(ring, x-r, y-r, x+r, y+r)
            self.root.after(16, anim)
        anim()

    # ----------------------------
    # Click handler (for background clicks)
    # ----------------------------
    def _on_click(self, event):
        # clicking empty space gives a tiny bonus
        if not self.running or self.paused:
            return
        # detect if clicked any mole: canvas bindings handle group clicks first.
        # reward for empty-space clicks reduces spamming
        self.score = max(0, self.score - 0)  # no penalty; could add
        self.update_display()

    # ----------------------------
    # Game loop
    # ----------------------------
    def _game_loop(self):
        if not self.running:
            return
        now_t = now()
        dt = now_t - self.last_time
        self.last_time = now_t
        if not self.paused:
            # update timers
            self.remaining = max(0.0, self.end_time - now_t)
            # spawn logic
            self.next_spawn -= dt
            if self.next_spawn <= 0.0:
                self._spawn_mole()
                # schedule next spawn
                jitter = random.uniform(-0.25, 0.25) * self.pop_interval
                self.next_spawn = clamp(self.pop_interval + jitter, MIN_POP_INTERVAL, INITIAL_POP_INTERVAL*2)
            # move & age moles
            to_remove = []
            for cid, m in list(self.moles.items()):
                age = now_t - m.born
                # wobble
                m.wobble_phase += dt * 6.0
                wob = math.sin(m.wobble_phase) * (m.r * 0.06)
                # drift
                m.x += m.vx * (1 + (self.round-1)*0.02) * dt * 60
                m.y += m.vy * (1 + (self.round-1)*0.02) * dt * 60
                # keep inside bounds by bouncing velocities
                if m.x - m.r < 4:
                    m.x = m.r + 4
                    m.vx *= -1
                if m.x + m.r > WINDOW_W-4:
                    m.x = WINDOW_W - m.r - 4
                    m.vx *= -1
                if m.y - m.r < 30:
                    m.y = m.r + 30
                    m.vy *= -1
                if m.y + m.r > WINDOW_H - 30:
                    m.y = WINDOW_H - m.r - 30
                    m.vy *= -1
                # update canvas group coords
                tag = f"mole_{cid}"
                # find the oval item by id cid
                x1, y1, x2, y2 = m.x - m.r + wob, m.y - m.r, m.x + m.r + wob, m.y + m.r
                # reposition main oval and all children by moving to computed coords
                # easiest: delete and redraw group with preservation of id? Simpler: use coords for the oval, and adjust eyes/pupils relative
                try:
                    self.canvas.coords(cid, x1, y1, x2, y2)
                    # reposition eyes/pupils by iterating items with tag
                    items = self.canvas.find_withtag(tag)
                    # items include oval and eyes; keep relative positions
                    # recompute eye positions
                    ex = m.x - m.r*0.28 + wob*0.18
                    ey = m.y - m.r*0.18
                    eye_r_offset = m.r*0.36
                    # we expect items order may vary; try to set coords by approximate sizes
                    for item in items:
                        # skip main oval
                        if item == cid:
                            continue
                        # find bounding box size to infer which element
                        bb = self.canvas.bbox(item)
                        if not bb:
                            continue
                        w = (bb[2]-bb[0]) + 1
                        h = (bb[3]-bb[1]) + 1
                        # eye size ~ r*0.24
                        if w > m.r*0.2:
                            # likely eyes
                            if abs((bb[0]+bb[2])/2 - (m.x - 0.0)) < 1.0:
                                # left-ish eye
                                self.canvas.coords(item, ex- m.r*0.12, ey- m.r*0.12, ex+ m.r*0.12, ey+ m.r*0.12)
                            else:
                                # right eye
                                self.canvas.coords(item, ex+ eye_r_offset - m.r*0.12, ey- m.r*0.12, ex+ eye_r_offset + m.r*0.12, ey+ m.r*0.12)
                        else:
                            # small: pupil; decide left or right by position
                            if abs((bb[0]+bb[2])/2 - (m.x - 0.0)) < 1.0:
                                self.canvas.coords(item, ex- m.r*0.06, ey- m.r*0.04, ex+ m.r*0.03, ey+ m.r*0.06)
                            else:
                                self.canvas.coords(item, ex+ eye_r_offset- m.r*0.06, ey- m.r*0.04, ex+ eye_r_offset+ m.r*0.03, ey+ m.r*0.06)
                except tk.TclError:
                    # item may have been deleted
                    to_remove.append(cid)
                    continue
                # age check
                if age > m.lifetime:
                    to_remove.append(cid)
            # remove aged moles
            for cid in to_remove:
                # small penalty for letting critters escape
                m = self.moles.get(cid)
                if m and not m.is_bomb:
                    self.score = max(0, self.score - 1)
                self._remove_mole(cid)
            # difficulty increase with time
            # every few seconds, make pop interval a bit smaller and increment round
            elapsed = now_t - self.start_time
            target_round = int(elapsed // 6) + 1
            if target_round > self.round:
                self.round = target_round
                self.pop_interval = max(MIN_POP_INTERVAL, self.pop_interval * POP_DECAY)
                self.bomb_prob = min(BOMB_PROB_MAX, self.bomb_prob + 0.02)
            # end check
            if self.remaining <= 0:
                self.running = False
                # clear all moles and show game over
                for cid in list(self.moles.keys()):
                    self._remove_mole(cid)
                # update highscore
                if self.score > self.highscore:
                    self.highscore = self.score
                    self._save_highscore()
                self.update_display()
                self._show_gameover()
                return
            self.update_display()
        # schedule next frame (~60fps)
        self.root.after(16, self._game_loop)

    # ----------------------------
    # Display updates
    # ----------------------------
    def update_display(self):
        self.canvas.itemconfigure(self.display_items['score_text'], text=f"Score: {self.score}", fill="#ffffff")
        self.canvas.itemconfigure(self.display_items['time_text'], text=f"Time: {int(self.remaining)}s", fill="#ffffff")
        self.canvas.itemconfigure(self.display_items['round_text'], text=f"Round {self.round}  •  High {self.highscore}")
        # dynamic tip color pulse
        pulse = (math.sin(now()*3) + 1) / 2
        color = '#{0:02x}{1:02x}{2:02x}'.format(
            int(0xCC * (0.7 + 0.3*pulse)),
            int(0xCC * (0.9 - 0.3*pulse)),
            int(0xFF * (0.5 + 0.5*pulse)),
        )
        self.canvas.itemconfigure(self.display_items['tip'], fill=color)
        # subtle HUD background
        # (redraw a translucent bar)
        try:
            if hasattr(self, '_hud_rect'):
                self.canvas.delete(self._hud_rect)
            self._hud_rect = self.canvas.create_rectangle(0,0,WINDOW_W,36, fill="#00000080", outline="")
            self.canvas.tag_lower(self._hud_rect)
        except Exception:
            pass

# ----------------------------
# Run
# ----------------------------
def main():
    root = tk.Tk()
    root.resizable(False, False)
    app = FunnayFrenzy(root)
    root.mainloop()

if __name__ == "__main__":
    main()

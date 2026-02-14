import streamlit as st
st.write("ISOM5240")

# Simple Streamlit playable game — turn-based and fully compatible with Streamlit's rerun model.

import random
import time

# --- Config ---
PLAY_W = 24
PLAY_H = 12
PLAYER_CHAR = "P"
HAZARD_CHAR = "X"
GOOD_CHAR = "o"
EMPTY_CHAR = " "

# --- Session-state initialization ---
if "score" not in st.session_state:
    st.session_state.score = 0
if "lives" not in st.session_state:
    st.session_state.lives = 3
if "player_x" not in st.session_state:
    st.session_state.player_x = PLAY_W // 2
if "objects" not in st.session_state:
    st.session_state.objects = []  # list of dicts: {"x":int,"y":int,"ch":str}
if "running" not in st.session_state:
    st.session_state.running = True
if "auto_steps" not in st.session_state:
    st.session_state.auto_steps = 0

# --- Helpers ---
def spawn_object():
    if random.random() < 0.18:
        ch = GOOD_CHAR
    else:
        ch = HAZARD_CHAR
    x = random.randint(1, PLAY_W)
    st.session_state.objects.append({"x": x, "y": 1, "ch": ch})

def step_game(times=1):
    if not st.session_state.running:
        return
    for _ in range(times):
        # spawn sometimes
        if random.random() < 0.6:
            spawn_object()
        # move objects
        new_objs = []
        for obj in st.session_state.objects:
            obj["y"] += 1
            if obj["y"] > PLAY_H:
                # reached bottom; check collision with player
                if obj["x"] == st.session_state.player_x:
                    if obj["ch"] == HAZARD_CHAR:
                        st.session_state.lives -= 1
                    else:
                        st.session_state.score += 3
                # else it falls out
            else:
                new_objs.append(obj)
        st.session_state.objects = new_objs
        if st.session_state.lives <= 0:
            st.session_state.running = False
            break

def render_playfield_text():
    grid = [[EMPTY_CHAR for _ in range(PLAY_W)] for _ in range(PLAY_H)]
    for obj in st.session_state.objects:
        x = obj["x"] - 1
        y = obj["y"] - 1
        if 0 <= x < PLAY_W and 0 <= y < PLAY_H:
            grid[y][x] = obj["ch"]
    # place player
    py = PLAY_H - 1
    px = st.session_state.player_x - 1
    if 0 <= px < PLAY_W:
        grid[py][px] = PLAYER_CHAR
    lines = []
    lines.append("+" + "-" * PLAY_W + "+")
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * PLAY_W + "+")
    return "```\n" + "\n".join(lines) + "\n```"

# --- Layout ---
st.title("Funnay Dodge — Streamlit Edition")
col_status, col_play = st.columns([1, 3])

with col_status:
    st.subheader("Status")
    st.write(f"Score: **{st.session_state.score}**")
    st.write(f"Lives: **{st.session_state.lives}**")
    st.write("Controls:")
    st.write("- Move: Left / Right buttons")
    st.write("- Step: advance one tick")
    st.write("- Auto Play: run a few automatic steps")

    if st.button("Reset Game"):
        st.session_state.score = 0
        st.session_state.lives = 3
        st.session_state.player_x = PLAY_W // 2
        st.session_state.objects = []
        st.session_state.running = True
        st.session_state.auto_steps = 0

    if st.session_state.running:
        if st.button("Step"):
            step_game(times=1)
    else:
        st.error("Game Over — press Reset Game to play again.")

    if st.button("Auto Play (10 steps)"):
        st.session_state.auto_steps = 10

with col_play:
    # Movement controls above playfield
    mc1, mc2, mc3 = st.columns([1,1,1])
    if mc1.button("⭠ Move Left"):
        st.session_state.player_x = max(1, st.session_state.player_x - 1)
    if mc3.button("Move Right ⭢"):
        st.session_state.player_x = min(PLAY_W, st.session_state.player_x + 1)
    # Display playfield
    st.markdown(render_playfield_text(), unsafe_allow_html=True)

# Process auto steps if requested (non-blocking, single-step per rerun)
if st.session_state.auto_steps > 0 and st.session_state.running:
    # take one automatic step, decrement counter, then rerun will re-render
    # simple heuristic: move player towards nearest good if present
    goods = [o for o in st.session_state.objects if o["ch"] == GOOD_CHAR]
    hazards = [o for o in st.session_state.objects if o["ch"] == HAZARD_CHAR]
    move = 0
    if goods:
        g = min(goods, key=lambda o: o["y"])
        if g["x"] < st.session_state.player_x:
            move = -1
        elif g["x"] > st.session_state.player_x:
            move = 1
    elif hazards:
        h = min(hazards, key=lambda o: o["y"])
        # move away if close
        if abs(h["x"] - st.session_state.player_x) <= 2:
            move = -1 if h["x"] > st.session_state.player_x else 1
    if move < 0:
        st.session_state.player_x = max(1, st.session_state.player_x - 1)
    elif move > 0:
        st.session_state.player_x = min(PLAY_W, st.session_state.player_x + 1)

    step_game(times=1)
    st.session_state.auto_steps -= 1
    # trigger a rerun so the user sees progress (Streamlit reruns on any state change)
    st.experimental_rerun()

# Footer
st.caption("Made with Streamlit. Running inside the browser — use the buttons to play.")

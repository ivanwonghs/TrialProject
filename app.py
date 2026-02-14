import streamlit as st
st.write("ISOM5240")

# Funnay Dodge - Streamlit edition (single file)
# Playable turn-based / simple animated game inside Streamlit.
# - Use the Left / Right buttons to move the player (P)
# - Press "Step" to advance one tick, or "Auto Play" to run a short automated run.
# - Catch goodies 'o' to score. Avoid hazards 'X' (lose a life if they reach you).
# Works fully within Streamlit (no terminal required).

import random
import time

# --- Game parameters ---
PLAY_W = 24
PLAY_H = 12
PLAYER_CHAR = "P"
HAZARD_CHAR = "X"
GOOD_CHAR = "o"
EMPTY_CHAR = " "

# initialize session state
if "score" not in st.session_state:
    st.session_state.score = 0
if "lives" not in st.session_state:
    st.session_state.lives = 3
if "player_x" not in st.session_state:
    st.session_state.player_x = PLAY_W // 2
if "objects" not in st.session_state:
    st.session_state.objects = []  # list of dicts: x,y,ch
if "running" not in st.session_state:
    st.session_state.running = True
if "auto" not in st.session_state:
    st.session_state.auto = False
if "tick_count" not in st.session_state:
    st.session_state.tick_count = 0

st.sidebar.title("Controls")
if st.sidebar.button("Reset Game"):
    st.session_state.score = 0
    st.session_state.lives = 3
    st.session_state.player_x = PLAY_W // 2
    st.session_state.objects = []
    st.session_state.running = True
    st.session_state.tick_count = 0

col1, col2, col3 = st.sidebar.columns(3)
if col1.button("⭠ Left"):
    st.session_state.player_x = max(1, st.session_state.player_x - 1)
if col3.button("Right ⭢"):
    st.session_state.player_x = min(PLAY_W, st.session_state.player_x + 1)

if col2.button("Step"):
    # advance one tick
    st.session_state.tick_count += 1

auto_toggle = st.sidebar.checkbox("Auto Play (short)", value=False)
st.session_state.auto = auto_toggle

st.sidebar.markdown("Rules: Catch 'o' to gain +3. If 'X' reaches you, you lose 1 life.")
st.sidebar.markdown("Goal: Survive and score as much as you can!")

# spawn logic
def spawn_object():
    if random.random() < 0.18:
        ch = GOOD_CHAR
    else:
        ch = HAZARD_CHAR
    x = random.randint(1, PLAY_W)
    st.session_state.objects.append({"x": x, "y": 1, "ch": ch})

def step_game():
    if not st.session_state.running:
        return
    st.session_state.tick_count += 1
    # spawn occasionally
    if random.random() < 0.6:
        spawn_object()
    # move objects down
    new_objs = []
    for obj in st.session_state.objects:
        obj["y"] += 1
        if obj["y"] > PLAY_H:
            # reached bottom: check collision with player
            if abs(obj["x"] - st.session_state.player_x) <= 0:
                if obj["ch"] == HAZARD_CHAR:
                    st.session_state.lives -= 1
                else:
                    st.session_state.score += 3
            # otherwise it falls past
        else:
            new_objs.append(obj)
    st.session_state.objects = new_objs
    if st.session_state.lives <= 0:
        st.session_state.running = False

# Auto Play loop (limited steps)
if st.session_state.auto and st.session_state.running:
    # run a short auto-loop but keep Streamlit responsive
    steps = 12
    placeholder = st.empty()
    for i in range(steps):
        # small AI: move toward nearest good or away from nearest hazard
        objs = list(st.session_state.objects)
        # choose target
        target_dx = 0
        nearest_good = None
        nearest_hazard = None
        for o in objs:
            if o["ch"] == GOOD_CHAR:
                if nearest_good is None or o["y"] < nearest_good["y"]:
                    nearest_good = o
            else:
                if nearest_hazard is None or o["y"] < nearest_hazard["y"]:
                    nearest_hazard = o
        if nearest_good:
            if nearest_good["x"] < st.session_state.player_x:
                target_dx = -1
            elif nearest_good["x"] > st.session_state.player_x:
                target_dx = 1
        elif nearest_hazard:
            # try to move away from hazard's x if it's close
            if abs(nearest_hazard["x"] - st.session_state.player_x) <= 2:
                if nearest_hazard["x"] <= st.session_state.player_x:
                    target_dx = 1
                else:
                    target_dx = -1
        # apply movement
        st.session_state.player_x = max(1, min(PLAY_W, st.session_state.player_x + target_dx))
        step_game()
        # render interim UI
        render_box = placeholder.container()
        with render_box:
            st.markdown(render_playfield(), unsafe_allow_html=True)
            st.write(f"Score: {st.session_state.score}   Lives: {st.session_state.lives}   Tick: {st.session_state.tick_count}")
        time.sleep(0.12)
        if not st.session_state.running:
            break
    placeholder.empty()
else:
    # if not auto, but Step clicked in sidebar, advance one tick
    if st.session_state.tick_count > 0:
        # ensure we only process one step per click (decrement tick_count)
        # Actually tick_count used to count steps; we process one step here per request.
        step_game()
        # reduce tick_count so consecutive Step clicks work
        # (we don't strictly need to decrement; it's just a click counter)
        st.session_state.tick_count = 0

# Render playfield using markdown with monospace and simple box
def render_playfield():
    grid = [[EMPTY_CHAR for _ in range(PLAY_W)] for _ in range(PLAY_H)]
    for obj in st.session_state.objects:
        x = int(obj["x"] - 1)
        y = int(obj["y"] - 1)
        if 0 <= x < PLAY_W and 0 <= y < PLAY_H:
            grid[y][x] = obj["ch"]
    # player row
    py = PLAY_H - 1
    px = int(st.session_state.player_x - 1)
    if 0 <= px < PLAY_W:
        grid[py][px] = PLAYER_CHAR
    # build string
    lines = []
    lines.append("+" + "-" * PLAY_W + "+")
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * PLAY_W + "+")
    return "```\n" + "\n".join(lines) + "\n```"

# main display
st.header("Funnay Dodge — Streamlit Edition")
st.markdown(render_playfield(), unsafe_allow_html=True)
st.write(f"Score: {st.session_state.score}   Lives: {st.session_state.lives}   Tick: {st.session_state.tick_count}")

# Movement buttons in main area
c1, c2, c3 = st.columns([1,1,1])
if c1.button("⭠ Move Left"):
    st.session_state.player_x = max(1, st.session_state.player_x - 1)
if c3.button("Move Right ⭢"):
    st.session_state.player_x = min(PLAY_W, st.session_state.player_x + 1)
if c2.button("Step"):
    step_game()

if not st.session_state.running:
    st.error("Game Over! Press Reset Game to play again.")
else:
    st.info("Use Step or Auto Play. Controls in the sidebar.")

# small footer
st.caption("Made with pure Python + Streamlit. Enjoy!")


#!/usr/bin/env python3
"""
Funnay Dodge - single-file terminal game (pure Python, no external libs)

Controls:
 - a / A / left-arrow  : move left
 - d / D / right-arrow : move right
 - s / S               : stay (no movement)
 - p / P               : pause/unpause
 - q / Q               : quit

Goal:
 - Catch goodies 'o' to gain points.
 - Dodge hazards 'X' (lose a life if hit).
 - Survive as long as possible. Difficulty increases over time.

Notes:
 - Works in standard POSIX terminals (Linux / macOS).
 - On Windows, it uses msvcrt for input.
 - No curses dependency; simple drawing with ANSI escape codes.
"""

import sys
import time
import random
import threading

import streamlit as st

st.write("ISOM5240")


# Platform-specific single-key input
IS_WIN = sys.platform.startswith("win")

if IS_WIN:
    import msvcrt

    def get_char_blocking(timeout=None):
        start = time.time()
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                return ch
            if timeout is not None and (time.time() - start) >= timeout:
                return None
            time.sleep(0.01)
else:
    import tty
    import termios
    import select

    def get_char_blocking(timeout=None):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            if timeout is None:
                r, _, _ = select.select([fd], [], [])
            else:
                r, _, _ = select.select([fd], [], [], timeout)
            if r:
                ch = sys.stdin.read(1)
                # handle escape sequences for arrows
                if ch == "\x1b":
                    # try to read two more chars if available quickly
                    r2, _, _ = select.select([fd], [], [], 0.02)
                    if r2:
                        ch2 = sys.stdin.read(1)
                        ch3 = sys.stdin.read(1) if select.select([fd], [], [], 0.02)[0] else ""
                        return ch + ch2 + ch3
                return ch
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

# Terminal helpers
ESC = "\x1b"
CSI = ESC + "["

def clear_screen():
    sys.stdout.write(CSI + "2J" + CSI + "H")
    sys.stdout.flush()

def hide_cursor():
    sys.stdout.write(CSI + "?25l")
    sys.stdout.flush()

def show_cursor():
    sys.stdout.write(CSI + "?25h")
    sys.stdout.flush()

def move_to(r, c):
    sys.stdout.write(f"{CSI}{r};{c}H")

def get_terminal_size():
    try:
        import shutil
        cols, lines = shutil.get_terminal_size((80, 24))
        return max(40, cols), max(10, lines)
    except Exception:
        return 80, 24

# Game constants
WIDTH, HEIGHT = get_terminal_size()
PLAY_H = max(10, HEIGHT - 6)
PLAY_W = min(80, max(40, WIDTH - 4))
FPS = 12
TICK = 1.0 / FPS

# Symbols
PLAYER_CHAR = "P"
HAZARD_CHAR = "X"
GOOD_CHAR = "o"
EMPTY_CHAR = " "

# Game state
lock = threading.Lock()
player_x = PLAY_W // 2
score = 0
lives = 3
level = 1
running = True
paused = False
objects = []  # each is dict: {'x':int,'y':int,'ch':str,'speed':float,'acc':float}
spawn_timer = 0.0
spawn_interval = 0.9

# Input state
last_input = None

def clamp_int(v, a, b):
    return max(a, min(b, v))

def reset_game():
    global player_x, score, lives, level, objects, spawn_timer, spawn_interval, running, paused
    with lock:
        player_x = PLAY_W // 2
        score = 0
        lives = 3
        level = 1
        objects = []
        spawn_timer = 0.0
        spawn_interval = 0.9
        running = True
        paused = False

# Input thread
def input_thread():
    global last_input, running, paused
    try:
        while running:
            ch = get_char_blocking(timeout=0.08)
            if not ch:
                continue
            # normalize arrow keys and letters
            if ch in ("\x1b[D", "\x1bOD"):  # left arrow
                last_input = "LEFT"
            elif ch in ("\x1b[C", "\x1bOC"):  # right arrow
                last_input = "RIGHT"
            else:
                c = ch.lower()
                if c == "a":
                    last_input = "LEFT"
                elif c == "d":
                    last_input = "RIGHT"
                elif c == "s":
                    last_input = "STAY"
                elif c == "p":
                    paused = not paused
                elif c == "q":
                    running = False
                    break
                else:
                    last_input = None
    except Exception:
        running = False

def spawn_object():
    # spawn either hazard or good object at top row (y=1)
    typ = random.random()
    if typ < 0.18:
        ch = GOOD_CHAR
        speed = random.uniform(0.35, 0.7)
    else:
        ch = HAZARD_CHAR
        speed = random.uniform(0.45, 0.95)
    x = random.randint(1, PLAY_W-2)
    obj = {"x": x, "y": 1, "ch": ch, "speed": speed, "acc": 0.0}
    objects.append(obj)

def update_game(dt):
    global spawn_timer, spawn_interval, level, score, lives, running
    if paused:
        return
    spawn_timer += dt
    # spawn more often with level
    if spawn_timer >= spawn_interval:
        spawn_timer = 0.0
        spawn_object()
    # gently increase difficulty with score/time
    level = 1 + score // 10
    spawn_interval = max(0.25, 0.9 - (level-1)*0.06)
    # update objects
    remove = []
    for obj in list(objects):
        obj["acc"] += obj["speed"] * dt * 60  # scale by tick rate
        if obj["acc"] >= 1.0:
            step = int(obj["acc"])
            obj["y"] += step
            obj["acc"] -= step
        # collision with player?
        if obj["y"] >= PLAY_H:
            if abs(obj["x"] - player_x) <= 0:
                # hit player
                if obj["ch"] == HAZARD_CHAR:
                    lives -= 1
                else:
                    score += 3
            else:
                # missed: hazards disappear; goodies fall past
                pass
            remove.append(obj)
    for r in remove:
        if r in objects:
            objects.remove(r)
    # check lose
    if lives <= 0:
        running = False

def render():
    # draw frame
    move_to(1,1)
    sys.stdout.write("\n")  # ensure top-left
    # top status
    sys.stdout.write(f" Funnay Dodge — Score: {score}   Lives: {lives}   Level: {level}    (A/D or ←/→ to move, S to stay, P pause, Q quit)\n")
    # playfield border
    sys.stdout.write("+" + "-" * PLAY_W + "+\n")
    # build rows
    grid = [[EMPTY_CHAR for _ in range(PLAY_W)] for _ in range(PLAY_H)]
    # place objects
    for obj in objects:
        x = int(obj["x"]-1)
        y = int(obj["y"]-1)
        if 0 <= y < PLAY_H and 0 <= x < PLAY_W:
            grid[y][x] = obj["ch"]
    # place player on bottom row
    py = PLAY_H - 1
    px = clamp_int(player_x-1, 0, PLAY_W-1)
    grid[py][px] = PLAYER_CHAR
    # write rows
    for row in grid:
        sys.stdout.write("|")
        sys.stdout.write("".join(row))
        sys.stdout.write("|\n")
    sys.stdout.write("+" + "-" * PLAY_W + "+\n")
    sys.stdout.write(" Press Q to quit. Press P to pause.                \n")
    sys.stdout.flush()

def game_loop():
    global last_input, player_x, running
    last_time = time.time()
    try:
        while running:
            t0 = time.time()
            dt = t0 - last_time
            last_time = t0
            # process input
            if last_input:
                li = last_input
                last_input = None
                if li == "LEFT":
                    player_x = clamp_int(player_x-1, 1, PLAY_W)
                elif li == "RIGHT":
                    player_x = clamp_int(player_x+1, 1, PLAY_W)
                elif li == "STAY":
                    pass
            # update
            update_game(dt)
            # render
            clear_screen()
            render()
            # frame timing
            t1 = time.time()
            elapsed = t1 - t0
            sleep = max(0, TICK - elapsed)
            time.sleep(sleep)
    except KeyboardInterrupt:
        running = False

def show_game_over():
    clear_screen()
    sys.stdout.write("\n\n")
    sys.stdout.write("  ===== GAME OVER =====\n")
    sys.stdout.write(f"   Final Score: {score}\n")
    sys.stdout.write("   Thanks for playing Funnay Dodge!\n\n")
    sys.stdout.flush()

def main():
    global running
    clear_screen()
    hide_cursor()
    sys.stdout.write("Welcome to Funnay Dodge (terminal edition)!\n")
    sys.stdout.write("Controls: A/D or ←/→ to move, S stay, P pause, Q quit\n")
    sys.stdout.write("Press Enter to start...")
    sys.stdout.flush()
    try:
        # wait for Enter
        if IS_WIN:
            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch == "\r":
                        break
                time.sleep(0.05)
        else:
            _ = sys.stdin.readline()
        reset_game()
        it = threading.Thread(target=input_thread, daemon=True)
        it.start()
        game_loop()
    finally:
        running = False
        show_cursor()
        show_game_over()

if __name__ == "__main__":
    main()

"""
Task 3 - Knight's Tour on a dynamic chessboard   (console version)

Run:   python Chessknight.py

Input  (stdin):   line 1: "w h"   -> number of columns and rows of the board
                  line 2: "x y"   -> starting square of the knight (0-indexed,
                                     column first, then row)
Output (stdout):  the squares in visiting order, one "x,y" per line,
                  or the single line "path not found".

Method: depth-first backtracking where the order of the moves is chosen by
Warnsdorff's rule - always try first the move that leads to the square with the
FEWEST onward moves. Hard-to-reach squares are visited before they get stranded,
so the search almost never has to backtrack. Backtracking is still there, so the
answer is exact: if a tour exists from the start square it is found, otherwise
"path not found". An explicit stack is used, so big boards never hit Python's
recursion limit.

Only the Python standard library is needed.
"""

import sys
import json
import time
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1),
         (-2, -1), (-1, -2), (1, -2), (2, -1)]


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def knight_tour(w, h, sx, sy, time_limit=20):
    """Return the tour as a list of (x, y). Tracks best partial path if no full tour exists."""
    if w < 1 or h < 1 or not (0 <= sx < w and 0 <= sy < h):
        return []
    total = w * h

    visited = [[False] * w for _ in range(h)]
    start_time = time.perf_counter()

    def free_neighbours(x, y):
        return [(x + dx, y + dy) for dx, dy in MOVES
                if 0 <= x + dx < w and 0 <= y + dy < h and not visited[y + dy][x + dx]]

    def candidates(x, y, done):
        """Next moves from (x, y), best first. `done` = squares visited so far."""
        if time.perf_counter() - start_time > time_limit:
            raise TimeoutError("Time limit exceeded")
        cands = free_neighbours(x, y)
        remaining = total - done
        scored = []
        for cx, cy in cands:
            onward = free_neighbours(cx, cy)
            if not onward and remaining > 1:
                return []
            tie = sum(len(free_neighbours(ox, oy)) for ox, oy in onward)
            scored.append((len(onward), tie, cx, cy))
        scored.sort(reverse=True)
        return [(cx, cy) for _, _, cx, cy in scored]

    path = [(sx, sy)]
    visited[sy][sx] = True
    if total == 1:
        return path

    best_path = list(path)
    stack = [candidates(sx, sy, 1)]
    while stack:
        if time.perf_counter() - start_time > time_limit:
            raise TimeoutError("Time limit exceeded")
        if len(path) > len(best_path):
            best_path = list(path)
        if stack[-1]:
            nx, ny = stack[-1].pop()
            visited[ny][nx] = True
            path.append((nx, ny))
            if len(path) == total:
                return path
            stack.append(candidates(nx, ny, len(path)))
        else:
            stack.pop()
            x, y = path.pop()
            visited[y][x] = False
            
    return best_path

def why_no_tour(w, h, x, y):
    total = w * h
    if total % 2 == 1 and (x + y) % 2 == 1:
        return "Parity rule: on an odd-sized board, a knight must start on an even square ((x+y) must be even)."
    return "No knight's tour exists starting from this square on this board size."


# --------------------------------------------------------------------------- #
# Console mode (task specification)
# --------------------------------------------------------------------------- #
def read_four_numbers():
    """Read lines until 4 numbers are collected (works typed by hand or piped)."""
    tokens = []
    while len(tokens) < 4:
        try:
            tokens.extend(input().split())
        except EOFError:
            break
    return tokens


def run_console():
    tokens = read_four_numbers()
    try:
        w, h, sx, sy = (int(t) for t in tokens[:4])
    except ValueError:
        print("path not found")
        return
    # Parity rule: on an odd-sized board a tour must start on an even square.
    # Checking it first answers instantly instead of searching until timeout.
    if (w * h) % 2 == 1 and (sx + sy) % 2 == 1:
        print("path not found")
        return
    try:
        tour = knight_tour(w, h, sx, sy)
    except TimeoutError:
        tour = []
    if not tour or len(tour) != w * h:
        print("path not found")
    else:
        print("\n".join(f"{x},{y}" for x, y in tour))


# --------------------------------------------------------------------------- #
# Visual simulator (tiny local web server, standard library only)
# --------------------------------------------------------------------------- #
PORTS = range(5003, 5013)
MAX_BOARD = 100              # largest side accepted by the visual simulator
SOLVE_TIME_LIMIT_S = 20      # the visual simulator stops a search after this long

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knight's Tour Simulator</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;font-family:Arial,sans-serif;font-size:14px;color:#222;background:#f4f1ea}
 #app{display:flex;gap:18px;padding:16px;align-items:flex-start;flex-wrap:wrap}
 #left{flex:0 0 auto} #right{width:330px;max-width:100%}
 h1{font-size:18px;margin:0 0 10px}
 canvas{background:#fff;border:2px solid #5c4a32;border-radius:4px;display:block;cursor:pointer}
 .card{background:#fff;border:1px solid #d9d2c3;border-radius:8px;padding:12px;margin-bottom:12px}
 .row{display:flex;gap:8px;align-items:center;margin:6px 0;flex-wrap:wrap}
 label{font-weight:bold} input[type=number]{width:62px;padding:4px}
 select{padding:4px;max-width:100%}
 button{padding:7px 12px;border:0;border-radius:6px;background:#5c4a32;color:#fff;font-size:14px;cursor:pointer}
 button.primary{background:#2e7d32;font-weight:bold;padding:9px 16px}
 button:disabled{background:#aaa;cursor:not-allowed}
 #status{min-height:38px;margin-top:8px;line-height:1.4}
 .err{color:#b00020;font-weight:bold} .ok{color:#1b5e20;font-weight:bold} .muted{color:#666}
 textarea{width:100%;height:210px;font-family:Consolas,monospace;font-size:12px}
 input[type=range]{width:150px}
</style></head>
<body><div id="app">
 <div id="left">
  <h1>Knight's Tour Simulator</h1>
  <canvas id="board"></canvas>
  <div id="hover" class="muted" style="margin-top:6px">Click a square to choose the start. Coordinates are column,row (like the task slide).</div>
 </div>
 <div id="right">
  <div class="card">
   <div class="row"><label for="ex">Example</label>
    <select id="ex">
     <option value="">- pick an example -</option>
     <option value="5 5 2 2">5 5 / 2 2  (slide example)</option>
     <option value="5 5 0 0">5 5 / 0 0</option>
     <option value="5 5 0 1">5 5 / 0 1  (no path example)</option>
     <option value="6 6 0 0">6 6 / 0 0</option>
     <option value="8 8 0 0">8 8 / 0 0  (classic chessboard)</option>
     <option value="8 8 3 4">8 8 / 3 4</option>
     <option value="8 5 0 0">8 5 / 0 0  (not square)</option>
     <option value="3 4 0 0">3 4 / 0 0  (tiny board)</option>
     <option value="10 10 4 4">10 10 / 4 4</option>
     <option value="20 20 0 0">20 20 / 0 0</option>
     <option value="3 3 0 0">3 3 / 0 0  (no path)</option>
     <option value="4 4 0 0">4 4 / 0 0  (no path)</option>
    </select></div>
   <div class="row"><label>Board</label>
    <input id="w" type="number" min="1" max="__MAX__" value="5"> columns &times;
    <input id="h" type="number" min="1" max="__MAX__" value="5"> rows</div>
   <div class="row"><label>Start</label>
    x <input id="sx" type="number" min="0" value="2"> y <input id="sy" type="number" min="0" value="2"></div>
   <div class="row"><button class="primary" id="solve">Solve &amp; play</button></div>
   <div id="status" class="muted"></div>
  </div>
  <div class="card">
   <div class="row">
    <button id="first" title="Back to start">&#9198;</button>
    <button id="prev" title="Previous move">&#9664;</button>
    <button id="play" title="Play / pause">&#9654; Play</button>
    <button id="next" title="Next move">&#9654;</button>
    <button id="last" title="Jump to end">&#9197;</button>
   </div>
   <div class="row"><label for="speed">Speed</label>
    <input id="speed" type="range" min="1" max="60" value="6"> <span id="speedv">6</span> moves/s</div>
   <div class="row">
    <label><input type="checkbox" id="nums" checked> move numbers</label>
    <label><input type="checkbox" id="lines" checked> path lines</label>
   </div>
   <div id="progress" class="muted"></div>
  </div>
  <div class="card">
   <div class="row"><label>Output (same as the console version)</label><button id="copy">Copy</button></div>
   <textarea id="out" readonly></textarea>
  </div>
 </div>
</div>
<script>
const $ = id => document.getElementById(id);
const cv = $('board'), ctx = cv.getContext('2d');
let W = 5, H = 5, SX = 2, SY = 2;      // board + start
let path = [], step = 0, timer = null, order = null, cell = 60, margin = 0, busy = false;

function readInputs(){
  W = Math.max(1, Math.min(__MAX__, parseInt($('w').value) || 1));
  H = Math.max(1, Math.min(__MAX__, parseInt($('h').value) || 1));
  SX = Math.max(0, Math.min(W - 1, parseInt($('sx').value) || 0));
  SY = Math.max(0, Math.min(H - 1, parseInt($('sy').value) || 0));
  $('w').value = W; $('h').value = H; $('sx').value = SX; $('sy').value = SY;
}
function layout(){
  margin = (W <= 30 && H <= 30) ? 22 : 0;
  const avail = Math.max(260, Math.min(720, window.innerWidth - 380)) - margin;
  cell = Math.max(4, Math.floor(avail / Math.max(W, H)));
  cv.width = margin + cell * W; cv.height = margin + cell * H;
}
function stopPlay(){ if(timer){ clearInterval(timer); timer = null; } $('play').innerHTML = '&#9654; Play'; }
function startPlay(){
  if(!path.length) return;
  if(step >= path.length - 1) step = 0;
  stopPlay(); $('play').innerHTML = '&#10074;&#10074; Pause';
  timer = setInterval(() => { if(step < path.length - 1){ step++; draw(); } else { stopPlay(); } },
                      1000 / parseInt($('speed').value));
}

function draw(){
  ctx.clearRect(0, 0, cv.width, cv.height);
  const n = path.length;
  order = Array.from({length: H}, () => new Array(W).fill(-1));
  for(let i = 0; i <= step && i < n; i++) order[path[i][1]][path[i][0]] = i;

  for(let y = 0; y < H; y++){
    for(let x = 0; x < W; x++){
      const px = margin + x * cell, py = margin + y * cell;
      ctx.fillStyle = ((x + y) % 2 === 0) ? '#f0d9b5' : '#b58863';
      ctx.fillRect(px, py, cell, cell);
      const k = order[y][x];
      if(k >= 0){
        const hue = 210 - 200 * (n > 1 ? k / (n - 1) : 0);
        ctx.fillStyle = 'hsla(' + hue + ',75%,50%,0.55)';
        ctx.fillRect(px, py, cell, cell);
        if($('nums').checked && cell >= 20){
          ctx.fillStyle = '#111'; ctx.font = 'bold ' + Math.floor(cell * 0.34) + 'px Arial';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(String(k + 1), px + cell / 2, py + cell * 0.78);
        }
      }
    }
  }
  if(margin){
    ctx.fillStyle = '#5c4a32'; ctx.font = '11px Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for(let x = 0; x < W; x++) ctx.fillText(x, margin + x * cell + cell / 2, margin / 2);
    for(let y = 0; y < H; y++) ctx.fillText(y, margin / 2, margin + y * cell + cell / 2);
  }
  if($('lines').checked && n > 1){
    ctx.strokeStyle = 'rgba(120,0,40,0.85)'; ctx.lineWidth = Math.max(1, cell / 14); ctx.lineJoin = 'round';
    ctx.beginPath();
    for(let i = 0; i <= step && i < n; i++){
      const cx = margin + path[i][0] * cell + cell / 2, cy = margin + path[i][1] * cell + cell / 2;
      if(i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
  }
  ctx.strokeStyle = '#1b8a2f'; ctx.lineWidth = Math.max(2, cell / 10);
  ctx.strokeRect(margin + SX * cell + 1, margin + SY * cell + 1, cell - 2, cell - 2);
  
  const kx = n ? path[Math.min(step, n - 1)][0] : SX, ky = n ? path[Math.min(step, n - 1)][1] : SY;
  ctx.font = Math.floor(cell * 0.8) + 'px "Segoe UI Symbol", "Apple Symbols", Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillStyle = '#fff'; ctx.fillText('\u265E', margin + kx * cell + cell / 2 + 1, margin + ky * cell + cell / 2 + 2);
  ctx.fillStyle = '#000'; ctx.fillText('\u265E', margin + kx * cell + cell / 2, margin + ky * cell + cell / 2 + 1);

  const has = n > 0;
  ['first', 'prev', 'play', 'next', 'last'].forEach(id => $(id).disabled = !has);
  $('progress').textContent = has ? ('Move ' + (step + 1) + ' of ' + n + '  -  knight at ' + path[step][0] + ',' + path[step][1]
                                      + (step === n - 1 ? '  -  end of partial exploration' : ''))
                                  : 'No tour loaded yet.';
}

function setStatus(html, cls){
  $('status').className = cls || 'muted';
  $('status').innerHTML = html;
}
function resetSolution(){ stopPlay(); path = []; step = 0; $('out').value = ''; }
function refresh(){ readInputs(); layout(); draw(); }

async function solve(){
  if(busy) return;
  readInputs(); resetSolution(); layout(); draw();
  busy = true; $('solve').disabled = true; setStatus('Solving...');
  try {
    const r = await fetch('/api/solve?w=' + W + '&h=' + H + '&x=' + SX + '&y=' + SY);
    const data = await r.json();
    if(data.error){ setStatus(data.error, 'err'); }
    else {
      path = data.path || []; step = 0;
      if(!data.found){
        $('out').value = 'path not found';
        setStatus('path not found (visualizing partial path of ' + path.length + ' squares)<br><span class="muted" style="font-weight:normal">' + data.reason + '</span>', 'err');
      } else {
        $('out').value = path.map(p => p[0] + ',' + p[1]).join('\n');
        setStatus('Tour found: ' + path.length + ' squares, solved in ' + data.ms.toFixed(1) + ' ms.', 'ok');
      }
      draw(); startPlay();
    }
  } catch(e){ setStatus('Could not reach the solver: ' + e.message, 'err'); }
  busy = false; $('solve').disabled = false; draw();
}

// events
$('solve').onclick = solve;
$('play').onclick = () => { timer ? stopPlay() : startPlay(); };
$('first').onclick = () => { stopPlay(); step = 0; draw(); };
$('last').onclick = () => { stopPlay(); step = path.length - 1; draw(); };
$('prev').onclick = () => { stopPlay(); if(step > 0) step--; draw(); };
$('next').onclick = () => { stopPlay(); if(step < path.length - 1) step++; draw(); };
$('speed').oninput = () => {
  $('speedv').textContent = $('speed').value;
  if(timer) startPlay();
};
$('nums').onchange = draw;
$('lines').onchange = draw;
['w', 'h', 'sx', 'sy'].forEach(id => $(id).addEventListener('change', () => { resetSolution(); setStatus(''); refresh(); }));
$('ex').onchange = () => {
  const v = $('ex').value; if(!v) return;
  const [a, b, c, d] = v.split(' '); $('w').value = a; $('h').value = b; $('sx').value = c; $('sy').value = d;
  resetSolution(); setStatus(''); refresh();
};
$('copy').onclick = () => { $('out').select(); document.execCommand('copy'); };

function squareAt(e){
  const r = cv.getBoundingClientRect();
  const x = Math.floor((e.clientX - r.left - margin) / cell), y = Math.floor((e.clientY - r.top - margin) / cell);
  return (x >= 0 && x < W && y >= 0 && y < H) ? [x, y] : null;
}
cv.addEventListener('mousemove', e => {
  const s = squareAt(e);
  $('hover').textContent = s ? 'Square ' + s[0] + ',' + s[1] + (order && order[s[1]][s[0]] >= 0 ? '  (move ' + (order[s[1]][s[0]] + 1) + ')' : '')
                             : 'Click a square to choose the start. Coordinates are column,row (like the task slide).';
});
cv.addEventListener('click', e => {
  const s = squareAt(e); if(!s) return;
  $('sx').value = s[0]; $('sy').value = s[1]; resetSolution(); setStatus('Start set to ' + s[0] + ',' + s[1] + '. Press "Solve & play".'); refresh();
});
window.addEventListener('resize', refresh);

window.onload = refresh;
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, PAGE.replace("__MAX__", str(MAX_BOARD)), "text/html; charset=utf-8")
            return
        if url.path == "/api/solve":
            q = parse_qs(url.query)
            try:
                w, h, x, y = (int(q[k][0]) for k in ("w", "h", "x", "y"))
            except (KeyError, ValueError, IndexError):
                self._send(400, json.dumps({"error": "Invalid input."}), "application/json")
                return
            if not (1 <= w <= MAX_BOARD and 1 <= h <= MAX_BOARD):
                self._send(400, json.dumps({"error": f"Board sides must be 1-{MAX_BOARD}."}), "application/json")
                return
            if not (0 <= x < w and 0 <= y < h):
                self._send(400, json.dumps({"error": "The start square must be inside the board."}), "application/json")
                return
            if (w * h) % 2 == 1 and (x + y) % 2 == 1:
                # Parity rule: impossible, so answer instantly instead of searching.
                self._send(200, json.dumps({"found": False, "path": [[x, y]],
                                            "reason": why_no_tour(w, h, x, y), "ms": 0.0}),
                           "application/json")
                return
            t0 = time.perf_counter()
            try:
                tour = knight_tour(w, h, x, y, time_limit=SOLVE_TIME_LIMIT_S)
            except TimeoutError:
                self._send(200, json.dumps({"error": f"Stopped after {SOLVE_TIME_LIMIT_S} s without an answer."}),
                           "application/json")
                return
            ms = (time.perf_counter() - t0) * 1000
            
            total = w * h
            complete = (tour is not None and len(tour) == total)
            body = {
                "found": complete,
                "path": [list(p) for p in tour] if tour else [],
                "reason": why_no_tour(w, h, x, y) if not complete else "",
                "ms": ms
            }
            self._send(200, json.dumps(body), "application/json")
            return
        self._send(404, "Not found", "text/plain")

    def log_message(self, *args):
        pass


def run_visual():
    server = None
    for port in PORTS:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        print("Could not find a free port.")
        return
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Knight's Tour visual simulator running at {url}   (Ctrl+C to stop)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    if any(a in ("--visual", "-v", "visual") for a in sys.argv[1:]):
        run_visual()
    else:
        run_console()

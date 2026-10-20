"""
DLH Dump Truck Simulator - Chinese Postman Problem (CPP) on Surabaya's road network.

Run:  python cpp.py
Then open http://127.0.0.1:5000  (a browser tab opens automatically).

In the page the user can:
  * click ANYWHERE on the map to place the dump-truck base,
  * adjust the delivery radius x (km) with a slider,
  * press "Solve" to get the CPP route overlaid on the map, the street
    traversal order, total distance, estimated travel time and solver time.

Requirements:  pip install osmnx networkx numpy flask shapely scipy
"""

import ast
import math
import os
import re
import threading
import time
import webbrowser
from collections import defaultdict

import networkx as nx
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLACE_NAME = "Surabaya, Indonesia"
GRAPH_FILENAME = os.path.join(BASE_DIR, "surabaya_road_network.graphml")

DEFAULT_CENTER = (-7.2575, 112.7521)   # Surabaya centre (map start view)
DEFAULT_RADIUS_KM = 1.5
MIN_RADIUS_KM, MAX_RADIUS_KM = 0.3, 5.0
MAX_SNAP_KM = 1.0                      # click must be within this of a road
ROUTING_BUFFER_KM = 1.0                # extra area the truck may use to detour
EXACT_TIME_LIMIT_S = 30                # max time for the exact (shortest-route) solver

# Roads that MUST be collected (task: trunk, primary, secondary, tertiary).
REQUIRED_HIGHWAYS = {
    "trunk", "primary", "secondary", "tertiary",
    "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}

# Assumed urban dump-truck speeds (km/h) per road class; an OSM `maxspeed`
# tag, when present, caps the value.
CLASS_SPEED_KMH = {
    "trunk": 40, "trunk_link": 25,
    "primary": 35, "primary_link": 25,
    "secondary": 30, "secondary_link": 25,
    "tertiary": 25, "tertiary_link": 20,
}
DEFAULT_SPEED_KMH = 20


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        return [s] if s else []
    return [str(value)]


def street_name(data):
    names = [n for n in as_list(data.get("name")) if n and n.lower() != "nan"]
    if names:
        return " / ".join(dict.fromkeys(names))
    hw = as_list(data.get("highway"))
    return f"(unnamed {hw[0] if hw else 'road'})"


def road_class(highways):
    for h in highways:
        if h in REQUIRED_HIGHWAYS:
            return h
    return highways[0] if highways else "road"


def parse_maxspeed(value):
    speeds = []
    for s in as_list(value):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(mph)?", s)
        if m:
            v = float(m.group(1))
            speeds.append(v * 1.609 if m.group(2) else v)
    return min(speeds) if speeds else None


def edge_speed_kmh(highway, maxspeed):
    v = CLASS_SPEED_KMH.get(highway, DEFAULT_SPEED_KMH)
    ms = parse_maxspeed(maxspeed)
    return min(v, ms) if ms else v


def get_geometry(data):
    geom = data.get("geometry")
    if isinstance(geom, str):
        try:
            from shapely import wkt as shapely_wkt
            geom = shapely_wkt.loads(geom)
        except Exception:
            geom = None
    return geom


def fmt_duration(minutes):
    h, m = divmod(minutes, 60)
    return f"{int(h)} h {m:.0f} min" if h >= 1 else f"{minutes:.1f} minutes"


# --------------------------------------------------------------------------- #
# 1. Road network  (OSM -> directed graph)
# --------------------------------------------------------------------------- #
def load_surabaya_graph():
    import osmnx as ox
    if os.path.exists(GRAPH_FILENAME):
        print(f"[-] Loading road network from '{GRAPH_FILENAME}'...")
        try:
            return ox.load_graphml(GRAPH_FILENAME)
        except Exception as e:
            print(f"[!] Cached GraphML unreadable ({e}); re-downloading...")
            os.remove(GRAPH_FILENAME)
    print(f"[-] Downloading road network for {PLACE_NAME} (one-time, may take minutes)...")
    G = ox.graph_from_place(PLACE_NAME, network_type="drive")
    ox.save_graphml(G, filepath=GRAPH_FILENAME)
    print(f"[+] Saved to '{GRAPH_FILENAME}'.")
    return G


def build_digraph(G):
    """MultiDiGraph from OSM -> simple DiGraph restricted to the largest
    strongly connected component (so every node can reach every other)."""
    scc = max(nx.strongly_connected_components(G), key=len)
    G_scc = G.subgraph(scc)

    D = nx.DiGraph()
    for n, d in G_scc.nodes(data=True):
        D.add_node(n, x=float(d["x"]), y=float(d["y"]))

    for u, v, d in G_scc.edges(data=True):
        length = float(d.get("length", 0.0))
        highways = as_list(d.get("highway"))
        required = any(h in REQUIRED_HIGHWAYS for h in highways)
        cls = road_class(highways)
        cur = D.get_edge_data(u, v)
        if cur is None or length < cur["length"]:
            D.add_edge(u, v, length=length, name=street_name(d), highway=cls,
                       geometry=get_geometry(d),
                       speed=edge_speed_kmh(cls, d.get("maxspeed")),
                       required=required or (cur["required"] if cur else False))
        elif required:
            cur["required"] = True

    print(f"[+] Digraph ready: {D.number_of_nodes():,} nodes, "
          f"{D.number_of_edges():,} directed edges.")
    return D


class RoadIndex:
    """Vectorised node lookup (no scikit-learn needed)."""

    def __init__(self, D):
        self.D = D
        self.ids = list(D.nodes)
        self.lat = np.array([D.nodes[n]["y"] for n in self.ids])
        self.lon = np.array([D.nodes[n]["x"] for n in self.ids])

    def dist_km(self, lat0, lon0):
        k = math.cos(math.radians(lat0))
        dy = (self.lat - lat0) * 110.574
        dx = (self.lon - lon0) * 111.320 * k
        return np.hypot(dx, dy)

    def nearest(self, lat, lon):
        d = self.dist_km(lat, lon)
        i = int(np.argmin(d))
        return self.ids[i], float(d[i])

    def within(self, lat, lon, radius_km):
        mask = self.dist_km(lat, lon) <= radius_km
        return {n for n, keep in zip(self.ids, mask) if keep}


# --------------------------------------------------------------------------- #
# 2. Problem construction
# --------------------------------------------------------------------------- #
def select_required(D, index, base, radius_km):
    """
    Returns (W, required_edges, dropped).

    W        : routing graph (strongly connected, contains base) the truck may
               drive on; it extends ROUTING_BUFFER_KM beyond the radius so
               detours are not artificially blocked at the boundary.
    required : directed trunk/primary/secondary/tertiary edges whose midpoint
               lies within radius_km of the base (each street is therefore
               counted once, with no ragged half-cut streets).
    dropped  : required edges inside the radius that are not connected to the
               base's component (allowed to be ignored by the task).
    """
    by, bx = D.nodes[base]["y"], D.nodes[base]["x"]
    region = index.within(by, bx, radius_km + ROUTING_BUFFER_KM)
    sub = D.subgraph(region)
    comp = next(c for c in nx.strongly_connected_components(sub) if base in c)
    W = D.subgraph(comp).copy()

    k = math.cos(math.radians(by))
    required, dropped = [], 0
    for u in region:
        for v, d in D[u].items():
            if not d["required"]:
                continue
            my = (D.nodes[u]["y"] + D.nodes[v]["y"]) / 2
            mx = (D.nodes[u]["x"] + D.nodes[v]["x"]) / 2
            if math.hypot((my - by) * 110.574, (mx - bx) * 111.320 * k) > radius_km:
                continue
            if W.has_edge(u, v):
                required.append((u, v))
            else:
                dropped += 1
    return W, required, dropped


# --------------------------------------------------------------------------- #
# 3. CPP solver (directed)
# --------------------------------------------------------------------------- #
def _add_path(M, path):
    for a, b in zip(path[:-1], path[1:]):
        M.add_edge(a, b)


def _heuristic_multigraph(W, required, base):
    """
    Fast heuristic: connect the pieces with shortest paths, then balance with
    min-cost flow. Always valid, usually good, not guaranteed shortest.
    """
    M = nx.MultiDiGraph()
    M.add_node(base)
    for u, v in required:
        M.add_edge(u, v)

    # connect components (closest piece to the base's piece, one path each)
    W_rev = W.reverse(copy=False)
    while True:
        comps = list(nx.weakly_connected_components(M))
        if len(comps) == 1:
            break
        main = next(c for c in comps if base in c)
        others = set().union(*[c for c in comps if base not in c])
        d_f, p_f = nx.multi_source_dijkstra(W, main, weight="length")
        d_b, p_b = nx.multi_source_dijkstra(W_rev, main, weight="length")
        b = min(others, key=lambda n: min(d_f.get(n, math.inf), d_b.get(n, math.inf)))
        if d_f.get(b, math.inf) <= d_b.get(b, math.inf):
            path = p_f[b]            # main -> b
        else:
            path = p_b[b][::-1]      # b -> main
        _add_path(M, path)

    # balance in/out degree by min-cost flow
    supply, demand = {}, {}
    for n in M.nodes:
        d = M.in_degree(n) - M.out_degree(n)
        if d > 0:
            supply[n] = d
        elif d < 0:
            demand[n] = -d
    if supply:
        F = nx.DiGraph()
        for s, c in supply.items():
            F.add_node(s, demand=-c)
        for t, c in demand.items():
            F.add_node(t, demand=c)
        for s in supply:
            dist = nx.single_source_dijkstra_path_length(W, s, weight="length")
            for t in demand:
                if t in dist:
                    F.add_edge(s, t, weight=int(round(dist[t] * 10)))
        flow = nx.min_cost_flow(F)
        for s, targets in flow.items():
            for t, f in targets.items():
                if f > 0:
                    path = nx.shortest_path(W, s, t, weight="length")
                    for _ in range(f):
                        _add_path(M, path)
    return M


def _exact_multigraph(W, required, base, time_limit_s):
    """
    Exact directed Rural/Chinese Postman by integer programming.

        minimise   sum_e length_e * x_e         (x_e = extra traversals of e)
        subject to out_x(v) - in_x(v) = in_req(v) - out_req(v)   for every node
                   x_e >= 0 integer
                   every piece of the required network can be reached from and
                   returned to the base (connectivity cuts added lazily)

    The balance constraints alone are a min-cost-flow problem; the connectivity
    cuts are what make the plain flow solution (which may leave isolated pieces)
    a true shortest closed route. Returns (multigraph, proven_optimal) or None.
    """
    try:
        import scipy.sparse as sp
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        return None

    nodes = list(W.nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    edges = list(W.edges)
    m = len(edges)
    tail = np.array([pos[u] for u, _ in edges])
    head = np.array([pos[v] for _, v in edges])
    cost = np.array([W[u][v]["length"] for u, v in edges])

    rows = np.concatenate([tail, head])
    cols = np.concatenate([np.arange(m), np.arange(m)])
    vals = np.concatenate([np.ones(m), -np.ones(m)])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), m))

    rhs = np.zeros(len(nodes))
    for u, v in required:
        rhs[pos[v]] += 1      # in_req
        rhs[pos[u]] -= 1      # out_req

    def comps_of(M):
        out = []
        for c in nx.weakly_connected_components(M):
            if base in c:
                continue
            if any(M.out_degree(n) for n in c):
                out.append(c)
        return out

    def cut_row(S):
        mask = np.array([(u in S) and (v not in S) for u, v in edges], dtype=float)
        return sp.csr_matrix(mask)

    R = nx.MultiDiGraph()
    R.add_node(base)
    for u, v in required:
        R.add_edge(u, v)
    cut_rows = [cut_row(S) for S in comps_of(R)]

    deadline = time.perf_counter() + time_limit_s
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.5:
            return None
        cons = [LinearConstraint(A, rhs, rhs)]
        if cut_rows:
            cons.append(LinearConstraint(sp.vstack(cut_rows), lb=1, ub=np.inf))
        res = milp(cost, constraints=cons, integrality=np.ones(m),
                   bounds=Bounds(0, np.inf), options={"time_limit": remaining})
        if res.x is None:
            return None
        x = np.rint(res.x).astype(int)

        M = nx.MultiDiGraph()
        M.add_node(base)
        for u, v in required:
            M.add_edge(u, v)
        for (u, v), k in zip(edges, x):
            for _ in range(int(k)):
                M.add_edge(u, v)
        bad = comps_of(M)
        if not bad:
            return M, (res.status == 0)
        cut_rows.extend(cut_row(S) for S in bad)


def _mg_cost(W, M):
    return sum(W[u][v]["length"] for u, v in M.edges())


def solve_cpp(W, required, base, time_limit_s=None):
    """
    Shortest closed route from `base` that traverses every required directed
    edge at least once, using real roads in W.

    Returns (circuit, info). info["optimal"] is True when the route is proven
    to be the shortest possible; otherwise it is the best route found (the
    heuristic result, or a better ILP incumbent).
    """
    if time_limit_s is None:
        time_limit_s = EXACT_TIME_LIMIT_S
    M = _heuristic_multigraph(W, required, base)
    info = {"heuristic_m": _mg_cost(W, M), "optimal": False, "method": "heuristic"}

    exact = _exact_multigraph(W, required, base, time_limit_s)
    if exact is not None:
        M_x, proven = exact
        if _mg_cost(W, M_x) <= info["heuristic_m"] + 1e-6:
            M, info["method"] = M_x, "exact"
        info["optimal"] = bool(proven)
    info["final_m"] = _mg_cost(W, M)

    if not nx.is_eulerian(M):
        raise RuntimeError("Internal error: balanced graph is not Eulerian.")
    return list(nx.eulerian_circuit(M, source=base)), info


def edge_coords(W, u, v, geom):
    if geom is None:
        return [(W.nodes[u]["y"], W.nodes[u]["x"]), (W.nodes[v]["y"], W.nodes[v]["x"])]
    coords = [(lat, lon) for lon, lat in geom.coords]
    uy, ux = W.nodes[u]["y"], W.nodes[u]["x"]
    d_start = (coords[0][0] - uy) ** 2 + (coords[0][1] - ux) ** 2
    d_end = (coords[-1][0] - uy) ** 2 + (coords[-1][1] - ux) ** 2
    return coords[::-1] if d_end < d_start else coords


def build_segments(W, circuit, required_set):
    """Merge consecutive edges of the same street & type into ordered segments."""
    collected = set()
    segments = []
    for u, v in circuit:
        d = W[u][v]
        is_collect = (u, v) in required_set and (u, v) not in collected
        if is_collect:
            collected.add((u, v))
        coords = edge_coords(W, u, v, d.get("geometry"))
        last = segments[-1] if segments else None
        if last and last["name"] == d["name"] and last["collect"] == is_collect:
            last["length"] += d["length"]
            last["coords"].extend(coords[1:])
        else:
            segments.append({"name": d["name"], "highway": d["highway"],
                             "collect": is_collect, "length": d["length"],
                             "coords": list(coords)})
    return segments


def run_cpp(D, index, lat, lon, radius_km):
    """Full pipeline for one request. Returns a JSON-serialisable dict."""
    base, snap_km = index.nearest(lat, lon)
    if snap_km > MAX_SNAP_KM:
        raise ValueError("No road found near that point. Click inside Surabaya.")

    W, required, dropped = select_required(D, index, base, radius_km)
    if not required:
        raise ValueError("No trunk/primary/secondary/tertiary roads within that "
                         "radius. Try a larger radius or another location.")

    t0 = time.perf_counter()
    circuit, info = solve_cpp(W, required, base)
    exec_ms = (time.perf_counter() - t0) * 1000

    required_set = set(required)
    segments = build_segments(W, circuit, required_set)

    total_m = sum(W[u][v]["length"] for u, v in circuit)
    collect_m = sum(W[u][v]["length"] for u, v in required_set)
    seconds = sum(W[u][v]["length"] / (W[u][v]["speed"] / 3.6) for u, v in circuit)

    return {
        "base": {"lat": W.nodes[base]["y"], "lon": W.nodes[base]["x"], "snap_m": snap_km * 1000},
        "radius_km": radius_km,
        "required_edges": len(required),
        "dropped": dropped,
        "stats": {
            "total_km": total_m / 1000,
            "collect_km": collect_m / 1000,
            "revisit_km": (total_m - collect_m) / 1000,
            "minutes": seconds / 60,
            "duration": fmt_duration(seconds / 60),
            "exec_ms": exec_ms,
            "optimal": info["optimal"],
            "method": info["method"],
            "heuristic_km": info["heuristic_m"] / 1000,
            "edges_traversed": len(circuit),
        },
        "segments": [
            {"i": i, "name": s["name"], "highway": s["highway"],
             "collect": s["collect"], "km": s["length"] / 1000,
             "coords": [[round(a, 6), round(b, 6)] for a, b in s["coords"]]}
            for i, s in enumerate(segments, 1)
        ],
    }


# --------------------------------------------------------------------------- #
# 4. Web app (Flask + Leaflet)
# --------------------------------------------------------------------------- #
PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DLH Dump Truck CPP Simulator</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
 *{box-sizing:border-box} html,body{margin:0;height:100%;font-family:Arial,sans-serif;font-size:13px;color:#222}
 #app{display:flex;height:100%} #side{width:370px;display:flex;flex-direction:column;border-right:1px solid #ccc;background:#fafafa}
 #map{flex:1} .pad{padding:12px 14px} h1{font-size:16px;margin:0 0 4px}
 .muted{color:#666} label{display:block;margin:10px 0 4px;font-weight:bold}
 input[type=range]{width:100%} button{width:100%;padding:10px;border:0;border-radius:6px;background:#28a745;color:#fff;font-size:14px;font-weight:bold;cursor:pointer}
 button:disabled{background:#999;cursor:wait} #msg{margin-top:8px;min-height:18px}
 .err{color:#c00} #stats{display:none;border-top:1px solid #ddd;border-bottom:1px solid #ddd;background:#fff}
 #stats table{width:100%;border-collapse:collapse} #stats td{padding:2px 0} #stats td:last-child{text-align:right;font-weight:bold}
 #orderhead{display:none;justify-content:space-between;align-items:center;padding:8px 14px 4px}
 #orderhead a{color:#0a58ca;cursor:pointer;text-decoration:underline}
 #order{flex:1;overflow:auto;margin:0;padding:0 8px 12px 8px;list-style:none}
 #order li{padding:4px 6px;border-bottom:1px solid #eee;cursor:pointer;line-height:1.35}
 #order li:hover{background:#eef6ff} .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
 .legend{display:flex;gap:12px;margin-top:8px}
</style></head>
<body><div id="app">
 <div id="side">
  <div class="pad">
   <h1>DLH Dump Truck &ndash; Chinese Postman Simulator</h1>
   <div class="muted">1. Click anywhere on the map to set the truck base.<br>2. Set the radius. 3. Press Solve.</div>
   <label>Base location</label><div id="coords" class="muted">Not selected</div>
   <label>Delivery radius x: <span id="rv"></span> km</label>
   <input id="radius" type="range" min="__MINR__" max="__MAXR__" step="0.1" value="__DEFR__">
   <button id="go" disabled>Solve route</button>
   <div id="msg" class="muted"></div>
   <div class="legend"><span><span class="dot" style="background:#28a745"></span>Collection</span>
    <span><span class="dot" style="background:#fd7e14"></span>Revisit / connector</span></div>
   <label style="font-weight:normal"><input type="checkbox" id="arrows" checked> Show direction arrows</label>
  </div>
  <div id="stats" class="pad"><table>
   <tr><td>Total distance</td><td id="s_total"></td></tr>
   <tr><td>&nbsp;&bull; Collection</td><td id="s_col"></td></tr>
   <tr><td>&nbsp;&bull; Revisit / connector</td><td id="s_rev"></td></tr>
   <tr><td>Estimated travel time</td><td id="s_time"></td></tr>
   <tr><td>CPP solver time</td><td id="s_exec"></td></tr>
   <tr><td>Route quality</td><td id="s_opt"></td></tr>
   <tr><td>Required road segments</td><td id="s_req"></td></tr></table></div>
  <div id="orderhead"><b>Street traversal order</b><a id="dl">download .txt</a></div>
  <ol id="order" style="counter-reset:none"></ol>
 </div>
 <div id="map"></div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet-polylinedecorator/1.6.0/leaflet.polylineDecorator.min.js"></script>
<script>
const map = L.map('map').setView([__LAT__, __LON__], 13);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}).addTo(map);

const radiusEl = document.getElementById('radius'), rv = document.getElementById('rv');
const go = document.getElementById('go'), msg = document.getElementById('msg');
let base = null, baseMarker = null, circle = null;
let routeLayer = L.layerGroup().addTo(map), arrowLayer = L.layerGroup().addTo(map);
let lines = [], lastData = null;

function setMsg(t, err){ msg.textContent = t || ''; msg.className = err ? 'err' : 'muted'; }
function radiusKm(){ return parseFloat(radiusEl.value); }
function drawCircle(){
  if(!base) return;
  if(circle) map.removeLayer(circle);
  circle = L.circle([base.lat, base.lon], {radius: radiusKm()*1000, color:'crimson', fillOpacity:0.05}).addTo(map);
}
rv.textContent = radiusKm().toFixed(1);
radiusEl.addEventListener('input', () => { rv.textContent = radiusKm().toFixed(1); drawCircle(); });

map.on('click', e => {
  base = {lat: e.latlng.lat, lon: e.latlng.lng};
  if(baseMarker) map.removeLayer(baseMarker);
  baseMarker = L.marker(e.latlng).addTo(map).bindTooltip('Dump truck base');
  document.getElementById('coords').textContent = base.lat.toFixed(5) + ', ' + base.lon.toFixed(5);
  go.disabled = false; drawCircle(); setMsg('');
});

function esc(s){ return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function drawArrows(){
  arrowLayer.clearLayers();
  if(!document.getElementById('arrows').checked || !window.L.polylineDecorator || !lastData) return;
  lastData.segments.forEach(s => {
    if(s.coords.length < 2) return;
    arrowLayer.addLayer(L.polylineDecorator(L.polyline(s.coords), {patterns:[{
      offset: '12%', repeat: 90,
      symbol: L.Symbol.arrowHead({pixelSize: 7, polygon: false,
        pathOptions: {stroke: true, weight: 2, color: s.collect ? '#145a24' : '#9a4a00', opacity: 0.9}})}]}));
  });
}
document.getElementById('arrows').addEventListener('change', drawArrows);

go.addEventListener('click', async () => {
  if(!base) return;
  go.disabled = true; setMsg('Solving... (up to ~30 s for large radii)');
  try {
    const r = await fetch('/api/solve', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({lat: base.lat, lon: base.lon, radius: radiusKm()})});
    const data = await r.json();
    if(!r.ok) throw new Error(data.error || 'Server error');
    show(data); setMsg(data.dropped ? data.dropped + ' disconnected segment(s) ignored.' : '');
  } catch(e){ setMsg(e.message, true); }
  go.disabled = false;
});

function show(data){
  lastData = data; routeLayer.clearLayers(); lines = [];
  base = {lat: data.base.lat, lon: data.base.lon};   // snapped to nearest road node
  if(baseMarker) map.removeLayer(baseMarker);
  baseMarker = L.marker([base.lat, base.lon]).addTo(map).bindTooltip('Dump truck base (start/end)');
  document.getElementById('coords').textContent =
    base.lat.toFixed(5) + ', ' + base.lon.toFixed(5) + '  (snapped ' + data.base.snap_m.toFixed(0) + ' m to road)';
  drawCircle();

  const ol = document.getElementById('order'); ol.innerHTML = '';
  const start = document.createElement('li'); start.innerHTML = '<b>0. [START]</b> Dump truck base'; ol.appendChild(start);
  data.segments.forEach((s, idx) => {
    const col = s.collect ? '#28a745' : '#fd7e14';
    const label = '#' + s.i + ' ' + s.name + ' (' + s.highway + ') - ' + s.km.toFixed(2) + ' km [' + (s.collect ? 'collection' : 'revisit/connector') + ']';
    const pl = L.polyline(s.coords, s.collect ? {color: col, weight: 5, opacity: 0.9}
                                              : {color: col, weight: 3, opacity: 0.8, dashArray: '6 6'})
      .bindTooltip(label).addTo(routeLayer);
    lines.push(pl);
    const li = document.createElement('li');
    li.innerHTML = '<span class="dot" style="background:' + col + '"></span><b>' + s.i + '.</b> ' + esc(s.name) +
      ' <span class="muted">(' + s.highway + ') ' + s.km.toFixed(2) + ' km</span>';
    li.onclick = () => { map.fitBounds(pl.getBounds(), {maxZoom: 17});
      const w = pl.options.weight; pl.setStyle({color:'#0000ff', weight: w+4});
      setTimeout(() => pl.setStyle({color: col, weight: w}), 1500); };
    ol.appendChild(li);
  });
  const end = document.createElement('li'); end.innerHTML = '<b>' + (data.segments.length + 1) + '. [END]</b> Return to base'; ol.appendChild(end);

  const st = data.stats;
  document.getElementById('s_total').textContent = st.total_km.toFixed(2) + ' km';
  document.getElementById('s_col').textContent = st.collect_km.toFixed(2) + ' km';
  document.getElementById('s_rev').textContent = st.revisit_km.toFixed(2) + ' km';
  document.getElementById('s_time').textContent = st.duration;
  document.getElementById('s_exec').textContent = st.exec_ms.toFixed(1) + ' ms';
  document.getElementById('s_opt').textContent = st.optimal ? 'Shortest possible (proven)' : 'Best found (not proven)';
  document.getElementById('s_req').textContent = data.required_edges;
  document.getElementById('stats').style.display = 'block';
  document.getElementById('orderhead').style.display = 'flex';
  drawArrows();
  map.fitBounds(L.latLngBounds(data.segments.flatMap(s => s.coords)), {padding: [30, 30]});
}

document.getElementById('dl').addEventListener('click', () => {
  if(!lastData) return;
  const st = lastData.stats;
  let t = 'Total distance: ' + st.total_km.toFixed(2) + ' km\\nEstimated time: ' + st.duration +
          '\\nCPP solver time: ' + st.exec_ms.toFixed(2) + ' ms\\n\\n0. [START] Base\\n';
  lastData.segments.forEach(s => { t += s.i + '. ' + s.name + ' (' + s.highway + ') - ' + s.km.toFixed(2) +
    ' km [' + (s.collect ? 'collection' : 'revisit/connector') + ']\\n'; });
  t += (lastData.segments.length + 1) + '. [END] Return to base\\n';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([t], {type: 'text/plain'})); a.download = 'dlh_street_order.txt'; a.click();
});
</script></body></html>
"""


def create_app(D):
    from flask import Flask, jsonify, request, Response

    app = Flask(__name__)
    index = RoadIndex(D)
    lock = threading.Lock()   # one solve at a time (CPU-bound)

    page = (PAGE.replace("__LAT__", str(DEFAULT_CENTER[0]))
                .replace("__LON__", str(DEFAULT_CENTER[1]))
                .replace("__MINR__", str(MIN_RADIUS_KM))
                .replace("__MAXR__", str(MAX_RADIUS_KM))
                .replace("__DEFR__", str(DEFAULT_RADIUS_KM)))

    @app.get("/")
    def home():
        return Response(page, mimetype="text/html")

    @app.post("/api/solve")
    def solve():
        try:
            body = request.get_json(force=True)
            lat, lon = float(body["lat"]), float(body["lon"])
            radius = float(body["radius"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="Invalid request."), 400
        if not (MIN_RADIUS_KM <= radius <= MAX_RADIUS_KM):
            return jsonify(error=f"Radius must be {MIN_RADIUS_KM}-{MAX_RADIUS_KM} km."), 400
        try:
            with lock:
                return jsonify(run_cpp(D, index, lat, lon, radius))
        except ValueError as e:
            return jsonify(error=str(e)), 422

    return app


def main():
    G = load_surabaya_graph()
    D = build_digraph(G)
    app = create_app(D)
    url = "http://127.0.0.1:5000"
    print(f"[+] Open {url}")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
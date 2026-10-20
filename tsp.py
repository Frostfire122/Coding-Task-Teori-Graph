"""
SPPG Simulator - Travelling Salesman Problem (TSP) on Surabaya's road network.

Run:  python tsp.py
Then open http://127.0.0.1:5001  (a browser tab opens automatically).

In the page the user can:
  * click ANYWHERE on the map to place the SPPG (the kitchen / depot),
  * adjust the delivery radius x (km) with a slider,
  * press "Solve" to get the shortest closed tour that starts at the SPPG,
    visits every school within x km exactly once and returns to the SPPG,
    driving on the real (one-way aware) road network.

Displayed: route overlaid on the map, school visiting order, total distance,
estimated travel time and the TSP solver execution time.

Requirements:  pip install osmnx networkx numpy scipy flask shapely
"""

import ast
import json
import math
import os
import re
import threading
import time
import webbrowser

import networkx as nx
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLACE_NAME = "Surabaya, Indonesia"
GRAPH_FILENAME = os.path.join(BASE_DIR, "surabaya_road_network.graphml")
SCHOOLS_FILENAME = os.path.join(BASE_DIR, "surabaya_schools.json")

DEFAULT_CENTER = (-7.2825, 112.7949)    # ITS Sukolilo (map start view)
DEFAULT_RADIUS_KM = 3.0
MIN_RADIUS_KM, MAX_RADIUS_KM = 0.5, 10.0
MAX_SNAP_KM = 1.0                       # a click must be within this of a road
EXACT_TIME_LIMIT_S = 30                 # max time for the exact TSP solver
PORT = 5001

# Assumed delivery-van speeds (km/h) per road class; an OSM `maxspeed` tag,
# when present, caps the value.
CLASS_SPEED_KMH = {
    "motorway": 50, "motorway_link": 30, "trunk": 40, "trunk_link": 25,
    "primary": 35, "primary_link": 25, "secondary": 30, "secondary_link": 25,
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


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; works with scalars or numpy arrays."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# 1. Data: road network + schools (OSM)
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
    """OSM MultiDiGraph -> simple DiGraph restricted to the largest strongly
    connected component (every node can reach every other node)."""
    scc = max(nx.strongly_connected_components(G), key=len)
    G_scc = G.subgraph(scc)
    D = nx.DiGraph()
    for n, d in G_scc.nodes(data=True):
        D.add_node(n, x=float(d["x"]), y=float(d["y"]))
    for u, v, d in G_scc.edges(data=True):
        length = float(d.get("length", 0.0))
        cur = D.get_edge_data(u, v)
        if cur is None or length < cur["length"]:
            hw = as_list(d.get("highway"))
            cls = hw[0] if hw else "road"
            D.add_edge(u, v, length=length, geometry=get_geometry(d),
                       speed=edge_speed_kmh(cls, d.get("maxspeed")))
    print(f"[+] Digraph ready: {D.number_of_nodes():,} nodes, "
          f"{D.number_of_edges():,} directed edges.")
    return D


def load_schools():
    """List of {name, lat, lon}. Re-uses surabaya_schools.json when present."""
    raw = None
    if os.path.exists(SCHOOLS_FILENAME):
        print(f"[-] Loading schools from '{SCHOOLS_FILENAME}'...")
        with open(SCHOOLS_FILENAME, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        import osmnx as ox
        print("[-] Fetching school locations from OpenStreetMap (one-time)...")
        gdf = ox.features_from_place(PLACE_NAME, tags={"amenity": "school"})
        gdf = gdf[~gdf.geometry.is_empty]
        cent = gdf.geometry.to_crs(epsg=32749).centroid.to_crs(epsg=4326)
        raw = [{"name": gdf.iloc[i].get("name"), "lat": float(c.y), "lon": float(c.x)}
               for i, c in enumerate(cent)]
        with open(SCHOOLS_FILENAME, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        print(f"[+] Saved to '{SCHOOLS_FILENAME}'.")

    schools, seen = [], set()
    for s in raw:
        name = s.get("name")
        if not isinstance(name, str) or not name.strip() or name.lower() == "nan":
            name = "Unnamed school"
        key = (name, round(float(s["lat"]), 5), round(float(s["lon"]), 5))
        if key in seen:
            continue
        seen.add(key)
        schools.append({"name": name, "lat": float(s["lat"]), "lon": float(s["lon"])})
    print(f"[+] {len(schools):,} schools loaded.")
    return schools


class RoadNet:
    """Digraph + fast vectorised node lookup + C-speed Dijkstra (scipy)."""

    def __init__(self, D):
        import scipy.sparse as sp
        self.D = D
        self.ids = list(D.nodes)
        self.pos = {n: i for i, n in enumerate(self.ids)}
        self.lat = np.array([D.nodes[n]["y"] for n in self.ids])
        self.lon = np.array([D.nodes[n]["x"] for n in self.ids])
        rows, cols, w = [], [], []
        for u, v, d in D.edges(data=True):
            rows.append(self.pos[u])
            cols.append(self.pos[v])
            w.append(max(d["length"], 0.01))     # keep zero-length edges as edges
        n = len(self.ids)
        self.csr = sp.csr_matrix((w, (rows, cols)), shape=(n, n))

    def nearest(self, lat, lon):
        """(node_index, distance_km) of the road node closest to a point."""
        d = haversine_km(lat, lon, self.lat, self.lon)
        i = int(np.argmin(d))
        return i, float(d[i])


# --------------------------------------------------------------------------- #
# 2. TSP solver (asymmetric: one-way streets make d[i][j] != d[j][i])
# --------------------------------------------------------------------------- #
def tour_cost(d, tour):
    return sum(d[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def tsp_nearest_neighbor(d):
    n = len(d)
    tour, left = [0], set(range(1, n))
    while left:
        cur = tour[-1]
        nxt = min(left, key=lambda j: d[cur][j])
        tour.append(nxt)
        left.remove(nxt)
    return tour


def tsp_or_opt(d, tour):
    """Or-opt local search (moves segments of 1-3 stops, keeps direction, so it
    is valid for asymmetric distances). Stop 0 (the SPPG) stays first."""
    tour = tour[:]
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for seg_len in (1, 2, 3):
            i = 1
            while i + seg_len <= n:
                p = tour[i - 1]
                q = tour[(i + seg_len) % n]
                s0, s1 = tour[i], tour[i + seg_len - 1]
                remove_gain = d[p][s0] + d[s1][q] - d[p][q]
                rest = tour[:i] + tour[i + seg_len:]
                best_delta, best_j = -1e-9, None
                for j in range(len(rest)):
                    a, b = rest[j], rest[(j + 1) % len(rest)]
                    if a == p:      # original position
                        continue
                    delta = d[a][s0] + d[s1][b] - d[a][b] - remove_gain
                    if delta < best_delta:
                        best_delta, best_j = delta, j
                if best_j is not None:
                    seg = tour[i:i + seg_len]
                    tour = rest[:best_j + 1] + seg + rest[best_j + 1:]
                    improved = True
                else:
                    i += 1
    return tour


def tsp_exact(d, time_limit_s):
    """
    Exact asymmetric TSP by integer programming (assignment constraints with
    lazily added subtour-elimination cuts). Returns (tour, proven_optimal) or
    None when scipy is missing or no solution is found within the time limit.
    """
    n = len(d)
    if n <= 2:
        return list(range(n)), True
    try:
        import scipy.sparse as sp
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        return None

    I = np.array([i for i in range(n) for j in range(n) if i != j])
    J = np.array([j for i in range(n) for j in range(n) if i != j])
    m = len(I)
    c = np.array([d[i][j] for i, j in zip(I, J)])
    ar = np.arange(m)
    A = sp.csr_matrix((np.ones(2 * m), (np.concatenate([I, n + J]), np.concatenate([ar, ar]))),
                      shape=(2 * n, m))      # rows 0..n-1: out-degree, n..2n-1: in-degree
    cut_rows, cut_ub = [], []

    deadline = time.perf_counter() + time_limit_s
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.5:
            return None
        cons = [LinearConstraint(A, 1, 1)]
        if cut_rows:
            cons.append(LinearConstraint(sp.vstack(cut_rows), -np.inf, np.array(cut_ub)))
        res = milp(c, constraints=cons, integrality=np.ones(m),
                   bounds=Bounds(0, 1), options={"time_limit": remaining})
        if res.x is None:
            return None

        succ = {}
        for k in np.flatnonzero(res.x > 0.5):
            succ[int(I[k])] = int(J[k])
        seen, cycles = set(), []
        for s in range(n):
            if s in seen:
                continue
            cyc, v = [], s
            while v not in seen:
                seen.add(v)
                cyc.append(v)
                v = succ[v]
            cycles.append(cyc)
        if len(cycles) == 1:
            cyc = cycles[0]
            k = cyc.index(0)
            return cyc[k:] + cyc[:k], (res.status == 0)

        for cyc in cycles:
            inS = np.zeros(n, dtype=bool)
            inS[cyc] = True
            cut_rows.append(sp.csr_matrix((inS[I] & inS[J]).astype(float)))
            cut_ub.append(len(cyc) - 1)


def solve_tsp(d, time_limit_s=None):
    """Returns (tour, info). tour is a list of matrix indices starting at 0."""
    if time_limit_s is None:
        time_limit_s = EXACT_TIME_LIMIT_S
    n = len(d)
    if n == 1:
        return [0], {"optimal": True, "method": "trivial", "heuristic_m": 0.0, "final_m": 0.0}

    tour = tsp_or_opt(d, tsp_nearest_neighbor(d))
    info = {"heuristic_m": tour_cost(d, tour), "optimal": False, "method": "heuristic"}

    exact = tsp_exact(d, time_limit_s)
    if exact is not None:
        t_x, proven = exact
        if tour_cost(d, t_x) <= info["heuristic_m"] + 1e-6:
            tour, info["method"] = t_x, "exact"
        info["optimal"] = bool(proven)
    info["final_m"] = tour_cost(d, tour)
    return tour, info


# --------------------------------------------------------------------------- #
# 3. Pipeline for one request
# --------------------------------------------------------------------------- #
def edge_coords(D, u, v, geom):
    if geom is None:
        return [(D.nodes[u]["y"], D.nodes[u]["x"]), (D.nodes[v]["y"], D.nodes[v]["x"])]
    coords = [(lat, lon) for lon, lat in geom.coords]
    uy, ux = D.nodes[u]["y"], D.nodes[u]["x"]
    d_start = (coords[0][0] - uy) ** 2 + (coords[0][1] - ux) ** 2
    d_end = (coords[-1][0] - uy) ** 2 + (coords[-1][1] - ux) ** 2
    return coords[::-1] if d_end < d_start else coords


def run_tsp(net, schools, lat, lon, radius_km):
    from scipy.sparse.csgraph import dijkstra

    D = net.D
    base_i, snap_km = net.nearest(lat, lon)
    if snap_km > MAX_SNAP_KM:
        raise ValueError("No road found near that point. Click inside Surabaya.")
    base_id = net.ids[base_i]
    by, bx = net.lat[base_i], net.lon[base_i]

    # schools within x km (straight-line, from the SPPG), grouped by road node
    groups = {}            # road-node index -> list of schools
    for s in schools:
        if haversine_km(by, bx, s["lat"], s["lon"]) <= radius_km:
            groups.setdefault(s["node"], []).append(s)
    at_hub = groups.pop(base_i, [])
    if not groups and not at_hub:
        raise ValueError("No schools within that radius. Try a larger radius or another location.")
    if not groups:
        raise ValueError("The only school in range is at the SPPG itself - nothing to deliver.")

    stops = [base_i] + list(groups.keys())            # index 0 = SPPG

    # all-pairs road distances between SPPG + schools (directed Dijkstra, C speed)
    t0 = time.perf_counter()
    dist, pred = dijkstra(net.csr, directed=True, indices=stops, return_predecessors=True)
    d = dist[:, stops]
    matrix_ms = (time.perf_counter() - t0) * 1000
    if not np.isfinite(d).all():
        raise ValueError("Some schools cannot be reached on the road network.")

    # TSP
    t1 = time.perf_counter()
    tour, info = solve_tsp(d.tolist())
    tsp_ms = (time.perf_counter() - t1) * 1000

    # expand into real road paths, leg by leg
    order = tour + [0]
    legs, total_m, total_s, cum = [], 0.0, 0.0, 0.0
    for a, b in zip(order[:-1], order[1:]):
        src, dst = stops[a], stops[b]
        path = [dst]
        while path[-1] != src:
            path.append(int(pred[a][path[-1]]))
        path.reverse()
        coords, leg_m, leg_s = [], 0.0, 0.0
        for x, y in zip(path[:-1], path[1:]):
            u, v = net.ids[x], net.ids[y]
            e = D[u][v]
            leg_m += e["length"]
            leg_s += e["length"] / (e["speed"] / 3.6)
            ec = edge_coords(D, u, v, e.get("geometry"))
            coords.extend(ec if not coords else ec[1:])
        total_m += leg_m
        total_s += leg_s
        cum += leg_m
        legs.append({"km": leg_m / 1000, "cum_km": cum / 1000,
                     "coords": [[round(p, 6), round(q, 6)] for p, q in coords]})

    out_stops = []
    for k, a in enumerate(order[1:-1], 1):            # visiting order of schools
        grp = groups[stops[a]]
        out_stops.append({"i": k, "name": " + ".join(s["name"] for s in grp),
                          "lat": grp[0]["lat"], "lon": grp[0]["lon"],
                          "leg_km": legs[k - 1]["km"], "cum_km": legs[k - 1]["cum_km"]})

    return {
        "base": {"lat": float(by), "lon": float(bx), "snap_m": snap_km * 1000},
        "radius_km": radius_km,
        "hub_schools": [s["name"] for s in at_hub],
        "stops": out_stops,
        "legs": legs,                                  # legs[k-1] ends at stop k; last returns to SPPG
        "stats": {
            "total_km": total_m / 1000,
            "minutes": total_s / 60,
            "duration": fmt_duration(total_s / 60),
            "tsp_ms": tsp_ms,
            "matrix_ms": matrix_ms,
            "n_schools": sum(len(g) for g in groups.values()) + len(at_hub),
            "n_stops": len(out_stops),
            "optimal": info["optimal"],
            "method": info["method"],
            "heuristic_km": info["heuristic_m"] / 1000,
        },
    }


# --------------------------------------------------------------------------- #
# 4. Web app (Flask + Leaflet)
# --------------------------------------------------------------------------- #
PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPPG TSP Simulator</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
 *{box-sizing:border-box} html,body{margin:0;height:100%;font-family:Arial,sans-serif;font-size:13px;color:#222}
 #app{display:flex;height:100%} #side{width:370px;display:flex;flex-direction:column;border-right:1px solid #ccc;background:#fafafa}
 #map{flex:1} .pad{padding:12px 14px} h1{font-size:16px;margin:0 0 4px}
 .muted{color:#666} label{display:block;margin:10px 0 4px;font-weight:bold}
 input[type=range]{width:100%} button{width:100%;padding:10px;border:0;border-radius:6px;background:#0d6efd;color:#fff;font-size:14px;font-weight:bold;cursor:pointer}
 button:disabled{background:#999;cursor:wait} #msg{margin-top:8px;min-height:18px} .err{color:#c00}
 #stats{display:none;border-top:1px solid #ddd;border-bottom:1px solid #ddd;background:#fff}
 #stats table{width:100%;border-collapse:collapse} #stats td{padding:2px 0} #stats td:last-child{text-align:right;font-weight:bold}
 #orderhead{display:none;justify-content:space-between;align-items:center;padding:8px 14px 4px}
 #orderhead a{color:#0a58ca;cursor:pointer;text-decoration:underline}
 #order{flex:1;overflow:auto;margin:0;padding:0 8px 12px 8px;list-style:none}
 #order li{padding:4px 6px;border-bottom:1px solid #eee;cursor:pointer;line-height:1.35} #order li:hover{background:#eef6ff}
 .num{background:#0d6efd;color:#fff;border:2px solid #fff;border-radius:50%;width:22px;height:22px;line-height:18px;text-align:center;font-size:11px;font-weight:bold;box-shadow:0 0 3px #000}
 .legend{display:flex;gap:12px;margin-top:8px;align-items:center}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
</style></head>
<body><div id="app">
 <div id="side">
  <div class="pad">
   <h1>SPPG Delivery &ndash; Travelling Salesman Simulator</h1>
   <div class="muted">1. Click anywhere on the map to place the SPPG.<br>2. Set the radius. 3. Press Solve.</div>
   <label>SPPG location</label><div id="coords" class="muted">Not selected</div>
   <label>Delivery radius x: <span id="rv"></span> km</label>
   <input id="radius" type="range" min="__MINR__" max="__MAXR__" step="0.1" value="__DEFR__">
   <div id="count" class="muted"></div>
   <button id="go" disabled style="margin-top:10px">Solve route</button>
   <div id="msg" class="muted"></div>
   <div class="legend"><span><span class="dot" style="background:#0d6efd"></span>School in radius</span>
    <span><span class="dot" style="background:#aaa"></span>Other school</span></div>
   <label style="font-weight:normal"><input type="checkbox" id="arrows" checked> Show direction arrows</label>
  </div>
  <div id="stats" class="pad"><table>
   <tr><td>Total distance</td><td id="s_total"></td></tr>
   <tr><td>Estimated travel time</td><td id="s_time"></td></tr>
   <tr><td>TSP solver time</td><td id="s_tsp"></td></tr>
   <tr><td>Road-distance matrix time</td><td id="s_mat"></td></tr>
   <tr><td>Schools visited</td><td id="s_n"></td></tr>
   <tr><td>Route quality</td><td id="s_opt"></td></tr></table></div>
  <div id="orderhead"><b>School visiting order</b><a id="dl">download .txt</a></div>
  <ol id="order"></ol>
 </div>
 <div id="map"></div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet-polylinedecorator/1.6.0/leaflet.polylineDecorator.min.js"></script>
<script>
const map = L.map('map').setView([__LAT__, __LON__], 12);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}).addTo(map);

const radiusEl = document.getElementById('radius'), rv = document.getElementById('rv');
const go = document.getElementById('go'), msg = document.getElementById('msg'), countEl = document.getElementById('count');
let base = null, baseMarker = null, circle = null, lastData = null;
const dotLayer = L.layerGroup().addTo(map), routeLayer = L.layerGroup().addTo(map), arrowLayer = L.layerGroup().addTo(map);
let dots = [], legLines = [];

function setMsg(t, err){ msg.textContent = t || ''; msg.className = err ? 'err' : 'muted'; }
function radiusKm(){ return parseFloat(radiusEl.value); }
function hav(a, b, c, d){
  const R = 6371, r = Math.PI/180, dl = (c-a)*r, dn = (d-b)*r;
  const x = Math.sin(dl/2)**2 + Math.cos(a*r)*Math.cos(c*r)*Math.sin(dn/2)**2;
  return 2*R*Math.asin(Math.sqrt(x));
}
function esc(s){ return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function refresh(){
  rv.textContent = radiusKm().toFixed(1);
  if(!base){ return; }
  if(circle) map.removeLayer(circle);
  circle = L.circle([base.lat, base.lon], {radius: radiusKm()*1000, color:'crimson', fillOpacity:0.05, interactive:false}).addTo(map);
  let n = 0;
  dots.forEach(d => {
    const inside = hav(base.lat, base.lon, d.lat, d.lon) <= radiusKm();
    if(inside) n++;
    d.m.setStyle(inside ? {color:'#0d6efd', fillColor:'#0d6efd', fillOpacity:0.9, radius:5}
                        : {color:'#aaa', fillColor:'#aaa', fillOpacity:0.6, radius:3});
  });
  countEl.textContent = n + ' school(s) within ' + radiusKm().toFixed(1) + ' km (straight-line)';
}
radiusEl.addEventListener('input', refresh);

fetch('/api/schools').then(r => r.json()).then(list => {
  list.forEach(s => {
    const m = L.circleMarker([s.lat, s.lon], {radius:3, color:'#aaa', fillColor:'#aaa', fillOpacity:0.6, weight:1})
      .bindTooltip(s.name).addTo(dotLayer);
    dots.push({lat: s.lat, lon: s.lon, m});
  });
  refresh();
});

map.on('click', e => {
  base = {lat: e.latlng.lat, lon: e.latlng.lng};
  if(baseMarker) map.removeLayer(baseMarker);
  baseMarker = L.marker(e.latlng).addTo(map).bindTooltip('SPPG');
  document.getElementById('coords').textContent = base.lat.toFixed(5) + ', ' + base.lon.toFixed(5);
  go.disabled = false; setMsg(''); refresh();
});

function drawArrows(){
  arrowLayer.clearLayers();
  if(!document.getElementById('arrows').checked || !window.L.polylineDecorator || !lastData) return;
  lastData.legs.forEach(l => {
    if(l.coords.length < 2) return;
    arrowLayer.addLayer(L.polylineDecorator(L.polyline(l.coords), {patterns:[{
      offset: '10%', repeat: 110,
      symbol: L.Symbol.arrowHead({pixelSize: 7, polygon: false,
        pathOptions: {stroke: true, weight: 2, color: '#064', opacity: 0.9}})}]}));
  });
}
document.getElementById('arrows').addEventListener('change', drawArrows);

go.addEventListener('click', async () => {
  if(!base) return;
  go.disabled = true; setMsg('Solving... (up to ~30 s when there are many schools)');
  try {
    const r = await fetch('/api/solve', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({lat: base.lat, lon: base.lon, radius: radiusKm()})});
    const data = await r.json();
    if(!r.ok) throw new Error(data.error || 'Server error');
    show(data);
    setMsg(data.hub_schools.length ? 'Also served at the SPPG itself: ' + data.hub_schools.join(', ') : '');
  } catch(e){ setMsg(e.message, true); }
  go.disabled = false;
});

const PALETTE = ['#2e7d32', '#1565c0', '#c62828', '#6a1b9a', '#ef6c00', '#00838f', '#ad1457'];

function show(data){
  lastData = data; routeLayer.clearLayers(); legLines = [];
  base = {lat: data.base.lat, lon: data.base.lon};       // snapped to the nearest road node
  if(baseMarker) map.removeLayer(baseMarker);
  baseMarker = L.marker([base.lat, base.lon]).addTo(map).bindTooltip('SPPG (start / end)');
  document.getElementById('coords').textContent =
    base.lat.toFixed(5) + ', ' + base.lon.toFixed(5) + '  (snapped ' + data.base.snap_m.toFixed(0) + ' m to road)';
  refresh();

  data.legs.forEach((l, k) => {
    const pl = L.polyline(l.coords, {color: PALETTE[k % PALETTE.length], weight: 5, opacity: 0.85}).addTo(routeLayer);
    pl.bindTooltip(k < data.stops.length ? 'to #' + (k+1) + ' ' + data.stops[k].name + ' (' + l.km.toFixed(2) + ' km)'
                                         : 'return to SPPG (' + l.km.toFixed(2) + ' km)');
    legLines.push(pl);
  });
  data.stops.forEach(s => {
    L.marker([s.lat, s.lon], {icon: L.divIcon({className:'', html:'<div class="num">' + s.i + '</div>', iconSize:[22,22], iconAnchor:[11,11]})})
      .bindTooltip(s.i + '. ' + s.name).addTo(routeLayer);
  });

  const ol = document.getElementById('order'); ol.innerHTML = '';
  const add = (html, idx) => { const li = document.createElement('li'); li.innerHTML = html;
    if(idx !== null){ li.onclick = () => { const pl = legLines[idx]; map.fitBounds(pl.getBounds(), {maxZoom: 17});
      const c = pl.options.color; pl.setStyle({color:'#000', weight: 9}); setTimeout(() => pl.setStyle({color: c, weight: 5}), 1500); }; }
    ol.appendChild(li); };
  add('<b>0. [START]</b> SPPG', null);
  data.stops.forEach((s, k) => add('<b>' + s.i + '.</b> ' + esc(s.name) +
    ' <span class="muted">+' + s.leg_km.toFixed(2) + ' km (total ' + s.cum_km.toFixed(2) + ')</span>', k));
  const last = data.legs[data.legs.length - 1];
  add('<b>' + (data.stops.length + 1) + '. [END]</b> Return to SPPG <span class="muted">+' + last.km.toFixed(2) + ' km</span>', data.legs.length - 1);

  const st = data.stats;
  document.getElementById('s_total').textContent = st.total_km.toFixed(2) + ' km';
  document.getElementById('s_time').textContent = st.duration;
  document.getElementById('s_tsp').textContent = st.tsp_ms.toFixed(1) + ' ms';
  document.getElementById('s_mat').textContent = st.matrix_ms.toFixed(1) + ' ms';
  document.getElementById('s_n').textContent = st.n_stops + (st.n_schools !== st.n_stops ? ' stops (' + st.n_schools + ' schools)' : '');
  document.getElementById('s_opt').textContent = st.optimal ? 'Shortest possible (proven)' : 'Best found (not proven)';
  document.getElementById('stats').style.display = 'block';
  document.getElementById('orderhead').style.display = 'flex';
  drawArrows();
  map.fitBounds(L.latLngBounds(data.legs.flatMap(l => l.coords)), {padding: [30, 30]});
}

document.getElementById('dl').addEventListener('click', () => {
  if(!lastData) return;
  const st = lastData.stats;
  let t = 'Total distance: ' + st.total_km.toFixed(2) + ' km\\nEstimated time: ' + st.duration +
          '\\nTSP solver time: ' + st.tsp_ms.toFixed(2) + ' ms\\n\\n0. [START] SPPG\\n';
  lastData.stops.forEach(s => { t += s.i + '. ' + s.name + ' (+' + s.leg_km.toFixed(2) + ' km)\\n'; });
  t += (lastData.stops.length + 1) + '. [END] Return to SPPG\\n';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([t], {type: 'text/plain'})); a.download = 'sppg_school_order.txt'; a.click();
});
</script></body></html>
"""


def create_app(net, schools):
    from flask import Flask, Response, jsonify, request

    # snap every school to its nearest road node once, at start-up
    for s in schools:
        s["node"], _ = net.nearest(s["lat"], s["lon"])

    app = Flask(__name__)
    lock = threading.Lock()          # one solve at a time (CPU-bound)
    public = [{"name": s["name"], "lat": s["lat"], "lon": s["lon"]} for s in schools]

    page = (PAGE.replace("__LAT__", str(DEFAULT_CENTER[0]))
                .replace("__LON__", str(DEFAULT_CENTER[1]))
                .replace("__MINR__", str(MIN_RADIUS_KM))
                .replace("__MAXR__", str(MAX_RADIUS_KM))
                .replace("__DEFR__", str(DEFAULT_RADIUS_KM)))

    @app.get("/")
    def home():
        return Response(page, mimetype="text/html")

    @app.get("/api/schools")
    def api_schools():
        return jsonify(public)

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
                return jsonify(run_tsp(net, schools, lat, lon, radius))
        except ValueError as e:
            return jsonify(error=str(e)), 422

    return app


def main():
    G = load_surabaya_graph()
    D = build_digraph(G)
    schools = load_schools()
    app = create_app(RoadNet(D), schools)
    url = f"http://127.0.0.1:{PORT}"
    print(f"[+] Open {url}")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()

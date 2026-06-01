#!/usr/bin/env python3
"""
Generate a self-contained interactive knowledge graph HTML file from value_mappings.yaml.

Usage:
    python3 generate_knowledge_graph.py
    python3 generate_knowledge_graph.py --input value_mappings.yaml --output knowledge_graph.html
"""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

CATEGORIES = [
    "diagnosis",
    "task",
    "suffix",
    "handedness",
    "sex",
    "datatype",
    "sidecar_fields",
    "participant_extra_fields",
]

CAT_COLORS = {
    "diagnosis":               "#4e79a7",
    "task":                    "#f28e2b",
    "suffix":                  "#e15759",
    "handedness":              "#76b7b2",
    "sex":                     "#59a14f",
    "datatype":                "#edc948",
    "sidecar_fields":          "#b07aa1",
    "participant_extra_fields":"#ff9da7",
}

HUB_RADIUS = 520
WARMUP_TICKS = 300


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_hub_positions() -> dict:
    positions = {}
    n = len(CATEGORIES)
    for i, cat in enumerate(CATEGORIES):
        angle = -math.pi / 2 + 2 * math.pi * i / n
        positions[cat] = (round(HUB_RADIUS * math.cos(angle), 2),
                          round(HUB_RADIUS * math.sin(angle), 2))
    return positions


def compute_depths(entries: dict) -> dict:
    """Iterative post-order depth computation for a DAG. Returns {key: depth}."""
    depths = {}
    # Collect all keys first
    all_keys = set(entries.keys())

    for start_key in all_keys:
        if start_key in depths:
            continue
        # Iterative DFS with explicit stack to avoid Python recursion limits
        stack = [(start_key, False)]
        visiting = set()
        while stack:
            key, returning = stack.pop()
            if key not in all_keys:
                continue
            if returning:
                visiting.discard(key)
                val = entries[key]
                parents = [p for p in (val.get("broader") or []) if p in all_keys]
                if parents:
                    depths[key] = max((depths.get(p, 0) for p in parents), default=0) + 1
                else:
                    depths[key] = 0
            else:
                if key in depths:
                    continue
                if key in visiting:
                    # Cycle — assign depth 0 to break it
                    depths[key] = 0
                    continue
                visiting.add(key)
                stack.append((key, True))
                val = entries[key]
                parents = [p for p in (val.get("broader") or []) if p in all_keys]
                for p in parents:
                    if p not in depths:
                        stack.append((p, False))

    # Any key not reached gets depth 0
    for key in all_keys:
        if key not in depths:
            depths[key] = 0
    return depths


def extract_synonyms(val: dict) -> list:
    raw = val.get("synonyms") or []
    terms = []
    for s in raw:
        if isinstance(s, dict):
            t = s.get("term")
            if t:
                terms.append(str(t))
        elif s:
            terms.append(str(s))
    return terms


def build_graph(data: dict) -> dict:
    hub_positions = compute_hub_positions()
    nodes = []
    links = []

    # Hub nodes
    for cat in CATEGORIES:
        x, y = hub_positions[cat]
        nodes.append({
            "id": f"cat_{cat}",
            "type": "hub",
            "cat": cat,
            "label": cat.replace("_", " "),
            "depth": -1,
            "fx": x,
            "fy": y,
        })

    # Entry nodes + edges
    for cat in CATEGORIES:
        if cat not in data:
            continue
        entries = data[cat]
        if not isinstance(entries, dict):
            continue

        depths = compute_depths(entries)

        for key, val in entries.items():
            if not isinstance(val, dict):
                continue

            depth = depths.get(key, 0)
            broader = [p for p in (val.get("broader") or []) if p in entries]
            synonyms = extract_synonyms(val)
            codes = [str(c) for c in (val.get("codes") or [])]
            desc = val.get("description") or ""

            nodes.append({
                "id": key,
                "type": "node",
                "cat": cat,
                "label": val.get("label") or key,
                "depth": depth,
                "g": bool(val.get("is_group")),
                "desc": desc[:300],  # truncate to keep JSON lean
                "syn": synonyms[:20],
                "codes": codes[:15],
            })

            # broader edges (child → parent, within same category)
            for parent in broader:
                links.append({"s": key, "t": parent})

            # hub-root edges for depth-0 nodes
            if depth == 0:
                links.append({"s": key, "t": f"cat_{cat}", "hub": True})

    # Build cats metadata
    cats_meta = {}
    for cat in CATEGORIES:
        x, y = hub_positions[cat]
        count = len(data.get(cat) or {})
        cats_meta[cat] = {
            "color": CAT_COLORS[cat],
            "fx": x,
            "fy": y,
            "count": count,
        }

    from datetime import date
    return {
        "meta": {
            "total": len(nodes),
            "edges": len(links),
            "generated": date.today().isoformat(),
        },
        "cats": cats_meta,
        "nodes": nodes,
        "links": links,
    }


# ─── HTML Template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BIDS-Eye Knowledge Graph</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{display:flex;height:100vh;background:#0f1117;color:#e0e0e0;font-family:system-ui,sans-serif;overflow:hidden}
#loading{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#0f1117;z-index:200;gap:16px;font-size:16px;color:#aaa}
.spinner{width:40px;height:40px;border:3px solid #333;border-top-color:#4e79a7;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#sidebar{width:230px;flex-shrink:0;overflow-y:auto;background:#13161f;border-right:1px solid #2a2d3a;padding:14px 12px;display:flex;flex-direction:column;gap:10px}
#sidebar h2{font-size:14px;font-weight:600;color:#c0c8e0;letter-spacing:.04em}
#search{width:100%;padding:7px 10px;background:#1e2130;border:1px solid #3a3d50;border-radius:6px;color:#e0e0e0;font-size:13px;outline:none}
#search:focus{border-color:#4e79a7}
#search-count{font-size:11px;color:#7a8099;min-height:16px}
#filters{display:flex;flex-direction:column;gap:3px}
.filter-row{display:flex;align-items:center;gap:8px;padding:4px 6px;border-radius:5px;cursor:pointer;font-size:12px;user-select:none}
.filter-row:hover{background:#1e2130}
.filter-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.filter-count{margin-left:auto;color:#666;font-size:11px}
#stats{font-size:11px;color:#666;line-height:1.7}
#reset-btn{margin-top:auto;padding:7px;background:#1e2130;border:1px solid #3a3d50;border-radius:6px;color:#aaa;font-size:12px;cursor:pointer;width:100%}
#reset-btn:hover{background:#2a2d3a;color:#e0e0e0}
#graph-container{flex:1;position:relative;overflow:hidden}
svg#graph{width:100%;height:100%;display:block}
#info-panel{position:fixed;right:0;top:0;width:310px;height:100vh;background:#13161f;border-left:1px solid #2a2d3a;overflow-y:auto;padding:16px;transform:translateX(100%);transition:transform .25s ease;z-index:20;display:flex;flex-direction:column;gap:10px}
#info-panel.open{transform:translateX(0)}
#close-panel{align-self:flex-end;background:none;border:none;color:#666;font-size:18px;cursor:pointer;padding:2px 6px;line-height:1}
#close-panel:hover{color:#e0e0e0}
.cat-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;color:#000;font-weight:600;margin-bottom:4px}
#info-panel h2{font-size:15px;font-weight:600;color:#e0e0e0;line-height:1.3}
.node-id{font-size:11px;color:#555;font-family:monospace}
.info-desc{font-size:12px;color:#9a9db0;line-height:1.5}
#info-panel h4{font-size:11px;font-weight:600;color:#7a8099;text-transform:uppercase;letter-spacing:.06em;margin-top:4px}
.info-chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{background:#1e2130;border:1px solid #3a3d50;border-radius:4px;padding:2px 6px;font-size:11px;color:#aab}
code.code-chip{background:#1e2130;border:1px solid #3a3d50;border-radius:3px;padding:1px 5px;font-size:10px;font-family:monospace;color:#8ab4f8}
.syn-list{font-size:12px;color:#9a9db0;line-height:1.7;list-style:none}
.syn-list li::before{content:"·";margin-right:6px;color:#555}
.children-list{font-size:12px;color:#9a9db0}
.children-list span{cursor:pointer;color:#7ab0e0}
.children-list span:hover{text-decoration:underline}
#tooltip{position:fixed;pointer-events:none;background:rgba(15,17,23,.96);border:1px solid #3a3d50;border-radius:7px;padding:8px 12px;font-size:12px;max-width:230px;z-index:100;display:none;line-height:1.5}
.tt-label{font-weight:600;color:#e0e0e0;margin-bottom:2px}
.tt-cat{font-size:10px}
.tt-syn{font-size:11px;color:#8a8d9f;margin-top:2px}
</style>
</head>
<body>
<div id="loading"><div class="spinner"></div><span>Building graph…</span></div>
<div id="sidebar">
  <h2>BIDS-Eye Ontology</h2>
  <input id="search" type="search" placeholder="Search nodes, synonyms, codes…">
  <div id="search-count"></div>
  <div id="filters"></div>
  <hr style="border-color:#2a2d3a;margin:4px 0">
  <div id="stats"></div>
  <button id="reset-btn">Reset View</button>
</div>
<div id="graph-container">
  <svg id="graph">
    <defs id="defs"></defs>
    <g id="zoom-layer">
      <g id="links-layer"></g>
      <g id="nodes-layer"></g>
      <g id="labels-layer"></g>
    </g>
  </svg>
</div>
<div id="info-panel"></div>
<div id="tooltip"></div>

<script>const GRAPH_DATA = __GRAPH_DATA__;</script>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script>
(function() {
'use strict';

const CONFIG = {
  hubR: 28, groupR: 7, leafR: 4,
  zoomMin: 0.06, zoomMax: 20,
  // zoom thresholds: stay sparse until user deliberately zooms in
  zoomLevels: { roots: 0.55, groups: 1.6, leaves: 4.0 },
  // labels only appear when quite zoomed in (hubs always visible)
  labelLevels: { hubs: 0, groups: 3.0, leaves: 6.0 },
  linkDist: [160, 90, 65, 50, 45],
  charge: -45, chargeMax: 300,
  centerStrength: 0.04,
  collidePad: 8, collideIter: 2,
  alphaDecay: 0.018, velocityDecay: 0.38,
  warmup: 300,
  viewportPad: 350,
  searchDebounce: 180,
};

// ── State ────────────────────────────────────────────────────────────────────
let currentTransform = d3.zoomIdentity;
let selectedNode = null;
let isolationHood = null;  // Set of node IDs to show in isolation mode (null = off)
let hiddenCats = new Set();
let searchMatches = new Set();
let isSearchActive = false;
let rafPending = false;

// ── Lookups ──────────────────────────────────────────────────────────────────
let nodeById, parentsOf, childrenOf;

function buildLookups() {
  nodeById = new Map();
  parentsOf = new Map();
  childrenOf = new Map();
  GRAPH_DATA.nodes.forEach(n => {
    nodeById.set(n.id, n);
    if (!parentsOf.has(n.id)) parentsOf.set(n.id, []);
    if (!childrenOf.has(n.id)) childrenOf.set(n.id, []);
  });
  GRAPH_DATA.links.forEach(l => {
    const sid = l.s, tid = l.t;
    if (!l.hub) {
      (parentsOf.get(sid) || []).push(tid);
      (childrenOf.get(tid) || []).push(sid);
    }
  });
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function catColor(cat) {
  return (GRAPH_DATA.cats[cat] || {}).color || '#888';
}

function nodeR(d) {
  return d.type === 'hub' ? CONFIG.hubR : d.g ? CONFIG.groupR : CONFIG.leafR;
}

function zoomLevel(k) {
  if (k < CONFIG.zoomLevels.roots)  return 'overview';
  if (k < CONFIG.zoomLevels.groups) return 'roots';
  if (k < CONFIG.zoomLevels.leaves) return 'groups';
  return 'leaves';
}

function viewport(t) {
  const svg = document.getElementById('graph');
  const W = svg.clientWidth, H = svg.clientHeight;
  const pad = CONFIG.viewportPad;
  return {
    x1: (-t.x - pad) / t.k,
    y1: (-t.y - pad) / t.k,
    x2: (W - t.x + pad) / t.k,
    y2: (H - t.y + pad) / t.k,
  };
}

function inViewport(d, vp) {
  return d.x >= vp.x1 && d.x <= vp.x2 && d.y >= vp.y1 && d.y <= vp.y2;
}

function isNodeVisible(d, zl, vp) {
  if (hiddenCats.has(d.cat)) return false;
  // Isolation mode: only show the selected node and its direct neighbours
  if (isolationHood) return isolationHood.has(d.id);
  if (d.type === 'hub') return true;
  if (zl === 'overview') return false;
  if (zl === 'roots' && d.depth !== 0) return false;
  if (zl === 'groups' && d.depth > 1) return false;
  return inViewport(d, vp);
}

function isLinkVisible(l, zl, vp) {
  if (isolationHood) {
    const sid = l.source.id ?? l.source;
    const tid = l.target.id ?? l.target;
    return isolationHood.has(sid) && isolationHood.has(tid);
  }
  const src = l.source, tgt = l.target;
  if (!isNodeVisible(src, zl, vp) || !isNodeVisible(tgt, zl, vp)) return false;
  return true;
}

function isLabelVisible(d, k) {
  if (hiddenCats.has(d.cat)) return false;
  // In isolation mode always show labels for visible nodes
  if (isolationHood) return isolationHood.has(d.id);
  if (d.type === 'hub') return k >= CONFIG.labelLevels.hubs;
  if (d.g) return k >= CONFIG.labelLevels.groups;
  return k >= CONFIG.labelLevels.leaves;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function getNeighborhood(d) {
  const set = new Set([d.id]);
  (parentsOf.get(d.id) || []).forEach(id => set.add(id));
  (childrenOf.get(d.id) || []).forEach(id => set.add(id));
  // Also include the hub for category nodes
  if (d.type !== 'hub') set.add('cat_' + d.cat);
  return set;
}

// ── SVG & D3 setup ───────────────────────────────────────────────────────────
const svg = d3.select('#graph');
const zoomLayer = d3.select('#zoom-layer');
let linkEl, nodeEl, labelEl;

function svgW() { return document.getElementById('graph').clientWidth; }
function svgH() { return document.getElementById('graph').clientHeight; }

function injectMarkers() {
  const defs = d3.select('#defs');
  Object.entries(GRAPH_DATA.cats).forEach(([cat, meta]) => {
    defs.append('marker')
      .attr('id', `arrow-${cat}`)
      .attr('viewBox', '0 0 6 6')
      .attr('refX', 5).attr('refY', 3)
      .attr('markerWidth', 5).attr('markerHeight', 5)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,0 L6,3 L0,6 Z')
      .attr('fill', meta.color)
      .attr('opacity', 0.75);
  });
}

// ── Simulation ───────────────────────────────────────────────────────────────
let simulation;

function setupSimulation() {
  const nodes = GRAPH_DATA.nodes;
  // D3 forceLink requires source/target fields (not s/t); also needs mutable objects
  const links = GRAPH_DATA.links.map(l => ({ source: l.s, target: l.t, hub: l.hub || false }));

  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links)
      .id(d => d.id)
      .distance(d => {
        if (d.hub) return CONFIG.linkDist[0];
        const src = nodeById.get(
          typeof d.source === 'object' ? d.source.id : d.source
        );
        const depth = src ? src.depth : 0;
        return CONFIG.linkDist[Math.min(depth + 1, CONFIG.linkDist.length - 1)];
      })
      .strength(d => d.hub ? 0.7 : 0.55)
    )
    .force('charge', d3.forceManyBody()
      .strength(d => d.type === 'hub' ? 0 : CONFIG.charge)
      .distanceMax(CONFIG.chargeMax)
    )
    .force('center', d3.forceCenter(0, 0).strength(CONFIG.centerStrength))
    .force('collide', d3.forceCollide()
      .radius(d => nodeR(d) + CONFIG.collidePad)
      .iterations(CONFIG.collideIter)
    )
    .alphaDecay(CONFIG.alphaDecay)
    .velocityDecay(CONFIG.velocityDecay)
    .stop();

  // Fix hub positions
  nodes.forEach(n => {
    if (n.type === 'hub') { n.fx = n.fx; n.fy = n.fy; }
  });

  // Warmup synchronously
  for (let i = 0; i < CONFIG.warmup; i++) simulation.tick();

  return links;
}

// ── Element creation ─────────────────────────────────────────────────────────
function createElements(resolvedLinks) {
  linkEl = d3.select('#links-layer')
    .selectAll('line')
    .data(resolvedLinks)
    .join('line')
    .attr('stroke', d => {
      const src = d.source;
      return catColor(typeof src === 'object' ? src.cat : '');
    })
    .attr('stroke-width', d => d.hub ? 1.2 : 0.7)
    .attr('stroke-dasharray', d => d.hub ? '4 3' : null)
    .attr('marker-end', d => {
      if (d.hub) return null;
      const src = d.source;
      const cat = typeof src === 'object' ? src.cat : '';
      return `url(#arrow-${cat})`;
    })
    .attr('stroke-opacity', 0.45);

  nodeEl = d3.select('#nodes-layer')
    .selectAll('circle')
    .data(GRAPH_DATA.nodes)
    .join('circle')
    .attr('r', nodeR)
    .attr('fill', d => catColor(d.cat))
    .attr('stroke', d => d.type === 'hub' ? '#fff' : 'none')
    .attr('stroke-width', d => d.type === 'hub' ? 2 : 0)
    .style('cursor', 'pointer')
    .on('click', onNodeClick)
    .on('mouseenter', onNodeHover)
    .on('mouseleave', onNodeLeave);

  labelEl = d3.select('#labels-layer')
    .selectAll('text')
    .data(GRAPH_DATA.nodes)
    .join('text')
    .text(d => d.label || d.id)
    .attr('font-size', d => d.type === 'hub' ? 12 : d.g ? 9 : 8)
    .attr('fill', d => d.type === 'hub' ? '#dde' : '#9a9db0')
    .attr('text-anchor', 'middle')
    .attr('dy', d => d.type === 'hub' ? nodeR(d) + 14 : -(nodeR(d) + 4))
    .style('pointer-events', 'none')
    .style('user-select', 'none');
}

// ── Tick ─────────────────────────────────────────────────────────────────────
function ticked() {
  const t = currentTransform;
  const k = t.k;
  const zl = zoomLevel(k);
  const vp = viewport(t);

  linkEl
    .attr('x1', d => d.source.x)
    .attr('y1', d => d.source.y)
    .attr('x2', d => {
      const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
      const dist = Math.sqrt(dx*dx + dy*dy) || 1;
      const r = nodeR(d.target) + 5;
      return d.target.x - dx / dist * r;
    })
    .attr('y2', d => {
      const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
      const dist = Math.sqrt(dx*dx + dy*dy) || 1;
      const r = nodeR(d.target) + 5;
      return d.target.y - dy / dist * r;
    })
    .style('display', d => isLinkVisible(d, zl, vp) ? null : 'none');

  nodeEl
    .attr('cx', d => d.x)
    .attr('cy', d => d.y)
    .style('display', d => isNodeVisible(d, zl, vp) ? null : 'none');

  labelEl
    .attr('x', d => d.x)
    .attr('y', d => d.y)
    .style('display', d => isLabelVisible(d, k) ? null : 'none');
}

// ── Zoom ─────────────────────────────────────────────────────────────────────
let zoom;

function setupZoom() {
  zoom = d3.zoom()
    .scaleExtent([CONFIG.zoomMin, CONFIG.zoomMax])
    .on('zoom', event => {
      currentTransform = event.transform;
      zoomLayer.attr('transform', event.transform);
      if (!rafPending) {
        rafPending = true;
        requestAnimationFrame(() => { ticked(); rafPending = false; });
      }
    });

  svg.call(zoom);
  // Show hubs at a comfortable size on load
  svg.call(zoom.transform,
    d3.zoomIdentity.translate(svgW() / 2, svgH() / 2).scale(0.5));
}

function resetView() {
  svg.transition().duration(500)
    .call(zoom.transform,
      d3.zoomIdentity.translate(svgW() / 2, svgH() / 2).scale(0.5));
  clearSelection();
}

// ── Interaction: click / hover ────────────────────────────────────────────────
function onNodeClick(event, d) {
  event.stopPropagation();
  if (selectedNode === d) { clearSelection(); return; }
  selectedNode = d;
  isolationHood = getNeighborhood(d);

  // Highlight selected node ring
  nodeEl
    .attr('stroke', n => n === d ? '#fff' : (n.type === 'hub' && isolationHood.has(n.id) ? '#fff' : 'none'))
    .attr('stroke-width', n => n === d ? 3 : (n.type === 'hub' && isolationHood.has(n.id) ? 2 : 0));

  ticked(); // visibility (display:none) is handled entirely by ticked() via isolationHood
  showInfoPanel(d);

  // Zoom to node if not a hub
  if (d.type !== 'hub') {
    const targetK = Math.max(currentTransform.k, 2.5);
    svg.transition().duration(600)
      .call(zoom.transform,
        d3.zoomIdentity.translate(svgW() / 2, svgH() / 2).scale(targetK)
          .translate(-d.x, -d.y));
  }
}

svg.on('click', () => clearSelection());

function clearSelection() {
  selectedNode = null;
  isolationHood = null;
  nodeEl
    .attr('stroke', n => n.type === 'hub' ? '#fff' : 'none')
    .attr('stroke-width', n => n.type === 'hub' ? 2 : 0);
  linkEl.attr('stroke-opacity', 0.45);
  document.getElementById('info-panel').classList.remove('open');
  ticked();
  if (isSearchActive) applySearchVisuals();
}

let ttTimeout;
function onNodeHover(event, d) {
  if (d.type === 'hub') return;
  clearTimeout(ttTimeout);
  const tt = document.getElementById('tooltip');
  const kids = (childrenOf.get(d.id) || []).length;
  const syn0 = (d.syn || [])[0] || '';
  tt.innerHTML =
    `<div class="tt-label">${escHtml(d.label || d.id)}</div>` +
    `<div class="tt-cat" style="color:${catColor(d.cat)}">${d.cat}</div>` +
    (syn0 ? `<div class="tt-syn">${escHtml(syn0)}</div>` : '') +
    (kids ? `<div class="tt-syn">${kids} child node${kids>1?'s':''}</div>` : '');
  tt.style.display = 'block';
  moveTT(event);
  nodeEl.filter(n => n === d).attr('r', nodeR(d) + 2);
}

function onNodeLeave(event, d) {
  ttTimeout = setTimeout(() => {
    document.getElementById('tooltip').style.display = 'none';
  }, 100);
  nodeEl.filter(n => n === d).attr('r', nodeR(d));
}

svg.on('mousemove', event => moveTT(event));

function moveTT(event) {
  const tt = document.getElementById('tooltip');
  if (tt.style.display === 'none') return;
  const W = window.innerWidth, H = window.innerHeight;
  let x = event.clientX + 14, y = event.clientY + 14;
  if (x + 240 > W) x = event.clientX - 240;
  if (y + 100 > H) y = event.clientY - 100;
  tt.style.left = x + 'px';
  tt.style.top = y + 'px';
}

// ── Info Panel ────────────────────────────────────────────────────────────────
function showInfoPanel(d) {
  const panel = document.getElementById('info-panel');
  const parents = (parentsOf.get(d.id) || []).map(id => {
    const n = nodeById.get(id);
    return n ? `<span class="children-list"><span onclick="zoomTo('${id}')">${escHtml(n.label||id)}</span></span>` : escHtml(id);
  });
  const kids = childrenOf.get(d.id) || [];
  const kidsHtml = kids.slice(0, 12).map(id => {
    const n = nodeById.get(id);
    return `<span onclick="zoomTo('${id}')" style="cursor:pointer;color:#7ab0e0;font-size:12px">${escHtml((n&&n.label)||id)}</span>`;
  }).join(', ') + (kids.length > 12 ? ` <span style="color:#666">+${kids.length - 12} more</span>` : '');

  panel.innerHTML =
    `<button id="close-panel">✕</button>` +
    `<span class="cat-badge" style="background:${catColor(d.cat)};color:#000">${d.cat}</span>` +
    `<h2>${escHtml(d.label || d.id)}</h2>` +
    `<div class="node-id">${escHtml(d.id)}</div>` +
    (d.desc ? `<div class="info-desc">${escHtml(d.desc)}</div>` : '') +
    (parents.length ? `<h4>Parents (${parents.length})</h4><div class="children-list">${parents.join(', ')}</div>` : '') +
    (kids.length ? `<h4>Children (${kids.length})</h4><div class="children-list">${kidsHtml}</div>` : '') +
    (d.syn && d.syn.length ? `<h4>Synonyms</h4><ul class="syn-list">${d.syn.slice(0,14).map(s=>`<li>${escHtml(s)}</li>`).join('')}</ul>` : '') +
    (d.codes && d.codes.length ? `<h4>Codes</h4><div class="info-chips">${d.codes.slice(0,12).map(c=>`<code class="code-chip">${escHtml(c)}</code>`).join('')}</div>` : '');

  panel.classList.add('open');
  panel.querySelector('#close-panel').addEventListener('click', () => {
    panel.classList.remove('open');
    clearSelection();
  });
}

window.zoomTo = function(id) {
  const n = nodeById.get(id);
  if (!n) return;
  clearSelection();
  setTimeout(() => onNodeClick({ stopPropagation: ()=>{} }, n), 50);
};

// ── Search ────────────────────────────────────────────────────────────────────
let searchTimer;
document.getElementById('search').addEventListener('input', e => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => applySearch(e.target.value.trim()), CONFIG.searchDebounce);
});
document.getElementById('search').addEventListener('keydown', e => {
  if (e.key === 'Escape') { e.target.value = ''; applySearch(''); }
  if (e.key === 'Enter') {
    const first = [...searchMatches][0];
    if (first) {
      const n = nodeById.get(first);
      if (n) zoomTo(first);
    }
  }
});

function applySearch(q) {
  const countEl = document.getElementById('search-count');
  if (!q) {
    isSearchActive = false;
    searchMatches = new Set();
    countEl.textContent = '';
    applySearchVisuals();
    return;
  }
  isSearchActive = true;
  const ql = q.toLowerCase();
  searchMatches = new Set();
  GRAPH_DATA.nodes.forEach(n => {
    if (n.type === 'hub') return;
    const hit =
      n.id.includes(ql) ||
      (n.label || '').toLowerCase().includes(ql) ||
      (n.syn || []).some(s => s.toLowerCase().includes(ql)) ||
      (n.codes || []).some(c => String(c).toLowerCase().includes(ql)) ||
      (n.desc || '').toLowerCase().includes(ql);
    if (hit) searchMatches.add(n.id);
  });
  countEl.textContent = searchMatches.size ? `${searchMatches.size} match${searchMatches.size>1?'es':''}` : 'No matches';
  applySearchVisuals();
}

function applySearchVisuals() {
  if (!nodeEl) return;
  if (!isSearchActive || searchMatches.size === 0) {
    nodeEl.attr('opacity', 1)
      .attr('stroke', n => n.type === 'hub' ? '#fff' : 'none')
      .attr('stroke-width', n => n.type === 'hub' ? 2 : 0);
    return;
  }
  nodeEl
    .attr('opacity', n => searchMatches.has(n.id) || n.type === 'hub' ? 1 : 0.06)
    .attr('stroke', n => searchMatches.has(n.id) ? '#ffdd00' : (n.type === 'hub' ? '#fff' : 'none'))
    .attr('stroke-width', n => searchMatches.has(n.id) ? 3 : (n.type === 'hub' ? 2 : 0));
}

// ── Category Filters ──────────────────────────────────────────────────────────
function buildFilterUI() {
  const container = document.getElementById('filters');
  Object.entries(GRAPH_DATA.cats).forEach(([cat, meta]) => {
    const label = document.createElement('label');
    label.className = 'filter-row';
    label.innerHTML =
      `<input type="checkbox" checked data-cat="${cat}" style="accent-color:${meta.color}">` +
      `<span class="filter-dot" style="background:${meta.color}"></span>` +
      `<span>${cat.replace(/_/g,' ')}</span>` +
      `<span class="filter-count">${meta.count}</span>`;
    label.querySelector('input').addEventListener('change', e => {
      if (e.target.checked) hiddenCats.delete(cat);
      else hiddenCats.add(cat);
      ticked();
    });
    container.appendChild(label);
  });
}

function buildStats() {
  const m = GRAPH_DATA.meta;
  document.getElementById('stats').innerHTML =
    `<div>${m.total} nodes &nbsp;·&nbsp; ${m.edges} edges</div>` +
    `<div>Generated ${m.generated}</div>`;
}

// ── Reset button ──────────────────────────────────────────────────────────────
document.getElementById('reset-btn').addEventListener('click', resetView);

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    buildLookups();
    injectMarkers();
    const resolvedLinks = setupSimulation();
    createElements(resolvedLinks);
    setupZoom();
    buildFilterUI();
    buildStats();
    simulation.on('tick', ticked).restart();
    document.getElementById('loading').style.display = 'none';
    ticked();
  }, 60);
});

})();
</script>
</body>
</html>
"""


def emit_html(graph: dict, output_path: str) -> None:
    json_str = json.dumps(graph, separators=(",", ":"), ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__GRAPH_DATA__", json_str)
    Path(output_path).write_text(html, encoding="utf-8")
    size_kb = Path(output_path).stat().st_size // 1024
    print(f"Generated {output_path}")
    print(f"  Nodes:  {graph['meta']['total']}")
    print(f"  Edges:  {graph['meta']['edges']}")
    print(f"  Size:   {size_kb} KB")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="value_mappings.yaml",
                        help="Path to value_mappings.yaml")
    parser.add_argument("--output", default="knowledge_graph.html",
                        help="Output HTML file path")
    args = parser.parse_args()

    print(f"Loading {args.input}…")
    try:
        data = load_yaml(args.input)
    except FileNotFoundError:
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading YAML: {e}", file=sys.stderr)
        sys.exit(1)

    print("Building graph…")
    graph = build_graph(data)
    emit_html(graph, args.output)


if __name__ == "__main__":
    main()

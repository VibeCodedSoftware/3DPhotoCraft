//! 3D extrusion (Photoshop's 3D › New 3D Extrusion from Selected Layer / Selection / Path, the
//! old Repoussé): outlines → capped, bevelled solid with deformations.
//!
//! ```text
//! coverage mask ──marching squares──► closed loops (pixels) ──Douglas–Peucker──► contours
//! contours ──nesting (even = outline, odd = hole)──► shapes ──ear clipping──► caps
//! caps + bevel rings + side walls ──twist / taper / bend / shear──► mesh (5 material slots)
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::math::{Vec3, clampf, v3};
use crate::mesh::Mesh;

/// A closed polygon in the XY plane (no repeated end point).
pub type Contour = Vec<[f32; 2]>;

/// Material slots of an extrusion, in Photoshop's order.
pub const EXTRUSION_SLOTS: [&str; 5] = ["Front Inflation Material", "Front Bevel Material", "Extrusion Material", "Back Bevel Material", "Back Inflation Material"];
pub const SLOT_FRONT: u16 = 0;
pub const SLOT_FRONT_BEVEL: u16 = 1;
pub const SLOT_SIDES: u16 = 2;
pub const SLOT_BACK_BEVEL: u16 = 3;
pub const SLOT_BACK: u16 = 4;

/// Bevel profile (Photoshop's Cap › Contour).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BevelContour {
    #[default]
    Linear,
    /// Quarter circle: rounded edges.
    Round,
    /// Inverted quarter circle.
    Cove,
    /// Two steps.
    Ring,
}

impl BevelContour {
    pub const ALL: [BevelContour; 4] = [BevelContour::Linear, BevelContour::Round, BevelContour::Cove, BevelContour::Ring];
    pub fn id(self) -> &'static str {
        match self {
            BevelContour::Linear => "linear",
            BevelContour::Round => "round",
            BevelContour::Cove => "cove",
            BevelContour::Ring => "ring",
        }
    }
    /// `(inset, height)` fractions at profile parameter `t` (0 = outer edge, 1 = cap).
    fn at(self, t: f32) -> (f32, f32) {
        let a = t * std::f32::consts::FRAC_PI_2;
        match self {
            BevelContour::Linear => (t, t),
            BevelContour::Round => (1.0 - a.cos(), a.sin()),
            BevelContour::Cove => (a.sin(), 1.0 - a.cos()),
            BevelContour::Ring => {
                let s = (t * 2.0).min(1.0);
                let h = if t < 0.5 { 0.0 } else { (t - 0.5) * 2.0 };
                (s, h)
            }
        }
    }
}

/// Which caps carry the bevel (Photoshop's Cap › Sides).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BevelSides {
    #[default]
    Front,
    Back,
    FrontAndBack,
}

/// Extrusion and cap parameters (Photoshop's Deform and Cap properties).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ExtrudeParams {
    /// Extrusion depth, in world units (the outline's height is 2 units when it fills the canvas).
    pub depth: f32,
    /// Bevel width, in world units.
    pub bevel_width: f32,
    /// Bevel height as a fraction of the bevel width.
    pub bevel_angle: f32,
    pub bevel_segments: u32,
    pub bevel_contour: BevelContour,
    pub bevel_sides: BevelSides,
    /// Twist around the extrusion axis over the depth, degrees.
    pub twist: f32,
    /// Scale of the back relative to the front, percent.
    pub taper: f32,
    /// Bend over the depth, degrees (horizontal, vertical).
    pub bend_x: f32,
    pub bend_y: f32,
    /// Shear over the depth, degrees (horizontal, vertical).
    pub shear_x: f32,
    pub shear_y: f32,
    /// Smoothing angle for normals, degrees.
    pub smoothing: f32,
}

impl Default for ExtrudeParams {
    fn default() -> Self {
        Self {
            depth: 0.35,
            bevel_width: 0.02,
            bevel_angle: 1.0,
            bevel_segments: 3,
            bevel_contour: BevelContour::Round,
            bevel_sides: BevelSides::Front,
            twist: 0.0,
            taper: 100.0,
            bend_x: 0.0,
            bend_y: 0.0,
            shear_x: 0.0,
            shear_y: 0.0,
            smoothing: 35.0,
        }
    }
}

impl ExtrudeParams {
    /// Clamp every field into a usable range.
    pub fn sanitized(&self) -> ExtrudeParams {
        ExtrudeParams {
            depth: clampf(self.depth, 0.0, 50.0),
            bevel_width: clampf(self.bevel_width, 0.0, 5.0),
            bevel_angle: clampf(self.bevel_angle, 0.0, 10.0),
            bevel_segments: self.bevel_segments.clamp(1, 16),
            bevel_contour: self.bevel_contour,
            bevel_sides: self.bevel_sides,
            twist: clampf(self.twist, -3600.0, 3600.0),
            taper: clampf(self.taper, 0.0, 1000.0),
            bend_x: clampf(self.bend_x, -360.0, 360.0),
            bend_y: clampf(self.bend_y, -360.0, 360.0),
            shear_x: clampf(self.shear_x, -80.0, 80.0),
            shear_y: clampf(self.shear_y, -80.0, 80.0),
            smoothing: clampf(self.smoothing, 0.0, 180.0),
        }
    }
}

// ---------------------------------------------------------------- tracing

/// Trace the outlines of `coverage` (row-major `w × h`, values 0–1) at the 0.5 level with
/// marching squares. Returns closed loops in pixel coordinates (x right, y down). Pixels outside
/// the grid count as empty, so every loop closes.
pub fn trace(coverage: &[f32], w: usize, h: usize) -> Vec<Contour> {
    if w == 0 || h == 0 || coverage.len() < w * h {
        return Vec::new();
    }
    let at = |x: i64, y: i64| -> f32 {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 { 0.0 } else { coverage[y as usize * w + x as usize] }
    };
    // Sample points are pixel centres; cells span between them, with a one-pixel empty border.
    // Edge ids: horizontal edge (x,y)-(x+1,y) = (x,y,0); vertical (x,y)-(x,y+1) = (x,y,1).
    type Edge = (i64, i64, u8);
    let point = |e: Edge| -> [f32; 2] {
        let (x, y, d) = e;
        let (a, b) = if d == 0 { (at(x, y), at(x + 1, y)) } else { (at(x, y), at(x, y + 1)) };
        let t = if (b - a).abs() > 1e-9 { ((0.5 - a) / (b - a)).clamp(0.0, 1.0) } else { 0.5 };
        if d == 0 { [x as f32 + 0.5 + t, y as f32 + 0.5] } else { [x as f32 + 0.5, y as f32 + 0.5 + t] }
    };
    // next[from edge] = to edge, oriented so the filled side is on the left in a y-up frame
    // (that is on the right as drawn in y-down pixels); orientation is normalised later anyway.
    let mut next: HashMap<Edge, Edge> = HashMap::new();
    for y in -1..h as i64 {
        for x in -1..w as i64 {
            let tl = at(x, y) >= 0.5;
            let tr = at(x + 1, y) >= 0.5;
            let br = at(x + 1, y + 1) >= 0.5;
            let bl = at(x, y + 1) >= 0.5;
            let code = (tl as u8) << 3 | (tr as u8) << 2 | (br as u8) << 1 | bl as u8;
            let top: Edge = (x, y, 0);
            let bottom: Edge = (x, y + 1, 0);
            let left: Edge = (x, y, 1);
            let right: Edge = (x + 1, y, 1);
            // Segments (from, to) with the inside on a consistent side.
            let segs: &[(Edge, Edge)] = match code {
                0 | 15 => &[],
                1 => &[(left, bottom)],
                2 => &[(bottom, right)],
                3 => &[(left, right)],
                4 => &[(right, top)],
                5 => &[(left, top), (right, bottom)],
                6 => &[(bottom, top)],
                7 => &[(left, top)],
                8 => &[(top, left)],
                9 => &[(top, bottom)],
                10 => &[(top, right), (bottom, left)],
                11 => &[(top, right)],
                12 => &[(right, left)],
                13 => &[(right, bottom)],
                14 => &[(bottom, left)],
                _ => &[],
            };
            for &(a, b) in segs {
                next.insert(a, b);
            }
        }
    }
    let mut loops = Vec::new();
    let mut keys: Vec<Edge> = next.keys().copied().collect();
    keys.sort_unstable();
    let mut used: std::collections::HashSet<Edge> = std::collections::HashSet::new();
    for start in keys {
        if used.contains(&start) {
            continue;
        }
        let mut c = Vec::new();
        let mut e = start;
        loop {
            if !used.insert(e) {
                break;
            }
            c.push(point(e));
            match next.get(&e) {
                Some(&n) => e = n,
                None => break,
            }
            if e == start {
                break;
            }
        }
        if c.len() >= 3 {
            loops.push(c);
        }
    }
    loops
}

/// Douglas–Peucker simplification of a closed loop with tolerance `eps`.
pub fn simplify(c: &Contour, eps: f32) -> Contour {
    if c.len() <= 4 || eps <= 0.0 {
        return c.clone();
    }
    // Split the loop at its two farthest-apart points (first and the farthest from it).
    let far = (1..c.len()).max_by(|&a, &b| dist2(c[0], c[a]).total_cmp(&dist2(c[0], c[b]))).unwrap_or(c.len() / 2);
    let mut keep = vec![false; c.len()];
    keep[0] = true;
    keep[far] = true;
    dp(c, 0, far, eps, &mut keep);
    let mut tail: Vec<[f32; 2]> = c[far..].to_vec();
    tail.push(c[0]);
    let mut keep_tail = vec![false; tail.len()];
    let last = tail.len() - 1;
    dp(&tail, 0, last, eps, &mut keep_tail);
    for (i, k) in keep_tail.iter().enumerate().take(last) {
        if *k {
            keep[far + i] = true;
        }
    }
    let out: Contour = c.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect();
    if out.len() >= 3 { out } else { c.clone() }
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

fn seg_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    if l2 < 1e-12 {
        return dist2(p, a).sqrt();
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0);
    dist2(p, [a[0] + t * dx, a[1] + t * dy]).sqrt()
}

fn dp(c: &[[f32; 2]], i: usize, j: usize, eps: f32, keep: &mut [bool]) {
    // Iterative to bound the stack on long outlines.
    let mut stack = vec![(i, j)];
    while let Some((i, j)) = stack.pop() {
        if j <= i + 1 {
            continue;
        }
        let (mut best, mut at) = (0.0f32, i);
        for k in i + 1..j {
            let d = seg_dist(c[k], c[i], c[j]);
            if d > best {
                best = d;
                at = k;
            }
        }
        if best > eps {
            keep[at] = true;
            stack.push((i, at));
            stack.push((at, j));
        }
    }
}

/// Signed area (positive = counter-clockwise in a y-up frame).
pub fn signed_area(c: &[[f32; 2]]) -> f32 {
    let n = c.len();
    (0..n).map(|i| {
        let (a, b) = (c[i], c[(i + 1) % n]);
        a[0] * b[1] - b[0] * a[1]
    }).sum::<f32>()
        * 0.5
}

pub fn point_in_polygon(p: [f32; 2], c: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let n = c.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (a, b) = (c[i], c[j % n]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// An outline and its holes, outline counter-clockwise and holes clockwise (y up).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shape {
    pub outer: Contour,
    pub holes: Vec<Contour>,
}

/// Group loops into shapes by nesting depth: even depth = outline, odd = hole of the nearest
/// enclosing outline. Loops must be in a y-up frame. Tiny loops (area < `min_area`) are dropped.
pub fn shapes(loops: &[Contour], min_area: f32) -> Vec<Shape> {
    let loops: Vec<&Contour> = loops.iter().filter(|c| c.len() >= 3 && signed_area(c).abs() >= min_area).collect();
    let depth: Vec<usize> = loops.iter().enumerate().map(|(i, c)| loops.iter().enumerate().filter(|(j, o)| *j != i && point_in_polygon(c[0], o)).count()).collect();
    let mut out: Vec<(usize, Shape)> = Vec::new();
    for (i, c) in loops.iter().enumerate() {
        if depth[i] % 2 == 0 {
            let mut outer = (*c).clone();
            if signed_area(&outer) < 0.0 {
                outer.reverse();
            }
            out.push((i, Shape { outer, holes: Vec::new() }));
        }
    }
    for (i, c) in loops.iter().enumerate() {
        if depth[i] % 2 == 1 {
            // Parent: an enclosing outline one level up.
            if let Some((_, s)) = out.iter_mut().find(|(j, s)| depth[*j] + 1 == depth[i] && point_in_polygon(c[0], &s.outer)) {
                let mut hole = (*c).clone();
                if signed_area(&hole) > 0.0 {
                    hole.reverse();
                }
                s.holes.push(hole);
            }
        }
    }
    out.into_iter().map(|(_, s)| s).collect()
}

// ---------------------------------------------------------------- triangulation

/// Triangulate a polygon with holes by ear clipping, holes bridged into the outline.
/// Returns triangles as indices into the concatenation `outer ++ holes…`.
pub fn triangulate(shape: &Shape) -> Vec<[u32; 3]> {
    let mut pts: Vec<[f32; 2]> = shape.outer.clone();
    let mut ring: Vec<u32> = (0..pts.len() as u32).collect();
    // Bridge holes, rightmost first.
    let mut holes: Vec<(usize, &Contour)> = Vec::new();
    for h in &shape.holes {
        if h.len() >= 3 {
            holes.push((pts.len(), h));
            pts.extend_from_slice(h);
        }
    }
    holes.sort_by(|a, b| {
        let ma = a.1.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
        let mb = b.1.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
        mb.total_cmp(&ma)
    });
    for (base, h) in holes {
        // Rightmost hole vertex.
        let hi = (0..h.len()).max_by(|&a, &b| h[a][0].total_cmp(&h[b][0])).unwrap_or(0);
        let hp = h[hi];
        // Closest ring vertex to the right that the bridge can reach without crossing an edge.
        let mut best: Option<(usize, f32)> = None;
        for (k, &ri) in ring.iter().enumerate() {
            let p = pts[ri as usize];
            if p[0] < hp[0] - 1e-6 {
                continue;
            }
            let d = dist2(p, hp);
            if best.is_some_and(|(_, bd)| d >= bd) {
                continue;
            }
            if !bridge_crosses(&pts, &ring, k, hp) {
                best = Some((k, d));
            }
        }
        let k = match best {
            Some((k, _)) => k,
            // Fallback: nearest ring vertex at all.
            None => (0..ring.len()).min_by(|&a, &b| dist2(pts[ring[a] as usize], hp).total_cmp(&dist2(pts[ring[b] as usize], hp))).unwrap_or(0),
        };
        let mut ins: Vec<u32> = Vec::with_capacity(h.len() + 2);
        for s in 0..=h.len() {
            ins.push((base + (hi + s) % h.len()) as u32);
        }
        ins.push(ring[k]);
        ring.splice(k + 1..k + 1, ins);
    }
    ear_clip(&pts, ring)
}

fn bridge_crosses(pts: &[[f32; 2]], ring: &[u32], k: usize, hp: [f32; 2]) -> bool {
    let a = pts[ring[k] as usize];
    let n = ring.len();
    for i in 0..n {
        let (p, q) = (pts[ring[i] as usize], pts[ring[(i + 1) % n] as usize]);
        if i == k || (i + 1) % n == k {
            continue;
        }
        if segments_cross(a, hp, p, q) {
            return true;
        }
    }
    false
}

fn cross2(o: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

fn segments_cross(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]) -> bool {
    let d1 = cross2(c, d, a);
    let d2 = cross2(c, d, b);
    let d3 = cross2(a, b, c);
    let d4 = cross2(a, b, d);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

fn in_triangle(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let (d1, d2, d3) = (cross2(a, b, p), cross2(b, c, p), cross2(c, a, p));
    d1 > 1e-9 && d2 > 1e-9 && d3 > 1e-9
}

fn ear_clip(pts: &[[f32; 2]], mut ring: Vec<u32>) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity(ring.len());
    let mut guard = 0usize;
    let mut i = 0usize;
    while ring.len() > 3 {
        let n = ring.len();
        let (ia, ib, ic) = (ring[(i + n - 1) % n], ring[i % n], ring[(i + 1) % n]);
        let (a, b, c) = (pts[ia as usize], pts[ib as usize], pts[ic as usize]);
        let convex = cross2(a, b, c) > 1e-12;
        let mut ear = convex;
        if ear {
            for &r in &ring {
                if r == ia || r == ib || r == ic {
                    continue;
                }
                let p = pts[r as usize];
                // Bridge duplicates share coordinates with ear corners; they don't block.
                if p == a || p == b || p == c {
                    continue;
                }
                if in_triangle(p, a, b, c) {
                    ear = false;
                    break;
                }
            }
        }
        if ear {
            out.push([ia, ib, ic]);
            ring.remove(i % n);
            guard = 0;
            if i > 0 {
                i -= 1;
            }
        } else {
            i = (i + 1) % n;
            guard += 1;
            if guard > n {
                // No ear found (self-intersecting input): clip the flattest corner so the cap
                // still closes instead of looping forever.
                let k = (0..n).max_by(|&x, &y| {
                    let f = |j: usize| cross2(pts[ring[(j + n - 1) % n] as usize], pts[ring[j] as usize], pts[ring[(j + 1) % n] as usize]);
                    f(x).total_cmp(&f(y))
                });
                let k = k.unwrap_or(0);
                out.push([ring[(k + n - 1) % n], ring[k], ring[(k + 1) % n]]);
                ring.remove(k);
                guard = 0;
                i = 0;
            }
        }
    }
    if ring.len() == 3 && cross2(pts[ring[0] as usize], pts[ring[1] as usize], pts[ring[2] as usize]).abs() > 0.0 {
        out.push([ring[0], ring[1], ring[2]]);
    }
    out
}

// ---------------------------------------------------------------- extrusion

/// Move every vertex of a closed contour inward (towards the solid, which lies on the left of a
/// counter-clockwise outline and of a clockwise hole) by `d`.
fn inset(c: &Contour, d: f32) -> Contour {
    let n = c.len();
    (0..n)
        .map(|i| {
            let (p, a, b) = (c[i], c[(i + n - 1) % n], c[(i + 1) % n]);
            let e1 = norm2([p[0] - a[0], p[1] - a[1]]);
            let e2 = norm2([b[0] - p[0], b[1] - p[1]]);
            // Left normals.
            let n1 = [-e1[1], e1[0]];
            let n2 = [-e2[1], e2[0]];
            let m = norm2([n1[0] + n2[0], n1[1] + n2[1]]);
            let cos = (m[0] * n1[0] + m[1] * n1[1]).max(0.25);
            let k = d / cos;
            [p[0] + m[0] * k, p[1] + m[1] * k]
        })
        .collect()
}

fn norm2(v: [f32; 2]) -> [f32; 2] {
    let l = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if l > 1e-12 { [v[0] / l, v[1] / l] } else { [0.0, 0.0] }
}

/// Build the extruded solid of `shapes` (y-up world units). `uv_of` maps an XY point to texture
/// coordinates for the caps (so a front material can show the original layer pixels).
pub fn extrude(shapes: &[Shape], params: &ExtrudeParams, uv_of: &dyn Fn([f32; 2]) -> [f32; 2]) -> Mesh {
    let p = params.sanitized();
    let mut m = Mesh::default();
    let half = p.depth * 0.5;
    let front_bevel = matches!(p.bevel_sides, BevelSides::Front | BevelSides::FrontAndBack);
    let back_bevel = matches!(p.bevel_sides, BevelSides::Back | BevelSides::FrontAndBack);
    let bw = p.bevel_width;
    let bh = (bw * p.bevel_angle).min(half.max(0.0));
    for s in shapes {
        let contours: Vec<&Contour> = std::iter::once(&s.outer).chain(s.holes.iter()).collect();
        let max_inset = contours.iter().map(|c| inradius_estimate(c)).fold(f32::MAX, f32::min);
        let bw = bw.min(max_inset * 0.9).max(0.0);
        let segs = if bw > 0.0 { p.bevel_segments } else { 0 };
        // Cap outline: fully inset contours.
        let cap_shape = Shape { outer: inset(&s.outer, bw), holes: s.holes.iter().map(|h| inset(h, bw)).collect() };
        let fb = if front_bevel { bw } else { 0.0 };
        let bb = if back_bevel { bw } else { 0.0 };
        let front_shape = if front_bevel { cap_shape.clone() } else { s.clone() };
        let back_shape = if back_bevel { cap_shape.clone() } else { s.clone() };
        let zf = half;
        let zb = -half;
        let tris_f = triangulate(&front_shape);
        let tris_b = triangulate(&back_shape);
        cap(&mut m, &front_shape, &tris_f, zf, true, uv_of, SLOT_FRONT);
        cap(&mut m, &back_shape, &tris_b, zb, false, uv_of, SLOT_BACK);
        let fbh = if front_bevel { bh } else { 0.0 };
        let bbh = if back_bevel { bh } else { 0.0 };
        for c in &contours {
            // Profile from the back cap edge to the front cap edge: (inset, z, slot of the band
            // that ends at this ring).
            let mut rings: Vec<(f32, f32, u16)> = Vec::new();
            if back_bevel && segs > 0 {
                for k in (0..=segs).rev() {
                    let (ins, ht) = p.bevel_contour.at(k as f32 / segs as f32);
                    // The back cap sits at `zb`; the bevel climbs to the wall at `zb + bbh`.
                    rings.push((ins * bb, zb + (1.0 - ht) * bbh, SLOT_BACK_BEVEL));
                }
            } else {
                rings.push((0.0, zb, SLOT_SIDES));
            }
            // Side wall.
            rings.push((0.0, zf - fbh, SLOT_SIDES));
            if front_bevel && segs > 0 {
                for k in 1..=segs {
                    let (ins, ht) = p.bevel_contour.at(k as f32 / segs as f32);
                    rings.push((ins * fb, zf - fbh + ht * fbh, SLOT_FRONT_BEVEL));
                }
            }
            let n = c.len();
            let mut prev: Option<(Vec<[f32; 2]>, f32)> = None;
            for &(ins, z, slot) in &rings {
                let ring = if ins > 0.0 { inset(c, ins) } else { (*c).clone() };
                if let Some((pr, pz)) = &prev {
                    if (z - pz).abs() > 1e-7 || ring != *pr {
                        for i in 0..n {
                            let j = (i + 1) % n;
                            let (a0, a1, b0, b1) = (pr[i], pr[j], ring[i], ring[j]);
                            let u0 = i as f32 / n as f32;
                            let u1 = (i + 1) as f32 / n as f32;
                            let va = m.vertex(v3(a0[0], a0[1], *pz), Vec3::ZERO, [u0, 1.0 - (*pz - zb) / p.depth.max(1e-6)]);
                            let vb = m.vertex(v3(a1[0], a1[1], *pz), Vec3::ZERO, [u1, 1.0 - (*pz - zb) / p.depth.max(1e-6)]);
                            let vc = m.vertex(v3(b1[0], b1[1], z), Vec3::ZERO, [u1, 1.0 - (z - zb) / p.depth.max(1e-6)]);
                            let vd = m.vertex(v3(b0[0], b0[1], z), Vec3::ZERO, [u0, 1.0 - (z - zb) / p.depth.max(1e-6)]);
                            // Solid on the left of the edge a0→a1 seen from +z; the outward side
                            // is the right, so wind (a0, a1, b1) facing out.
                            m.tri(va, vb, vc, slot);
                            m.tri(va, vc, vd, slot);
                        }
                    }
                }
                prev = Some((ring, z));
            }
        }
    }
    deform(&mut m, &p);
    m.compute_normals(p.smoothing.max(1.0));
    fix_winding(&mut m);
    m
}

/// Largest inset that keeps a contour from turning inside out, roughly: twice the area over the
/// perimeter (exact for a circle's radius / 2 scale).
fn inradius_estimate(c: &[[f32; 2]]) -> f32 {
    let n = c.len();
    let per: f32 = (0..n).map(|i| dist2(c[i], c[(i + 1) % n]).sqrt()).sum();
    if per <= 0.0 { 0.0 } else { (2.0 * signed_area(c).abs() / per).max(0.0) }
}

fn cap(m: &mut Mesh, s: &Shape, tris: &[[u32; 3]], z: f32, front: bool, uv_of: &dyn Fn([f32; 2]) -> [f32; 2], slot: u16) {
    let base = m.positions.len() as u32;
    let n = if front { Vec3::Z } else { -Vec3::Z };
    for c in std::iter::once(&s.outer).chain(s.holes.iter().filter(|h| h.len() >= 3)) {
        for &pt in c {
            m.vertex(v3(pt[0], pt[1], z), n, uv_of(pt));
        }
    }
    let count = m.positions.len() as u32 - base;
    for t in tris {
        if t.iter().any(|&i| i >= count) {
            continue;
        }
        if front {
            m.tri(base + t[0], base + t[1], base + t[2], slot);
        } else {
            m.tri(base + t[0], base + t[2], base + t[1], slot);
        }
    }
}

/// Twist, taper, bend and shear along z.
fn deform(m: &mut Mesh, p: &ExtrudeParams) {
    let half = p.depth * 0.5;
    if p.depth <= 0.0 {
        return;
    }
    let taper = p.taper / 100.0;
    for v in &mut m.positions {
        // t = 0 at the front, 1 at the back.
        let t = ((half - v.z) / p.depth).clamp(0.0, 1.0);
        let s = 1.0 + (taper - 1.0) * t;
        let (mut x, mut y) = (v.x * s, v.y * s);
        if p.twist != 0.0 {
            let (sn, cs) = (p.twist * t).to_radians().sin_cos();
            (x, y) = (x * cs - y * sn, x * sn + y * cs);
        }
        x += (p.shear_x.to_radians().tan()) * (half - v.z) - p.shear_x.to_radians().tan() * half;
        y += (p.shear_y.to_radians().tan()) * (half - v.z) - p.shear_y.to_radians().tan() * half;
        // Bend: offset grows quadratically with depth, sideways.
        let bend = |deg: f32| -> f32 { (deg / 90.0) * p.depth * t * t };
        x += bend(p.bend_x);
        y += bend(p.bend_y);
        v.x = x;
        v.y = y;
    }
}

/// Make every triangle face away from the solid's centre line where it clearly points inward
/// (the deformations and the inset can flip a few on odd outlines).
fn fix_winding(m: &mut Mesh) {
    for t in 0..m.indices.len() {
        let [a, b, c] = m.indices[t];
        let fn_ = m.face_normal(t);
        let vn = m.normals[a as usize] + m.normals[b as usize] + m.normals[c as usize];
        if fn_.dot(vn) < 0.0 {
            m.indices[t] = [a, c, b];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_mask(w: usize, h: usize, hole: bool) -> Vec<f32> {
        let mut m = vec![0.0; w * h];
        for y in 2..h - 2 {
            for x in 2..w - 2 {
                let inner = hole && x > w / 3 && x < 2 * w / 3 && y > h / 3 && y < 2 * h / 3;
                m[y * w + x] = if inner { 0.0 } else { 1.0 };
            }
        }
        m
    }

    #[test]
    fn traces_a_square_with_a_hole() {
        let mask = square_mask(30, 30, true);
        let loops = trace(&mask, 30, 30);
        assert_eq!(loops.len(), 2);
        let flipped: Vec<Contour> = loops.iter().map(|c| c.iter().map(|p| [p[0], -p[1]]).collect()).collect();
        let s = shapes(&flipped, 1.0);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].holes.len(), 1);
        assert!(signed_area(&s[0].outer) > 0.0);
        assert!(signed_area(&s[0].holes[0]) < 0.0);
    }

    #[test]
    fn triangulation_covers_the_area() {
        let mask = square_mask(40, 40, true);
        let loops: Vec<Contour> = trace(&mask, 40, 40).iter().map(|c| simplify(c, 0.5).iter().map(|p| [p[0], -p[1]]).collect()).collect();
        let s = &shapes(&loops, 1.0)[0];
        let tris = triangulate(s);
        let pts: Vec<[f32; 2]> = s.outer.iter().chain(s.holes.iter().flatten()).copied().collect();
        let area: f32 = tris.iter().map(|t| cross2(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]).abs() * 0.5).sum();
        let expect = signed_area(&s.outer) + signed_area(&s.holes[0]);
        assert!((area - expect).abs() / expect < 0.02, "area {area} vs {expect}");
    }

    #[test]
    fn extrusion_is_closed_and_slotted() {
        let s = Shape { outer: vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], holes: vec![] };
        let m = extrude(&[s], &ExtrudeParams { bevel_sides: BevelSides::FrontAndBack, ..Default::default() }, &|p| p);
        assert!(m.triangle_count() > 12);
        for slot in [SLOT_FRONT, SLOT_FRONT_BEVEL, SLOT_SIDES, SLOT_BACK_BEVEL, SLOT_BACK] {
            assert!(m.material.contains(&slot), "slot {slot}");
        }
        let b = m.bounds();
        assert!((b.size().z - 0.35).abs() < 1e-3);
        // Front cap faces +z.
        let t = m.material.iter().position(|&s| s == SLOT_FRONT).unwrap();
        assert!(m.face_normal(t).z > 0.9);
        // Sides face outward.
        for t in 0..m.triangle_count() {
            if m.material[t] == SLOT_SIDES {
                let [a, b, c] = m.indices[t];
                let centre = (m.positions[a as usize] + m.positions[b as usize] + m.positions[c as usize]) / 3.0;
                assert!(m.face_normal(t).dot(v3(centre.x, centre.y, 0.0)) > 0.0);
            }
        }
    }

    #[test]
    fn degenerate_input_does_not_panic() {
        assert!(trace(&[], 0, 0).is_empty());
        let s = Shape { outer: vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], holes: vec![] };
        let _ = extrude(&[s], &ExtrudeParams::default(), &|p| p);
    }
}

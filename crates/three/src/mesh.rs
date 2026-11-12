//! Triangle meshes and Photoshop's mesh presets (3D › New Mesh from Layer › Mesh Preset).
//!
//! Every mesh is indexed triangles with per-vertex normals and UVs, plus a material slot per
//! triangle. Presets are about one unit across, centred on the origin, resting on nothing (the
//! scene drops them onto the ground plane).

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};

use serde::{Deserialize, Serialize};

use crate::math::{Mat4, Vec3, v3};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    /// Triangles, three vertex indices each.
    pub indices: Vec<[u32; 3]>,
    /// Material slot of each triangle (same length as `indices`).
    pub material: Vec<u16>,
}

/// Axis-aligned box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb { min: v3(f32::INFINITY, f32::INFINITY, f32::INFINITY), max: v3(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY) };
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x
    }
    pub fn add(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    pub fn union(&self, o: &Aabb) -> Aabb {
        Aabb { min: self.min.min(o.min), max: self.max.max(o.max) }
    }
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    pub fn size(&self) -> Vec3 {
        if self.is_empty() { Vec3::ZERO } else { self.max - self.min }
    }
    pub fn corners(&self) -> [Vec3; 8] {
        let (a, b) = (self.min, self.max);
        [v3(a.x, a.y, a.z), v3(b.x, a.y, a.z), v3(a.x, b.y, a.z), v3(b.x, b.y, a.z), v3(a.x, a.y, b.z), v3(b.x, a.y, b.z), v3(a.x, b.y, b.z), v3(b.x, b.y, b.z)]
    }
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }
    pub fn slot_count(&self) -> usize {
        self.material.iter().map(|&m| m as usize + 1).max().unwrap_or(1)
    }

    pub fn bounds(&self) -> Aabb {
        let mut b = Aabb::EMPTY;
        for &p in &self.positions {
            b.add(p);
        }
        b
    }

    /// Bounds after transforming by `m`.
    pub fn bounds_with(&self, m: &Mat4) -> Aabb {
        let mut b = Aabb::EMPTY;
        for &p in &self.positions {
            b.add(m.transform_point(p));
        }
        b
    }

    /// Append a vertex, returning its index.
    pub fn vertex(&mut self, p: Vec3, n: Vec3, uv: [f32; 2]) -> u32 {
        self.positions.push(p);
        self.normals.push(n);
        self.uvs.push(uv);
        (self.positions.len() - 1) as u32
    }

    pub fn tri(&mut self, a: u32, b: u32, c: u32, slot: u16) {
        self.indices.push([a, b, c]);
        self.material.push(slot);
    }

    /// Append another mesh, shifting its material slots by `slot_offset`.
    pub fn append(&mut self, other: &Mesh, slot_offset: u16) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&other.positions);
        self.normals.extend_from_slice(&other.normals);
        self.uvs.extend_from_slice(&other.uvs);
        self.indices.extend(other.indices.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        self.material.extend(other.material.iter().map(|&m| m.saturating_add(slot_offset)));
    }

    /// Apply a transform to positions and normals.
    pub fn transform(&mut self, m: &Mat4) {
        let nm = m.normal_matrix();
        for p in &mut self.positions {
            *p = m.transform_point(*p);
        }
        for n in &mut self.normals {
            *n = nm.transform_vector(*n).normalize();
        }
    }

    /// Drop triangles that reference missing vertices or are degenerate, and fix array lengths,
    /// so a mesh read from a file can't make the renderers index out of range.
    pub fn sanitize(&mut self) {
        let n = self.positions.len();
        self.normals.resize(n, Vec3::ZERO);
        self.uvs.resize(n, [0.0, 0.0]);
        self.material.resize(self.indices.len(), 0);
        let mut keep_i = Vec::with_capacity(self.indices.len());
        let mut keep_m = Vec::with_capacity(self.indices.len());
        for (t, m) in self.indices.iter().zip(&self.material) {
            if t.iter().all(|&i| (i as usize) < n) && t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                let (a, b, c) = (self.positions[t[0] as usize], self.positions[t[1] as usize], self.positions[t[2] as usize]);
                if a.is_finite() && b.is_finite() && c.is_finite() {
                    keep_i.push(*t);
                    keep_m.push(*m);
                }
            }
        }
        self.indices = keep_i;
        self.material = keep_m;
        if self.normals.iter().any(|n| n.length() < 0.5 || !n.is_finite()) {
            self.compute_normals(60.0);
        }
    }

    /// Face normal of triangle `t`.
    pub fn face_normal(&self, t: usize) -> Vec3 {
        let [a, b, c] = self.indices[t];
        let (a, b, c) = (self.positions[a as usize], self.positions[b as usize], self.positions[c as usize]);
        (b - a).cross(c - a).normalize_or(Vec3::ZERO)
    }

    /// Recompute normals with an auto-smooth crease angle (degrees): faces meeting at less than
    /// `crease` share smoothed normals, sharper edges stay hard. Vertices are split as needed.
    pub fn compute_normals(&mut self, crease: f32) {
        let cos_crease = crease.to_radians().cos();
        let faces: Vec<Vec3> = (0..self.indices.len()).map(|t| self.face_normal(t)).collect();
        // Corners grouped by quantized position.
        let key = |p: Vec3| ((p.x * 1e4).round() as i64, (p.y * 1e4).round() as i64, (p.z * 1e4).round() as i64);
        let mut by_pos: HashMap<(i64, i64, i64), Vec<(usize, usize)>> = HashMap::new();
        for (t, tri) in self.indices.iter().enumerate() {
            for (k, &i) in tri.iter().enumerate() {
                by_pos.entry(key(self.positions[i as usize])).or_default().push((t, k));
            }
        }
        let mut out = Mesh { material: self.material.clone(), ..Default::default() };
        out.indices = vec![[0; 3]; self.indices.len()];
        // (original vertex, smoothed normal key) → new vertex
        let mut made: HashMap<(u32, (i32, i32, i32)), u32> = HashMap::new();
        for (t, tri) in self.indices.iter().enumerate() {
            for (k, &i) in tri.iter().enumerate() {
                let p = self.positions[i as usize];
                let fnorm = faces[t];
                let mut n = Vec3::ZERO;
                if let Some(list) = by_pos.get(&key(p)) {
                    for &(t2, _) in list {
                        if faces[t2].dot(fnorm) >= cos_crease {
                            n += faces[t2];
                        }
                    }
                }
                let n = n.normalize_or(fnorm.normalize_or(Vec3::Z));
                let nk = ((n.x * 1000.0) as i32, (n.y * 1000.0) as i32, (n.z * 1000.0) as i32);
                let idx = *made.entry((i, nk)).or_insert_with(|| out.vertex(p, n, self.uvs.get(i as usize).copied().unwrap_or([0.0, 0.0])));
                out.indices[t][k] = idx;
            }
        }
        *self = out;
    }

    /// Flip triangle winding and normals (the inside of a shape).
    pub fn flip(&mut self) {
        for t in &mut self.indices {
            t.swap(1, 2);
        }
        for n in &mut self.normals {
            *n = -*n;
        }
    }
}

/// Photoshop's mesh presets plus the plain primitives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MeshPreset {
    Cone,
    Cube,
    CubeWrap,
    Cylinder,
    Donut,
    Hat,
    Pyramid,
    Ring,
    SodaCan,
    Sphere,
    WineBottle,
    Plane,
    TwoSidedPlane,
}

impl MeshPreset {
    pub const ALL: [MeshPreset; 13] = [
        MeshPreset::Cone,
        MeshPreset::Cube,
        MeshPreset::CubeWrap,
        MeshPreset::Cylinder,
        MeshPreset::Donut,
        MeshPreset::Hat,
        MeshPreset::Pyramid,
        MeshPreset::Ring,
        MeshPreset::SodaCan,
        MeshPreset::Sphere,
        MeshPreset::WineBottle,
        MeshPreset::Plane,
        MeshPreset::TwoSidedPlane,
    ];

    pub fn id(self) -> &'static str {
        match self {
            MeshPreset::Cone => "cone",
            MeshPreset::Cube => "cube",
            MeshPreset::CubeWrap => "cubeWrap",
            MeshPreset::Cylinder => "cylinder",
            MeshPreset::Donut => "donut",
            MeshPreset::Hat => "hat",
            MeshPreset::Pyramid => "pyramid",
            MeshPreset::Ring => "ring",
            MeshPreset::SodaCan => "sodaCan",
            MeshPreset::Sphere => "sphere",
            MeshPreset::WineBottle => "wineBottle",
            MeshPreset::Plane => "plane",
            MeshPreset::TwoSidedPlane => "twoSidedPlane",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MeshPreset::Cone => "Cone",
            MeshPreset::Cube => "Cube",
            MeshPreset::CubeWrap => "Cube Wrap",
            MeshPreset::Cylinder => "Cylinder",
            MeshPreset::Donut => "Donut",
            MeshPreset::Hat => "Hat",
            MeshPreset::Pyramid => "Pyramid",
            MeshPreset::Ring => "Ring",
            MeshPreset::SodaCan => "Soda Can",
            MeshPreset::Sphere => "Sphere",
            MeshPreset::WineBottle => "Wine Bottle",
            MeshPreset::Plane => "Plane",
            MeshPreset::TwoSidedPlane => "Two-Sided Plane",
        }
    }

    pub fn from_id(id: &str) -> Option<MeshPreset> {
        MeshPreset::ALL.into_iter().find(|p| p.id().eq_ignore_ascii_case(id))
    }

    /// Material slot names, in slot order.
    pub fn slot_names(self) -> &'static [&'static str] {
        match self {
            MeshPreset::Cylinder | MeshPreset::SodaCan => &["Top Material", "Label Material", "Bottom Material"],
            MeshPreset::Cube | MeshPreset::CubeWrap => &["Front Material", "Right Material", "Back Material", "Left Material", "Top Material", "Bottom Material"],
            MeshPreset::WineBottle => &["Label Material", "Glass Material", "Cork Material"],
            MeshPreset::TwoSidedPlane => &["Front Material", "Back Material"],
            _ => &["Default Material"],
        }
    }

    /// Build the mesh. `aspect` (width / height of the source layer) shapes planes.
    pub fn build(self, aspect: f32) -> Mesh {
        let aspect = if aspect.is_finite() && aspect > 0.0 { aspect.clamp(0.01, 100.0) } else { 1.0 };
        let mut m = match self {
            MeshPreset::Cube => cube(false),
            MeshPreset::CubeWrap => cube(true),
            MeshPreset::Sphere => uv_sphere(0.5, 48, 24),
            MeshPreset::Cylinder => lathe_capped(&[(0.5, -0.5), (0.5, 0.5)], 48, [0, 1, 2]),
            MeshPreset::Cone => lathe(&[(0.0, 0.5), (0.5, -0.5), (0.0, -0.5)], 48, 0, 60.0),
            MeshPreset::Pyramid => lathe(&[(0.0, 0.5), (0.5, -0.5), (0.0, -0.5)], 4, 0, 30.0),
            MeshPreset::Donut => torus(0.35, 0.15, 48, 24),
            MeshPreset::Ring => torus(0.42, 0.06, 64, 16),
            MeshPreset::Hat => hat(),
            MeshPreset::SodaCan => soda_can(),
            MeshPreset::WineBottle => wine_bottle(),
            MeshPreset::Plane => plane(aspect, false),
            MeshPreset::TwoSidedPlane => plane(aspect, true),
        };
        m.sanitize();
        m
    }
}

/// Flat plane in the XY plane facing +Z, height 1, width `aspect`.
pub fn plane(aspect: f32, two_sided: bool) -> Mesh {
    let mut m = Mesh::default();
    let (w, h) = (aspect * 0.5, 0.5);
    let a = m.vertex(v3(-w, -h, 0.0), Vec3::Z, [0.0, 1.0]);
    let b = m.vertex(v3(w, -h, 0.0), Vec3::Z, [1.0, 1.0]);
    let c = m.vertex(v3(w, h, 0.0), Vec3::Z, [1.0, 0.0]);
    let d = m.vertex(v3(-w, h, 0.0), Vec3::Z, [0.0, 0.0]);
    m.tri(a, b, c, 0);
    m.tri(a, c, d, 0);
    if two_sided {
        let a = m.vertex(v3(-w, -h, 0.0), -Vec3::Z, [1.0, 1.0]);
        let b = m.vertex(v3(w, -h, 0.0), -Vec3::Z, [0.0, 1.0]);
        let c = m.vertex(v3(w, h, 0.0), -Vec3::Z, [0.0, 0.0]);
        let d = m.vertex(v3(-w, h, 0.0), -Vec3::Z, [1.0, 0.0]);
        m.tri(a, c, b, 1);
        m.tri(a, d, c, 1);
    }
    m
}

/// Unit cube; `wrap` maps one image around the four sides (Cube Wrap) instead of the whole image
/// on every face.
pub fn cube(wrap: bool) -> Mesh {
    let mut m = Mesh::default();
    // (normal, u axis, v axis, slot, wrap u range)
    let faces: [(Vec3, Vec3, Vec3, u16, f32); 6] = [
        (Vec3::Z, Vec3::X, Vec3::Y, 0, 0.0),
        (Vec3::X, -Vec3::Z, Vec3::Y, 1, 0.25),
        (-Vec3::Z, -Vec3::X, Vec3::Y, 2, 0.5),
        (-Vec3::X, Vec3::Z, Vec3::Y, 3, 0.75),
        (Vec3::Y, Vec3::X, -Vec3::Z, 4, 0.0),
        (-Vec3::Y, Vec3::X, Vec3::Z, 5, 0.0),
    ];
    for (n, u, v, slot, u0) in faces {
        let c = n * 0.5;
        let corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)];
        let mut ids = [0u32; 4];
        for (k, (a, b)) in corners.into_iter().enumerate() {
            let p = c + u * a + v * b;
            let (s, t) = (a + 0.5, 0.5 - b);
            let uv = if wrap && slot < 4 { [u0 + s * 0.25, t] } else { [s, t] };
            ids[k] = m.vertex(p, n, uv);
        }
        m.tri(ids[0], ids[1], ids[2], slot);
        m.tri(ids[0], ids[2], ids[3], slot);
    }
    m
}

pub fn uv_sphere(r: f32, seg: u32, rings: u32) -> Mesh {
    let mut m = Mesh::default();
    let (seg, rings) = (seg.max(3), rings.max(2));
    for j in 0..=rings {
        let v = j as f32 / rings as f32;
        let phi = v * PI;
        for i in 0..=seg {
            let u = i as f32 / seg as f32;
            let th = u * TAU;
            let n = v3(phi.sin() * th.sin(), phi.cos(), phi.sin() * th.cos());
            m.vertex(n * r, n, [u, v]);
        }
    }
    let row = seg + 1;
    for j in 0..rings {
        for i in 0..seg {
            let a = j * row + i;
            let (b, c, d) = (a + 1, a + row, a + row + 1);
            if j != 0 {
                m.tri(a, c, b, 0);
            }
            if j != rings - 1 {
                m.tri(b, c, d, 0);
            }
        }
    }
    m
}

pub fn torus(major: f32, minor: f32, seg: u32, sides: u32) -> Mesh {
    let mut m = Mesh::default();
    let (seg, sides) = (seg.max(3), sides.max(3));
    for j in 0..=sides {
        let v = j as f32 / sides as f32;
        let ph = v * TAU;
        for i in 0..=seg {
            let u = i as f32 / seg as f32;
            let th = u * TAU;
            let dir = v3(th.sin(), 0.0, th.cos());
            let n = dir * ph.cos() + Vec3::Y * ph.sin();
            m.vertex(dir * major + n * minor, n, [u, v]);
        }
    }
    let row = seg + 1;
    for j in 0..sides {
        for i in 0..seg {
            let a = j * row + i;
            m.tri(a, a + 1, a + row, 0);
            m.tri(a + 1, a + row + 1, a + row, 0);
        }
    }
    m
}

/// Surface of revolution around +Y of the profile `(radius, y)` (top to bottom), `seg` around.
/// Normals come from auto-smoothing at `crease` degrees.
pub fn lathe(profile: &[(f32, f32)], seg: u32, slot: u16, crease: f32) -> Mesh {
    let mut m = Mesh::default();
    let seg = seg.max(3);
    // Arc length for v.
    let mut acc = vec![0.0f32];
    for w in profile.windows(2) {
        let d = ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt();
        acc.push(acc.last().copied().unwrap_or(0.0) + d);
    }
    let total = acc.last().copied().unwrap_or(1.0).max(1e-6);
    for (k, &(r, y)) in profile.iter().enumerate() {
        for i in 0..=seg {
            let u = i as f32 / seg as f32;
            let th = u * TAU;
            m.vertex(v3(r * th.sin(), y, r * th.cos()), Vec3::ZERO, [u, acc[k] / total]);
        }
    }
    let row = seg + 1;
    for k in 0..profile.len().saturating_sub(1) as u32 {
        for i in 0..seg {
            let a = k * row + i;
            m.tri(a, a + row, a + 1, slot);
            m.tri(a + 1, a + row, a + row + 1, slot);
        }
    }
    m.sanitize_degenerate_area();
    m.compute_normals(crease);
    m
}

impl Mesh {
    /// Remove zero-area triangles (lathes collapse on the axis).
    fn sanitize_degenerate_area(&mut self) {
        let keep: Vec<bool> = (0..self.indices.len()).map(|t| self.face_normal(t).length() > 0.5).collect();
        let mut k = keep.iter();
        self.indices.retain(|_| k.next().copied().unwrap_or(false));
        let mut k = keep.iter();
        self.material.retain(|_| k.next().copied().unwrap_or(false));
    }
}

/// A lathe side with flat caps on top and bottom, in slots `[top, side, bottom]`.
pub fn lathe_capped(profile: &[(f32, f32)], seg: u32, slots: [u16; 3]) -> Mesh {
    let mut m = lathe(profile, seg, slots[1], 30.0);
    if let (Some(&(rt, yt)), Some(&(rb, yb))) = (profile.first(), profile.last()) {
        disc(&mut m, rt, yt, seg, true, slots[0]);
        disc(&mut m, rb, yb, seg, false, slots[2]);
    }
    m
}

fn disc(m: &mut Mesh, r: f32, y: f32, seg: u32, up: bool, slot: u16) {
    if r <= 1e-6 {
        return;
    }
    let n = if up { Vec3::Y } else { -Vec3::Y };
    let c = m.vertex(v3(0.0, y, 0.0), n, [0.5, 0.5]);
    let first = m.positions.len() as u32;
    for i in 0..=seg {
        let th = i as f32 / seg as f32 * TAU;
        m.vertex(v3(r * th.sin(), y, r * th.cos()), n, [0.5 + 0.5 * th.sin(), 0.5 - 0.5 * th.cos()]);
    }
    for i in 0..seg {
        let (a, b) = (first + i, first + i + 1);
        if up { m.tri(c, a, b, slot) } else { m.tri(c, b, a, slot) }
    }
}

fn hat() -> Mesh {
    let profile = [(0.0, 0.42), (0.2, 0.40), (0.26, 0.30), (0.27, 0.0), (0.28, -0.02), (0.5, -0.03), (0.52, -0.05), (0.5, -0.07), (0.28, -0.06), (0.0, -0.06)];
    lathe(&profile, 48, 0, 50.0)
}

fn soda_can() -> Mesh {
    // Label band in the middle, rounded rim top and bottom.
    let mut m = lathe(&[(0.30, 0.38), (0.34, 0.33), (0.34, -0.33), (0.30, -0.38)], 48, 1, 40.0);
    // Mark the rim bands as top/bottom material.
    let tris = m.indices.clone();
    for (t, tri) in tris.iter().enumerate() {
        let y = (m.positions[tri[0] as usize].y + m.positions[tri[1] as usize].y + m.positions[tri[2] as usize].y) / 3.0;
        if y > 0.33 {
            m.material[t] = 0;
        } else if y < -0.33 {
            m.material[t] = 2;
        }
    }
    disc(&mut m, 0.30, 0.38, 48, true, 0);
    disc(&mut m, 0.30, -0.38, 48, false, 2);
    m
}

fn wine_bottle() -> Mesh {
    let profile = [(0.0, 0.62), (0.055, 0.62), (0.055, 0.5), (0.06, 0.45), (0.065, 0.3), (0.12, 0.12), (0.17, 0.02), (0.18, -0.08), (0.18, -0.5), (0.17, -0.52), (0.0, -0.52)];
    let mut m = lathe(&profile, 48, 1, 35.0);
    let tris = m.indices.clone();
    for (t, tri) in tris.iter().enumerate() {
        let y = (m.positions[tri[0] as usize].y + m.positions[tri[1] as usize].y + m.positions[tri[2] as usize].y) / 3.0;
        if y > 0.45 {
            m.material[t] = 2;
        } else if (-0.42..-0.1).contains(&y) {
            m.material[t] = 0;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_builds_a_valid_mesh() {
        for p in MeshPreset::ALL {
            let m = p.build(1.5);
            assert!(m.triangle_count() > 0, "{p:?}");
            assert_eq!(m.material.len(), m.indices.len());
            assert!(m.indices.iter().flatten().all(|&i| (i as usize) < m.positions.len()));
            assert!(m.normals.iter().all(|n| (n.length() - 1.0).abs() < 1e-3), "{p:?} normals");
            assert!((m.slot_count()) <= p.slot_names().len(), "{p:?} slots {} vs {}", m.slot_count(), p.slot_names().len());
            let b = m.bounds();
            assert!(b.size().max_elem() <= 1.6, "{p:?} size");
        }
    }

    #[test]
    fn cube_normals_point_outwards() {
        let m = MeshPreset::Cube.build(1.0);
        for t in 0..m.triangle_count() {
            let [a, b, c] = m.indices[t];
            let centre = (m.positions[a as usize] + m.positions[b as usize] + m.positions[c as usize]) / 3.0;
            assert!(m.face_normal(t).dot(centre) > 0.0);
        }
    }
}

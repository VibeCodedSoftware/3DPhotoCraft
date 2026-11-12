//! Rendering a scene to RGBA pixels.
//!
//! Two renderers share one shading model ([`shade`]):
//! - [`render_preview`]: a parallel scanline rasterizer (z-buffer, deferred shading, PCF shadow
//!   maps for infinite and spot lights, ground-plane shadow catcher, every surface style). Fast
//!   enough to re-render while dragging, so it keeps the layer up to date as you edit.
//! - [`crate::trace::render_traced`]: a ray tracer for 3D › Render 3D Layer (soft shadows from
//!   every light including point lights, reflections, refraction, depth of field, antialiasing).
//!
//! Output is straight-alpha RGBA, sRGB-encoded 0–1, row-major, top row first.

use std::sync::LazyLock;

use rayon::prelude::*;

use crate::math::{Mat4, Vec3, linear_to_srgb, srgb_to_linear, v3};
use crate::mesh::Aabb;
use crate::scene::{Background, CrossAxis, Image, Light, LightKind, Material, Scene, Style};

/// sRGB byte → linear float.
pub static SRGB_LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut t = [0.0; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = srgb_to_linear(i as f32 / 255.0);
    }
    t
});

/// Largest output side either renderer accepts.
pub const MAX_SIDE: u32 = 16384;

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 4]>,
}

impl Output {
    pub fn to_rgba8(&self) -> Vec<u8> {
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        self.pixels.iter().flat_map(|p| [q(p[0]), q(p[1]), q(p[2]), q(p[3])]).collect()
    }
}

// ---------------------------------------------------------------- prepared scene

/// A world-space triangle ready to render.
#[derive(Clone, Debug)]
pub struct Tri {
    pub p: [Vec3; 3],
    pub n: [Vec3; 3],
    pub uv: [[f32; 2]; 3],
    /// Tangent and bitangent (for bump and normal maps).
    pub t: Vec3,
    pub b: Vec3,
    /// Index into [`Prepared::materials`].
    pub mat: u32,
    pub obj: u32,
    /// Seen by the camera (not Invisible).
    pub seen: bool,
    pub casts: bool,
}

pub struct Prepared<'a> {
    pub scene: &'a Scene,
    pub tris: Vec<Tri>,
    pub materials: Vec<&'a Material>,
    pub bounds: Aabb,
}

pub fn prepare(scene: &Scene) -> Prepared<'_> {
    let mut tris = Vec::new();
    let mut materials: Vec<&Material> = Vec::new();
    let mut bounds = Aabb::EMPTY;
    for o in scene.objects.iter().filter(|o| o.visible) {
        let mesh = o.mesh();
        let m = o.transform.matrix();
        let nm = m.normal_matrix();
        let base = materials.len() as u32;
        let fallback: &Material = o.materials.first().unwrap_or(&DEFAULT_MATERIAL);
        let slots = mesh.slot_count().max(1);
        for s in 0..slots {
            materials.push(o.materials.get(s).unwrap_or(fallback));
        }
        let wp: Vec<Vec3> = mesh.positions.iter().map(|&p| m.transform_point(p)).collect();
        let wn: Vec<Vec3> = mesh.normals.iter().map(|&n| nm.transform_vector(n).normalize()).collect();
        // Mirror transforms flip the winding.
        let det_neg = o.transform.scale.x * o.transform.scale.y * o.transform.scale.z < 0.0;
        for (k, t) in mesh.indices.iter().enumerate() {
            let [a, b, c] = t.map(|i| i as usize);
            let (Some(&pa), Some(&pb), Some(&pc)) = (wp.get(a), wp.get(b), wp.get(c)) else { continue };
            let n = |i: usize| wn.get(i).copied().unwrap_or(Vec3::Z);
            let uv = |i: usize| mesh.uvs.get(i).copied().unwrap_or([0.0, 0.0]);
            let (mut p, mut nn, mut uu) = ([pa, pb, pc], [n(a), n(b), n(c)], [uv(a), uv(b), uv(c)]);
            if det_neg {
                p.swap(1, 2);
                nn.swap(1, 2);
                uu.swap(1, 2);
            }
            let (tt, bb) = tangents(&p, &uu);
            let slot = mesh.material.get(k).copied().unwrap_or(0) as u32;
            for q in p {
                bounds.add(q);
            }
            tris.push(Tri { p, n: nn, uv: uu, t: tt, b: bb, mat: base + slot.min(slots as u32 - 1), obj: o.id, seen: !o.invisible, casts: o.cast_shadows });
        }
    }
    Prepared { scene, tris, materials, bounds }
}

static DEFAULT_MATERIAL: LazyLock<Material> = LazyLock::new(Material::default);

fn tangents(p: &[Vec3; 3], uv: &[[f32; 2]; 3]) -> (Vec3, Vec3) {
    let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
    let (du1, dv1, du2, dv2) = (uv[1][0] - uv[0][0], uv[1][1] - uv[0][1], uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]);
    let det = du1 * dv2 - du2 * dv1;
    if det.abs() < 1e-12 {
        let n = e1.cross(e2).normalize();
        let t = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let t = (t - n * n.dot(t)).normalize();
        return (t, n.cross(t));
    }
    let r = 1.0 / det;
    ((e1 * dv2 - e2 * dv1) * r, (e2 * du1 - e1 * du2) * r)
}

// ---------------------------------------------------------------- shading

/// A visible surface point.
pub struct Surface<'a> {
    pub pos: Vec3,
    /// Interpolated (smooth) normal, facing the viewer.
    pub normal: Vec3,
    pub uv: [f32; 2],
    pub tri: &'a Tri,
    pub material: &'a Material,
}

/// Material values at a surface point (textures applied, linear colour).
pub struct Sampled {
    pub base: Vec3,
    pub alpha: f32,
    pub normal: Vec3,
    pub specular: Vec3,
    pub emission: Vec3,
    pub ambient: Vec3,
}

pub fn lin3(c: [f32; 3]) -> Vec3 {
    v3(srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]))
}

pub fn sample_material(scene: &Scene, s: &Surface<'_>) -> Sampled {
    let m = s.material;
    let uv = [s.uv[0] * m.uv_scale[0] + m.uv_offset[0], s.uv[1] * m.uv_scale[1] + m.uv_offset[1]];
    let mut base = lin3(m.diffuse);
    let mut alpha = m.opacity;
    if let Some(img) = scene.texture(m.diffuse_texture) {
        let t = img.sample(uv);
        // A diffuse texture replaces the diffuse colour, as in Photoshop.
        base = v3(t[0], t[1], t[2]);
        alpha *= t[3];
    }
    if let Some(img) = scene.texture(m.opacity_texture) {
        alpha *= img.luminance(uv);
    }
    let mut n = s.normal;
    let (t, b) = (s.tri.t, s.tri.b);
    if let Some(img) = scene.texture(m.normal_texture) {
        let c = img.sample(uv);
        // Normal maps are linear data; undo the sRGB decode of the sampler.
        let d = v3(linear_to_srgb(c[0]) * 2.0 - 1.0, linear_to_srgb(c[1]) * 2.0 - 1.0, linear_to_srgb(c[2]) * 2.0 - 1.0);
        let (tt, bb) = (t.normalize(), b.normalize());
        n = (tt * d.x + bb * d.y + n * d.z.max(0.05)).normalize_or(n);
    }
    if m.bump > 0.0
        && let Some(img) = scene.texture(m.bump_texture)
    {
        let e = 1.0 / img.width.max(img.height).max(1) as f32;
        let h0 = img.luminance(uv);
        let hu = img.luminance([uv[0] + e, uv[1]]) - h0;
        let hv = img.luminance([uv[0], uv[1] + e]) - h0;
        let k = m.bump * 0.02 / e.max(1e-4) * 0.05;
        let (tt, bb) = (t.normalize(), b.normalize());
        n = (n - tt * (hu * k) - bb * (hv * k)).normalize_or(n);
    }
    Sampled { base, alpha: alpha.clamp(0.0, 1.0), normal: n, specular: lin3(m.specular), emission: v3(m.illumination[0], m.illumination[1], m.illumination[2]), ambient: lin3(m.ambient) }
}

/// Light arriving at `pos` from `l`: (direction towards the light, radiance, distance).
pub fn light_at(l: &Light, pos: Vec3) -> Option<(Vec3, Vec3, f32)> {
    if !l.visible || l.intensity <= 0.0 {
        return None;
    }
    let color = lin3(l.color) * l.intensity;
    match l.kind {
        LightKind::Infinite => Some((-l.direction(), color, f32::INFINITY)),
        LightKind::Point | LightKind::Spot => {
            let d = l.position - pos;
            let dist = d.length();
            if dist < 1e-6 {
                return None;
            }
            let dir = d / dist;
            let mut k = 1.0;
            if l.falloff {
                k *= 1.0 - smoothstep(l.inner_radius, l.outer_radius, dist);
            }
            if l.kind == LightKind::Spot {
                let cos = (-dir).dot(l.direction());
                let outer = (l.cone.to_radians() * 0.5).cos();
                let inner = (l.hotspot.to_radians() * 0.5).cos().max(outer + 1e-4);
                k *= smoothstep(outer, inner, cos);
            }
            (k > 0.0).then_some((dir, color * k, dist))
        }
    }
}

pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if (e1 - e0).abs() < 1e-9 {
        return if x >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Environment (image-based light) radiance in direction `d`.
pub fn environment(scene: &Scene, d: Vec3) -> Vec3 {
    let env = &scene.environment;
    let k = env.ibl_intensity;
    if let Some(img) = scene.texture(env.ibl_texture) {
        let d = d.normalize();
        let u = 0.5 + d.x.atan2(-d.z) / std::f32::consts::TAU + env.ibl_rotation / 360.0;
        let v = 0.5 - d.y.clamp(-1.0, 1.0).asin() / std::f32::consts::PI;
        let c = img.sample([u, v.clamp(0.0, 1.0)]);
        return v3(c[0], c[1], c[2]) * k;
    }
    let t = (d.y * 0.5 + 0.5).clamp(0.0, 1.0);
    lin3(env.ground).lerp(lin3(env.sky), smoothstep(0.35, 0.65, t)) * k
}

/// Blinn–Phong specular exponent for a shine value.
pub fn exponent(shine: f32) -> f32 {
    2.0 + shine * shine * 600.0
}

/// Shade a surface point lit by the scene's lights. `visibility(light index, dir to light,
/// distance)` returns 0–1 (shadowing). `reflected` supplies what a mirror reflection sees (the
/// rasterizer passes the environment, the ray tracer traces). Returns linear RGB and alpha.
pub fn shade(scene: &Scene, s: &Surface<'_>, view: Vec3, visibility: &dyn Fn(usize, Vec3, f32) -> f32, reflected: &dyn Fn(Vec3, f32) -> Vec3) -> (Vec3, f32, Sampled) {
    let sm = sample_material(scene, s);
    let n = sm.normal;
    let mut c = sm.emission;
    // Ambient + image-based diffuse.
    let env = &scene.environment;
    c += sm.base.mul_elem(lin3(env.ambient) + sm.ambient);
    c += sm.base.mul_elem(environment(scene, n)) * 0.6;
    let m = s.material;
    let e = exponent(m.shine);
    let norm = (e + 8.0) / (8.0 * std::f32::consts::PI) * 0.6;
    for (i, l) in scene.lights.iter().enumerate() {
        let Some((ld, rad, dist)) = light_at(l, s.pos) else { continue };
        let ndl = n.dot(ld);
        if ndl <= 0.0 {
            continue;
        }
        let vis = if l.shadow { visibility(i, ld, dist) } else { 1.0 };
        if vis <= 0.0 {
            continue;
        }
        let h = (ld + view).normalize();
        let spec = n.dot(h).max(0.0).powf(e) * norm * m.shine.max(0.05);
        c += (sm.base * ndl + sm.specular * (spec * ndl)).mul_elem(rad) * vis;
    }
    if m.reflection > 0.0 {
        let r = (-view).reflect(n).normalize();
        let fres = m.reflection + (1.0 - m.reflection) * (1.0 - n.dot(view).max(0.0)).powi(5) * m.reflection;
        c = c * (1.0 - fres * 0.85) + reflected(r, m.roughness).mul_elem(v3(1.0, 1.0, 1.0).lerp(sm.base, 0.35)) * fres;
    }
    (c, sm.alpha, sm)
}

/// Exposure (stops) and the sRGB encode.
pub fn finish(c: Vec3, exposure: f32) -> [f32; 3] {
    let k = 2f32.powf(exposure);
    // Soft shoulder to keep highlights from clipping harshly.
    let tone = |v: f32| {
        let v = (v * k).max(0.0);
        if v <= 0.8 { v } else { 0.8 + 0.2 * (1.0 - (-(v - 0.8) / 0.2).exp()) }
    };
    [linear_to_srgb(tone(c.x)), linear_to_srgb(tone(c.y)), linear_to_srgb(tone(c.z))]
}

/// Is `p` removed by the cross section?
pub fn clipped(scene: &Scene, p: Vec3) -> bool {
    let r = &scene.render;
    if !r.cross_section {
        return false;
    }
    let v = match r.cross_axis {
        CrossAxis::X => p.x,
        CrossAxis::Y => p.y,
        CrossAxis::Z => p.z,
    } - r.cross_offset;
    if r.cross_flip { v < 0.0 } else { v > 0.0 }
}

// ---------------------------------------------------------------- camera projection

#[derive(Clone, Copy)]
pub struct Projector {
    pub view: Mat4,
    pub inv_view: Mat4,
    pub w: f32,
    pub h: f32,
    pub ortho: bool,
    /// tan(fov/2) (perspective) or half visible height (orthographic).
    pub half: f32,
    pub aspect: f32,
    pub near: f32,
}

impl Projector {
    pub fn new(scene: &Scene, w: u32, h: u32) -> Projector {
        let c = &scene.camera;
        let view = c.view();
        let aspect = w as f32 / h.max(1) as f32;
        let half = if c.orthographic { c.ortho_height * 0.5 } else { (c.fov.to_radians() * 0.5).tan() };
        Projector { view, inv_view: view.inverse(), w: w as f32, h: h as f32, ortho: c.orthographic, half, aspect, near: (c.distance * 1e-3).max(1e-4) }
    }
    /// View-space point → (screen x, screen y, depth = −z).
    pub fn project_view(&self, p: Vec3) -> (f32, f32, f32) {
        let d = -p.z;
        let (nx, ny) = if self.ortho { (p.x / (self.half * self.aspect), p.y / self.half) } else { (p.x / (d * self.half * self.aspect), p.y / (d * self.half)) };
        ((nx + 1.0) * 0.5 * self.w, (1.0 - ny) * 0.5 * self.h, d)
    }
    /// Camera ray through screen point `(x, y)`: (origin, direction), world space.
    pub fn ray(&self, x: f32, y: f32) -> (Vec3, Vec3) {
        let nx = x / self.w * 2.0 - 1.0;
        let ny = 1.0 - y / self.h * 2.0;
        if self.ortho {
            let o = v3(nx * self.half * self.aspect, ny * self.half, 0.0);
            (self.inv_view.transform_point(o), self.inv_view.transform_vector(-Vec3::Z).normalize())
        } else {
            let d = v3(nx * self.half * self.aspect, ny * self.half, -1.0);
            (self.inv_view.transform_point(Vec3::ZERO), self.inv_view.transform_vector(d).normalize())
        }
    }
    pub fn world_to_screen(&self, p: Vec3) -> Option<(f32, f32, f32)> {
        let v = self.view.transform_point(p);
        (-v.z > self.near).then(|| self.project_view(v))
    }
}

// ---------------------------------------------------------------- shadow maps

struct ShadowMap {
    m: Mat4,
    ortho: bool,
    half: f32,
    size: usize,
    depth: Vec<f32>,
    softness: f32,
}

impl ShadowMap {
    fn build(prep: &Prepared<'_>, l: &Light, size: usize) -> Option<ShadowMap> {
        let b = prep.bounds;
        if b.is_empty() || !l.shadow || !l.visible {
            return None;
        }
        let c = b.center();
        let r = (b.size().length() * 0.5).max(1e-3);
        let (m, ortho, half) = match l.kind {
            LightKind::Infinite => {
                let d = l.direction();
                let eye = c - d * (r * 3.0);
                (Mat4::look_at(eye, c, if d.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y }), true, r * 1.05)
            }
            LightKind::Spot => {
                let d = l.direction();
                let up = if d.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
                (Mat4::look_at(l.position, l.position + d, up), false, (l.cone.to_radians() * 0.5).tan().clamp(0.05, 20.0))
            }
            LightKind::Point => return None,
        };
        let mut sm = ShadowMap { m, ortho, half, size, depth: vec![f32::INFINITY; size * size], softness: l.softness };
        for t in prep.tris.iter().filter(|t| t.casts) {
            let v = t.p.map(|p| m.transform_point(p));
            if v.iter().any(|q| -q.z <= 1e-4) {
                continue;
            }
            let s = v.map(|q| sm.to_map(q));
            raster_depth(&mut sm.depth, size, s);
        }
        Some(sm)
    }
    fn to_map(&self, q: Vec3) -> (f32, f32, f32) {
        let d = -q.z;
        let (nx, ny) = if self.ortho { (q.x / self.half, q.y / self.half) } else { (q.x / (d * self.half), q.y / (d * self.half)) };
        ((nx + 1.0) * 0.5 * self.size as f32, (1.0 - ny) * 0.5 * self.size as f32, d)
    }
    /// Lit fraction at world point `p` with normal `n`.
    fn lit(&self, p: Vec3, n: Vec3, ld: Vec3) -> f32 {
        let q = self.m.transform_point(p + n * 0.002);
        if -q.z <= 1e-4 {
            return 1.0;
        }
        let (x, y, d) = self.to_map(q);
        let slope = (1.0 - n.dot(ld).clamp(0.0, 1.0)).sqrt();
        let bias = 0.004 * d.max(1.0) * (1.0 + 3.0 * slope);
        let k = (1.0 + self.softness * 6.0).round() as i32;
        let (mut lit, mut cnt) = (0.0, 0.0);
        let step = 1.0 + self.softness * 2.0;
        for j in -k..=k {
            for i in -k..=k {
                let sx = (x + i as f32 * step) as i64;
                let sy = (y + j as f32 * step) as i64;
                cnt += 1.0;
                if sx < 0 || sy < 0 || sx >= self.size as i64 || sy >= self.size as i64 {
                    lit += 1.0;
                    continue;
                }
                if d - bias <= self.depth[sy as usize * self.size + sx as usize] {
                    lit += 1.0;
                }
            }
        }
        lit / cnt
    }
}

fn raster_depth(buf: &mut [f32], size: usize, s: [(f32, f32, f32); 3]) {
    let minx = s.iter().map(|v| v.0).fold(f32::INFINITY, f32::min).floor().max(0.0) as i64;
    let maxx = s.iter().map(|v| v.0).fold(f32::NEG_INFINITY, f32::max).ceil().min(size as f32 - 1.0) as i64;
    let miny = s.iter().map(|v| v.1).fold(f32::INFINITY, f32::min).floor().max(0.0) as i64;
    let maxy = s.iter().map(|v| v.1).fold(f32::NEG_INFINITY, f32::max).ceil().min(size as f32 - 1.0) as i64;
    let area = edge(s[0], s[1], s[2].0, s[2].1);
    if area.abs() < 1e-12 || minx > maxx || miny > maxy {
        return;
    }
    for y in miny..=maxy {
        for x in minx..=maxx {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let w0 = edge(s[1], s[2], px, py) / area;
            let w1 = edge(s[2], s[0], px, py) / area;
            let w2 = 1.0 - w0 - w1;
            if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                continue;
            }
            let d = w0 * s[0].2 + w1 * s[1].2 + w2 * s[2].2;
            let i = y as usize * size + x as usize;
            if d < buf[i] {
                buf[i] = d;
            }
        }
    }
}

fn edge(a: (f32, f32, f32), b: (f32, f32, f32), x: f32, y: f32) -> f32 {
    (b.0 - a.0) * (y - a.1) - (b.1 - a.1) * (x - a.0)
}

// ---------------------------------------------------------------- rasterizer

/// A clipped, projected triangle.
struct STri {
    /// Screen x, y and 1/depth (perspective) or depth (orthographic) per corner.
    s: [(f32, f32, f32); 3],
    /// Barycentric weights of the original triangle at each corner (clipping makes new corners).
    bary: [Vec3; 3],
    src: u32,
    y0: f32,
    y1: f32,
}

#[derive(Clone, Copy)]
struct Frag {
    depth: f32,
    tri: u32,
    bary: Vec3,
}

/// Render the interactive preview. `aa` supersamples (1–3).
pub fn render_preview(scene: &Scene, width: u32, height: u32, aa: u32) -> Output {
    let (width, height) = (width.clamp(1, MAX_SIDE), height.clamp(1, MAX_SIDE));
    let aa = aa.clamp(1, 3);
    let (w, h) = ((width * aa).min(MAX_SIDE), (height * aa).min(MAX_SIDE));
    let prep = prepare(scene);
    let proj = Projector::new(scene, w, h);
    let stris = project_all(&prep, &proj);
    let frags = rasterize(&stris, w as usize, h as usize, proj.ortho);
    let shadows: Vec<Option<ShadowMap>> = scene.lights.iter().map(|l| ShadowMap::build(&prep, l, 1536)).collect();
    let big = shade_all(&prep, &proj, &stris, &frags, &shadows, w as usize, h as usize);
    let out = if aa == 1 { big } else { downsample(&big, w as usize, h as usize, aa as usize, width as usize, height as usize) };
    Output { width, height, pixels: out }
}

fn project_all(prep: &Prepared<'_>, proj: &Projector) -> Vec<STri> {
    prep.tris
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, t)| {
            let mut out = Vec::new();
            if !t.seen {
                return out;
            }
            let v = t.p.map(|p| proj.view.transform_point(p));
            let corners = [(v[0], Vec3::X), (v[1], Vec3::Y), (v[2], Vec3::Z)];
            for poly in clip_near(&corners, proj.near) {
                let s = poly.map(|(p, _)| {
                    let (x, y, d) = proj.project_view(p);
                    (x, y, if proj.ortho { d } else { 1.0 / d.max(1e-9) })
                });
                if s.iter().any(|c| !c.0.is_finite() || !c.1.is_finite()) {
                    continue;
                }
                let y0 = s.iter().map(|c| c.1).fold(f32::INFINITY, f32::min);
                let y1 = s.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
                out.push(STri { s, bary: poly.map(|(_, b)| b), src: i as u32, y0, y1 });
            }
            out
        })
        .collect()
}

/// Clip a view-space triangle against z = −near; returns 0–2 triangles with barycentrics.
fn clip_near(c: &[(Vec3, Vec3); 3], near: f32) -> Vec<[(Vec3, Vec3); 3]> {
    let inside = |p: Vec3| -p.z >= near;
    let mut poly: Vec<(Vec3, Vec3)> = Vec::with_capacity(4);
    for i in 0..3 {
        let (a, b) = (c[i], c[(i + 1) % 3]);
        if inside(a.0) {
            poly.push(a);
        }
        if inside(a.0) != inside(b.0) {
            let t = (-near - a.0.z) / (b.0.z - a.0.z);
            poly.push((a.0.lerp(b.0, t), a.1.lerp(b.1, t)));
        }
    }
    let mut out = Vec::new();
    for k in 1..poly.len().saturating_sub(1) {
        out.push([poly[0], poly[k], poly[k + 1]]);
    }
    out
}

const BAND: usize = 16;

fn rasterize(stris: &[STri], w: usize, h: usize, ortho: bool) -> Vec<Option<Frag>> {
    let mut frags: Vec<Option<Frag>> = vec![None; w * h];
    frags.par_chunks_mut(w * BAND).enumerate().for_each(|(band, rows)| {
        let by0 = (band * BAND) as f32;
        let by1 = by0 + BAND as f32;
        let rows_n = rows.len() / w;
        for (k, t) in stris.iter().enumerate() {
            if t.y1 < by0 || t.y0 > by1 {
                continue;
            }
            let s = t.s;
            let area = edge(s[0], s[1], s[2].0, s[2].1);
            if area.abs() < 1e-12 {
                continue;
            }
            let minx = s.iter().map(|v| v.0).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
            let maxx = (s.iter().map(|v| v.0).fold(f32::NEG_INFINITY, f32::max).ceil().max(0.0) as usize).min(w.saturating_sub(1));
            let ya = (t.y0.floor().max(by0) as usize).saturating_sub(band * BAND);
            let yb = ((t.y1.ceil().min(by1 - 1.0)).max(0.0) as usize).saturating_sub(band * BAND).min(rows_n.saturating_sub(1));
            if minx > maxx {
                continue;
            }
            for ly in ya..=yb {
                let py = (band * BAND + ly) as f32 + 0.5;
                for x in minx..=maxx {
                    let px = x as f32 + 0.5;
                    let w0 = edge(s[1], s[2], px, py) / area;
                    let w1 = edge(s[2], s[0], px, py) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    // Depth: orthographic interpolates depth linearly; perspective interpolates 1/z.
                    let (depth, pw) = if ortho {
                        (w0 * s[0].2 + w1 * s[1].2 + w2 * s[2].2, [w0, w1, w2])
                    } else {
                        let iz = w0 * s[0].2 + w1 * s[1].2 + w2 * s[2].2;
                        let d = 1.0 / iz.max(1e-12);
                        (d, [w0 * s[0].2 * d, w1 * s[1].2 * d, w2 * s[2].2 * d])
                    };
                    let slot = &mut rows[ly * w + x];
                    if slot.is_none_or(|f| depth < f.depth) {
                        let bary = t.bary[0] * pw[0] + t.bary[1] * pw[1] + t.bary[2] * pw[2];
                        *slot = Some(Frag { depth, tri: k as u32, bary });
                    }
                }
            }
        }
    });
    frags
}

fn shade_all(prep: &Prepared<'_>, proj: &Projector, stris: &[STri], frags: &[Option<Frag>], shadows: &[Option<ShadowMap>], w: usize, h: usize) -> Vec<[f32; 4]> {
    let scene = prep.scene;
    let style = scene.render.style;
    let exposure = scene.render.exposure;
    let (dmin, dmax) = depth_range(prep, proj);
    let line = lin3(scene.render.line_color);
    let mut out = vec![[0.0f32; 4]; w * h];
    // Pass 1: surfaces and ground.
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, px) in row.iter_mut().enumerate() {
            let frag = frags[y * w + x];
            let (ro, rd) = proj.ray(x as f32 + 0.5, y as f32 + 0.5);
            let bg = background(scene, rd);
            let mut surf: Option<([f32; 3], f32, f32)> = None; // (srgb colour, alpha, depth)
            if let Some(f) = frag
                && let Some(st) = stris.get(f.tri as usize)
                && let Some(t) = prep.tris.get(st.src as usize)
            {
                let pos = t.p[0] * f.bary.x + t.p[1] * f.bary.y + t.p[2] * f.bary.z;
                if !clipped(scene, pos) {
                    let mut n = (t.n[0] * f.bary.x + t.n[1] * f.bary.y + t.n[2] * f.bary.z).normalize();
                    let view = (ro - pos).normalize_or(-rd);
                    let geo = (t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]).normalize();
                    if geo.dot(view) < 0.0 {
                        n = -n;
                    }
                    if n.dot(view) < 0.0 {
                        n = (n - view * (2.0 * n.dot(view))).normalize();
                    }
                    let uv = [
                        t.uv[0][0] * f.bary.x + t.uv[1][0] * f.bary.y + t.uv[2][0] * f.bary.z,
                        t.uv[0][1] * f.bary.x + t.uv[1][1] * f.bary.y + t.uv[2][1] * f.bary.z,
                    ];
                    let material = prep.materials.get(t.mat as usize).copied().unwrap_or(&DEFAULT_MATERIAL);
                    let s = Surface { pos, normal: n, uv, tri: t, material };
                    let (c, a) = match style {
                        Style::Solid | Style::ShadedWireframe => {
                            let vis = |i: usize, ld: Vec3, _d: f32| shadows.get(i).and_then(|m| m.as_ref()).map_or(1.0, |m| m.lit(pos, n, ld));
                            let refl = |r: Vec3, _rough: f32| environment(scene, r);
                            let (c, a, _) = shade(scene, &s, view, &vis, &refl);
                            (finish(c, exposure), a)
                        }
                        Style::Unlit | Style::LineIllustration => {
                            let sm = sample_material(scene, &s);
                            let c = if style == Style::LineIllustration { v3(1.0, 1.0, 1.0) } else { sm.base };
                            (finish(c, 0.0), sm.alpha)
                        }
                        Style::Normals => {
                            let wn = sample_material(scene, &s).normal;
                            ([wn.x * 0.5 + 0.5, wn.y * 0.5 + 0.5, wn.z * 0.5 + 0.5], 1.0)
                        }
                        Style::DepthMap => {
                            let g = 1.0 - ((f.depth - dmin) / (dmax - dmin).max(1e-6)).clamp(0.0, 1.0);
                            ([g, g, g], 1.0)
                        }
                        Style::Wireframe | Style::Points => ([0.0, 0.0, 0.0], 0.0),
                    };
                    surf = Some((c, a, f.depth));
                }
            }
            // Ground-plane shadow catcher behind or without a surface.
            let shadow = ground_shadow(prep, shadows, ro, rd, surf.map(|s| s.2), proj);
            let mut base = bg;
            if shadow > 0.0 {
                base = over([0.0, 0.0, 0.0, shadow * scene.environment.ground_shadow_opacity], base);
            }
            *px = match surf {
                Some((c, a, _)) => over([c[0], c[1], c[2], a], base),
                None => base,
            };
        }
    });
    // Pass 2: lines and points.
    match style {
        Style::Wireframe | Style::ShadedWireframe => draw_wire(&mut out, stris, frags, w, h, line, style == Style::Wireframe),
        Style::LineIllustration => draw_edges(&mut out, frags, prep, stris, w, h, line),
        Style::Points => draw_points(&mut out, prep, proj, frags, w, h, line),
        _ => {}
    }
    out
}

fn depth_range(prep: &Prepared<'_>, proj: &Projector) -> (f32, f32) {
    let b = prep.bounds;
    if b.is_empty() {
        return (0.0, 1.0);
    }
    let ds: Vec<f32> = b.corners().iter().map(|&c| -proj.view.transform_point(c).z).collect();
    let lo = ds.iter().copied().fold(f32::INFINITY, f32::min).max(proj.near);
    let hi = ds.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (lo, hi.max(lo + 1e-3))
}

/// Straight-alpha "over".
pub fn over(top: [f32; 4], bottom: [f32; 4]) -> [f32; 4] {
    let a = top[3] + bottom[3] * (1.0 - top[3]);
    if a <= 1e-6 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let mix = |t: f32, b: f32| (t * top[3] + b * bottom[3] * (1.0 - top[3])) / a;
    [mix(top[0], bottom[0]), mix(top[1], bottom[1]), mix(top[2], bottom[2]), a]
}

/// What the camera sees with nothing in front.
pub fn background(scene: &Scene, rd: Vec3) -> [f32; 4] {
    match scene.environment.background {
        Background::Transparent => [0.0, 0.0, 0.0, 0.0],
        Background::Color => {
            let c = scene.environment.background_color;
            [c[0], c[1], c[2], 1.0]
        }
        Background::Environment => {
            let c = finish(environment(scene, rd), 0.0);
            [c[0], c[1], c[2], 1.0]
        }
    }
}

fn ground_shadow(prep: &Prepared<'_>, shadows: &[Option<ShadowMap>], ro: Vec3, rd: Vec3, surface_depth: Option<f32>, proj: &Projector) -> f32 {
    let scene = prep.scene;
    if scene.environment.ground_shadow_opacity <= 0.0 || rd.y.abs() < 1e-6 {
        return 0.0;
    }
    let t = (scene.environment.ground_y - ro.y) / rd.y;
    if t <= 0.0 {
        return 0.0;
    }
    let g = ro + rd * t;
    if let Some(sd) = surface_depth {
        let gd = -proj.view.transform_point(g).z;
        if gd >= sd {
            return 0.0;
        }
    }
    let (mut lit, mut total) = (0.0, 0.0);
    for (i, l) in scene.lights.iter().enumerate() {
        let Some((ld, rad, _)) = light_at(l, g) else { continue };
        let ndl = ld.y.max(0.0);
        let weight = (rad.x + rad.y + rad.z) / 3.0 * ndl;
        if weight <= 0.0 {
            continue;
        }
        total += weight;
        let v = if l.shadow { shadows.get(i).and_then(|m| m.as_ref()).map_or(1.0, |m| m.lit(g, Vec3::Y, ld)) } else { 1.0 };
        lit += weight * v;
    }
    if total <= 0.0 { 0.0 } else { (1.0 - lit / total).clamp(0.0, 1.0) }
}

fn downsample(big: &[[f32; 4]], bw: usize, _bh: usize, aa: usize, w: usize, h: usize) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, px) in row.iter_mut().enumerate() {
            let mut acc = [0.0f32; 4];
            for j in 0..aa {
                for i in 0..aa {
                    let p = big[(y * aa + j) * bw + x * aa + i];
                    // Premultiplied average.
                    acc[0] += p[0] * p[3];
                    acc[1] += p[1] * p[3];
                    acc[2] += p[2] * p[3];
                    acc[3] += p[3];
                }
            }
            let n = (aa * aa) as f32;
            *px = if acc[3] > 1e-6 { [acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3], acc[3] / n] } else { [0.0; 4] };
        }
    });
    out
}

fn put(out: &mut [[f32; 4]], w: usize, h: usize, x: i64, y: i64, c: Vec3, a: f32) {
    if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
        return;
    }
    let i = y as usize * w + x as usize;
    let col = finish(c, 0.0);
    out[i] = over([col[0], col[1], col[2], a], out[i]);
}

/// Edges of visible triangles (barycentric distance in screen space).
fn draw_wire(out: &mut [[f32; 4]], stris: &[STri], frags: &[Option<Frag>], w: usize, h: usize, line: Vec3, all: bool) {
    let _ = h;
    let _ = all;
    let lines: Vec<(usize, f32)> = frags
        .par_iter()
        .enumerate()
        .filter_map(|(i, f)| {
            let f = (*f)?;
            let st = stris.get(f.tri as usize)?;
            let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
            let s = st.s;
            let mut best = f32::INFINITY;
            for k in 0..3 {
                let (a, b) = (s[k], s[(k + 1) % 3]);
                // Skip edges introduced by near clipping (both corners on a clip boundary would
                // be rare; accept the extra line).
                let d = seg_dist2(x, y, a.0, a.1, b.0, b.1);
                best = best.min(d);
            }
            (best < 1.0).then(|| (i, 1.0 - best))
        })
        .collect();
    for (i, cov) in lines {
        let col = finish(line, 0.0);
        out[i] = over([col[0], col[1], col[2], cov.clamp(0.0, 1.0)], out[i]);
    }
}

fn seg_dist2(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let (dx, dy) = (bx - ax, by - ay);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 1e-12 { (((px - ax) * dx + (py - ay) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    ((px - ax - t * dx).powi(2) + (py - ay - t * dy).powi(2)).sqrt()
}

/// Line Illustration: lines where depth or normal jump between neighbouring pixels.
fn draw_edges(out: &mut [[f32; 4]], frags: &[Option<Frag>], prep: &Prepared<'_>, stris: &[STri], w: usize, h: usize, line: Vec3) {
    let normal_at = |i: usize| -> Option<(Vec3, f32)> {
        let f = frags.get(i).copied().flatten()?;
        let t = prep.tris.get(stris.get(f.tri as usize)?.src as usize)?;
        Some(((t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]).normalize(), f.depth))
    };
    let hits: Vec<usize> = (0..w * h)
        .into_par_iter()
        .filter(|&i| {
            let (x, y) = (i % w, i / w);
            let here = normal_at(i);
            let mut edge = false;
            for (dx, dy) in [(1i64, 0i64), (0, 1)] {
                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                if nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let there = normal_at(ny as usize * w + nx as usize);
                edge |= match (here, there) {
                    (Some((n1, d1)), Some((n2, d2))) => n1.dot(n2) < 0.8 || (d1 - d2).abs() > 0.02 * d1.max(d2),
                    (None, None) => false,
                    _ => true,
                };
            }
            edge
        })
        .collect();
    let col = finish(line, 0.0);
    for i in hits {
        out[i] = over([col[0], col[1], col[2], 1.0], out[i]);
    }
}

fn draw_points(out: &mut [[f32; 4]], prep: &Prepared<'_>, proj: &Projector, frags: &[Option<Frag>], w: usize, h: usize, line: Vec3) {
    let mut seen = std::collections::HashSet::new();
    for t in prep.tris.iter().filter(|t| t.seen) {
        for p in t.p {
            let key = ((p.x * 1e4) as i64, (p.y * 1e4) as i64, (p.z * 1e4) as i64);
            if !seen.insert(key) {
                continue;
            }
            let Some((x, y, d)) = proj.world_to_screen(p) else { continue };
            let (xi, yi) = (x.floor() as i64, y.floor() as i64);
            let front = xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h && frags[yi as usize * w + xi as usize].is_none_or(|f| d <= f.depth * 1.01 + 1e-3);
            if front {
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    put(out, w, h, xi + dx, yi + dy, line, 1.0);
                }
            }
        }
    }
}

/// Render a small RGBA8 thumbnail of a single material on a sphere (the material preview).
pub fn material_swatch(material: &Material, textures: &[crate::scene::Texture], size: u32) -> Vec<u8> {
    let mut s = Scene::default();
    s.textures = textures.to_vec();
    s.environment.ground_shadow_opacity = 0.0;
    s.add_object("Sphere", crate::scene::Geometry::Preset { preset: crate::mesh::MeshPreset::Sphere, aspect: 1.0 }, vec![material.clone()]);
    s.camera.distance = 1.0 / (s.camera.fov.to_radians() * 0.5).tan() * 0.55;
    crate::scene::apply_light_preset(&mut s, "threePoint");
    render_preview(&s, size, size, 2).to_rgba8()
}

/// Make an [`Image`] from rendered output (for using a render as a texture).
pub fn to_image(o: &Output) -> Option<Image> {
    Image::new(o.width, o.height, o.to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::MeshPreset;
    use crate::scene::{Geometry, Material};

    fn cube_scene() -> Scene {
        let mut s = Scene::default();
        let m = Material { diffuse: [0.9, 0.1, 0.1], ..Default::default() };
        s.add_object("Cube", Geometry::Preset { preset: MeshPreset::Cube, aspect: 1.0 }, vec![m]);
        s.camera.yaw = 30.0;
        s.camera.pitch = 20.0;
        s.fit_ground();
        s
    }

    #[test]
    fn renders_a_red_cube_in_the_middle() {
        let s = cube_scene();
        let o = render_preview(&s, 64, 48, 1);
        let c = o.pixels[24 * 64 + 32];
        assert!(c[3] > 0.99, "covered");
        assert!(c[0] > c[1] && c[0] > c[2], "red: {c:?}");
        let corner = o.pixels[0];
        assert!(corner[3] < 0.01, "transparent background");
    }

    #[test]
    fn every_style_renders() {
        let mut s = cube_scene();
        for st in Style::ALL {
            s.render.style = st;
            let o = render_preview(&s, 40, 30, 1);
            assert!(o.pixels.iter().any(|p| p[3] > 0.0), "{st:?}");
        }
    }

    #[test]
    fn camera_inside_object_and_empty_scene_do_not_panic() {
        let mut s = cube_scene();
        s.camera.distance = 0.01;
        let _ = render_preview(&s, 16, 16, 2);
        let _ = render_preview(&Scene::default(), 16, 16, 1);
    }

    #[test]
    fn swatch_is_opaque_in_the_middle() {
        let px = material_swatch(&Material::default(), &[], 32);
        assert_eq!(px.len(), 32 * 32 * 4);
        assert!(px[(16 * 32 + 16) * 4 + 3] > 250);
    }
}

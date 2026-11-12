//! Ray tracer for 3D › Render 3D Layer: a BVH over the scene's triangles, soft shadows from
//! every light (point lights too), mirror and rough reflections, refraction through transparent
//! materials, ground-plane shadows and reflections, depth of field and stratified antialiasing.
//! Deterministic: the random numbers are hashed from pixel and sample indices.

use rayon::prelude::*;

use crate::math::{Vec3, v3};
use crate::mesh::Aabb;
use crate::render::{Output, Prepared, Projector, Surface, background, clipped, environment, finish, light_at, prepare, shade};
use crate::scene::{LightKind, Scene};

const LEAF: usize = 4;

struct Node {
    b: Aabb,
    /// Leaf: first triangle index and count; inner: left child index (right = left + 1).
    start: u32,
    count: u32,
    left: u32,
}

pub struct Bvh {
    nodes: Vec<Node>,
    order: Vec<u32>,
}

impl Bvh {
    pub fn build(prep: &Prepared<'_>) -> Bvh {
        let n = prep.tris.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let cent: Vec<Vec3> = prep.tris.iter().map(|t| (t.p[0] + t.p[1] + t.p[2]) / 3.0).collect();
        let tb: Vec<Aabb> = prep
            .tris
            .iter()
            .map(|t| {
                let mut b = Aabb::EMPTY;
                for p in t.p {
                    b.add(p);
                }
                b
            })
            .collect();
        let mut nodes = vec![Node { b: Aabb::EMPTY, start: 0, count: n as u32, left: 0 }];
        let mut stack = vec![0usize];
        while let Some(ni) = stack.pop() {
            let (start, count) = (nodes[ni].start as usize, nodes[ni].count as usize);
            let mut b = Aabb::EMPTY;
            let mut cb = Aabb::EMPTY;
            for &t in &order[start..start + count] {
                b = b.union(&tb[t as usize]);
                cb.add(cent[t as usize]);
            }
            nodes[ni].b = b;
            if count <= LEAF {
                continue;
            }
            let size = cb.size();
            let axis = if size.x >= size.y && size.x >= size.z { 0 } else if size.y >= size.z { 1 } else { 2 };
            let slice = &mut order[start..start + count];
            slice.sort_unstable_by(|a, c| cent[*a as usize].get(axis).total_cmp(&cent[*c as usize].get(axis)));
            let mid = count / 2;
            let left = nodes.len();
            nodes.push(Node { b: Aabb::EMPTY, start: start as u32, count: mid as u32, left: 0 });
            nodes.push(Node { b: Aabb::EMPTY, start: (start + mid) as u32, count: (count - mid) as u32, left: 0 });
            nodes[ni].left = left as u32;
            nodes[ni].count = 0;
            stack.push(left);
            stack.push(left + 1);
        }
        Bvh { nodes, order }
    }
}

#[derive(Clone, Copy)]
struct Hit {
    t: f32,
    tri: u32,
    u: f32,
    v: f32,
}

fn ray_box(o: Vec3, inv: Vec3, b: &Aabb, tmax: f32) -> bool {
    let mut t0: f32 = 0.0;
    let mut t1 = tmax;
    for a in 0..3 {
        let (oa, ia) = (o.get(a), inv.get(a));
        let mut ta = (b.min.get(a) - oa) * ia;
        let mut tb = (b.max.get(a) - oa) * ia;
        if ta > tb {
            std::mem::swap(&mut ta, &mut tb);
        }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
        if t0 > t1 {
            return false;
        }
    }
    true
}

struct Tracer<'a> {
    prep: Prepared<'a>,
    bvh: Bvh,
}

impl Tracer<'_> {
    /// Nearest hit along the ray (camera-visible triangles only when `seen_only`).
    fn intersect(&self, o: Vec3, d: Vec3, tmax: f32, seen_only: bool, shadow: bool) -> Option<Hit> {
        if self.bvh.nodes.is_empty() || self.prep.tris.is_empty() {
            return None;
        }
        let inv = v3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        let mut best: Option<Hit> = None;
        let mut tmax = tmax;
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let Some(node) = self.bvh.nodes.get(ni as usize) else { continue };
            if !ray_box(o, inv, &node.b, tmax) {
                continue;
            }
            if node.count > 0 {
                for &ti in &self.bvh.order[node.start as usize..(node.start + node.count) as usize] {
                    let t = &self.prep.tris[ti as usize];
                    if (seen_only && !t.seen) || (shadow && !t.casts) {
                        continue;
                    }
                    if let Some((tt, u, v)) = tri_hit(o, d, t.p[0], t.p[1], t.p[2])
                        && tt < tmax
                        && tt > 1e-4
                    {
                        let p = o + d * tt;
                        if clipped(self.prep.scene, p) {
                            continue;
                        }
                        tmax = tt;
                        best = Some(Hit { t: tt, tri: ti, u, v });
                        if shadow {
                            return best;
                        }
                    }
                }
            } else {
                stack.push(node.left);
                stack.push(node.left + 1);
            }
        }
        best
    }
}

/// Möller–Trumbore; returns (t, u, v).
fn tri_hit(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<(f32, f32, f32)> {
    let e1 = b - a;
    let e2 = c - a;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some((e2.dot(q) * inv, u, v))
}

/// Hash-based random number in [0, 1).
fn rnd(a: u32, b: u32, c: u32) -> f32 {
    let mut h = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_mul(0x85EB_CA77) ^ c.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// A unit vector near `d` within a cone of half-angle `spread` radians.
fn jitter(d: Vec3, spread: f32, r1: f32, r2: f32) -> Vec3 {
    if spread <= 0.0 {
        return d;
    }
    let t = if d.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = d.cross(t).normalize();
    let v = d.cross(u);
    let a = r1 * std::f32::consts::TAU;
    let r = spread * r2.sqrt();
    (d + (u * a.cos() + v * a.sin()) * r.tan()).normalize()
}

struct Ctx {
    px: u32,
    sample: u32,
}

impl Tracer<'_> {
    /// Radiance and coverage along a ray. `depth` counts bounces.
    fn radiance(&self, o: Vec3, d: Vec3, depth: u32, cx: &Ctx, camera_ray: bool) -> (Vec3, f32) {
        let scene = self.prep.scene;
        let hit = self.intersect(o, d, f32::INFINITY, true, false);
        let ground_t = if d.y.abs() > 1e-6 { Some((scene.environment.ground_y - o.y) / d.y).filter(|t| *t > 1e-4) } else { None };
        let ground_first = match (ground_t, hit) {
            (Some(g), Some(h)) => g < h.t,
            (Some(_), None) => true,
            _ => false,
        };
        let bg = || -> (Vec3, f32) {
            if camera_ray {
                let b = background(scene, d);
                (srgb_vec(b), b[3])
            } else {
                (environment(scene, d), 1.0)
            }
        };
        if ground_first && let Some(gt) = ground_t {
            // Ground plane: shadow catcher and reflections over whatever is beyond it.
            let g = o + d * gt;
            let (behind, behind_a) = bg();
            let env = &scene.environment;
            let shadow = self.ground_shadow(g, cx);
            let mut col = behind;
            let mut alpha = behind_a;
            if env.ground_reflection_opacity > 0.0 && depth < scene.render.bounces {
                let r = d.reflect(Vec3::Y);
                if self.intersect(g + Vec3::Y * 1e-3, r, f32::INFINITY, true, false).is_some() {
                    let (rc, ra) = self.radiance(g + Vec3::Y * 1e-3, r, depth + 1, cx, false);
                    let k = env.ground_reflection_opacity * ra;
                    col = col * (1.0 - k) + rc * k;
                    alpha = alpha + (1.0 - alpha) * k;
                }
            }
            let s = shadow * env.ground_shadow_opacity;
            col = col * (1.0 - s);
            alpha = alpha + (1.0 - alpha) * s;
            return (col, alpha);
        }
        let Some(h) = hit else { return bg() };
        let t = &self.prep.tris[h.tri as usize];
        let w = 1.0 - h.u - h.v;
        let pos = o + d * h.t;
        let mut n = (t.n[0] * w + t.n[1] * h.u + t.n[2] * h.v).normalize();
        let geo = (t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]).normalize();
        let view = -d;
        let entering = geo.dot(view) >= 0.0;
        if !entering {
            n = -n;
        }
        if n.dot(view) < 0.0 {
            n = (n - view * (2.0 * n.dot(view))).normalize();
        }
        let gface = if entering { geo } else { -geo };
        let uv = [t.uv[0][0] * w + t.uv[1][0] * h.u + t.uv[2][0] * h.v, t.uv[0][1] * w + t.uv[1][1] * h.u + t.uv[2][1] * h.v];
        let material = self.prep.materials[t.mat as usize];
        let s = Surface { pos, normal: n, uv, tri: t, material };
        let off = gface * 1e-3;
        let vis = |li: usize, ld: Vec3, dist: f32| self.light_visibility(li, pos + off, ld, dist, cx);
        let refl = |r: Vec3, rough: f32| -> Vec3 {
            if depth >= scene.render.bounces {
                return environment(scene, r);
            }
            let r = jitter(r, rough * 0.35, rnd(cx.px, cx.sample, 11 + depth), rnd(cx.px, cx.sample, 13 + depth));
            self.radiance(pos + off, r, depth + 1, cx, false).0
        };
        let (mut c, alpha, sm) = shade(scene, &s, view, &vis, &refl);
        let mut a = alpha;
        if alpha < 0.999 && depth < scene.render.bounces + 2 {
            // Refraction through the surface.
            let eta = if entering { 1.0 / material.refraction } else { material.refraction };
            let ni = sm.normal;
            let cosi = ni.dot(view).clamp(-1.0, 1.0);
            let k = 1.0 - eta * eta * (1.0 - cosi * cosi);
            let dir = if k < 0.0 || (material.refraction - 1.0).abs() < 1e-4 { d } else { (d * eta + ni * (eta * cosi - k.sqrt())).normalize() };
            let (tc, ta) = self.radiance(pos - off, dir, depth + 1, cx, camera_ray);
            // Transparent colour filters what's behind it by the diffuse colour.
            let tint = v3(1.0, 1.0, 1.0).lerp(sm.base, alpha.max(0.25));
            c = c * alpha + tc.mul_elem(tint) * (1.0 - alpha);
            a = alpha + (1.0 - alpha) * ta;
        }
        (c, a)
    }

    fn light_visibility(&self, li: usize, p: Vec3, ld: Vec3, dist: f32, cx: &Ctx) -> f32 {
        let Some(l) = self.prep.scene.lights.get(li) else { return 1.0 };
        let (r1, r2) = (rnd(cx.px, cx.sample, 101 + li as u32 * 7), rnd(cx.px, cx.sample, 103 + li as u32 * 7));
        let (dir, maxd) = match l.kind {
            LightKind::Infinite => (jitter(ld, l.softness * 0.12, r1, r2), f32::INFINITY),
            _ => {
                // Area light of radius ∝ softness around the light's position.
                let target = l.position + jitter_offset(l.softness * 0.4, r1, r2, rnd(cx.px, cx.sample, 107 + li as u32));
                let v = target - p;
                let len = v.length();
                (v / len.max(1e-6), len.min(dist * 2.0))
            }
        };
        if self.intersect(p, dir, maxd, false, true).is_some() { 0.0 } else { 1.0 }
    }

    fn ground_shadow(&self, g: Vec3, cx: &Ctx) -> f32 {
        let (mut lit, mut total) = (0.0, 0.0);
        for (i, l) in self.prep.scene.lights.iter().enumerate() {
            let Some((ld, rad, dist)) = light_at(l, g) else { continue };
            let wgt = (rad.x + rad.y + rad.z) / 3.0 * ld.y.max(0.0);
            if wgt <= 0.0 {
                continue;
            }
            total += wgt;
            lit += wgt * if l.shadow { self.light_visibility(i, g + Vec3::Y * 1e-3, ld, dist, cx) } else { 1.0 };
        }
        if total <= 0.0 { 0.0 } else { 1.0 - lit / total }
    }
}

fn jitter_offset(r: f32, a: f32, b: f32, c: f32) -> Vec3 {
    v3(a * 2.0 - 1.0, b * 2.0 - 1.0, c * 2.0 - 1.0) * r
}

fn srgb_vec(c: [f32; 4]) -> Vec3 {
    v3(crate::math::srgb_to_linear(c[0]), crate::math::srgb_to_linear(c[1]), crate::math::srgb_to_linear(c[2]))
}

/// Ray trace the scene. `cancel` is polled per row (return true to stop: the result is `None`);
/// `progress` receives 0–1.
pub fn render_traced(scene: &Scene, width: u32, height: u32, cancel: &(dyn Fn() -> bool + Sync), progress: &(dyn Fn(f32) + Sync)) -> Option<Output> {
    let (width, height) = (width.clamp(1, crate::render::MAX_SIDE), height.clamp(1, crate::render::MAX_SIDE));
    let prep = prepare(scene);
    let bvh = Bvh::build(&prep);
    let tracer = Tracer { prep, bvh };
    let proj = Projector::new(scene, width, height);
    let samples = scene.render.samples.clamp(1, 256);
    let side = (samples as f32).sqrt().ceil() as u32;
    let cam = &scene.camera;
    let done = std::sync::atomic::AtomicU32::new(0);
    let stop = std::sync::atomic::AtomicBool::new(false);
    let w = width as usize;
    let mut pixels = vec![[0.0f32; 4]; w * height as usize];
    pixels.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        if cancel() {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        for (x, px) in row.iter_mut().enumerate() {
            let id = (y * w + x) as u32;
            let (mut acc, mut acc_a) = (Vec3::ZERO, 0.0f32);
            for sidx in 0..samples {
                let (sx, sy) = (sidx % side, sidx / side);
                let jx = (sx as f32 + rnd(id, sidx, 1)) / side as f32;
                let jy = (sy as f32 + rnd(id, sidx, 2)) / side as f32;
                let (mut o, mut d) = proj.ray(x as f32 + jx, y as f32 + jy);
                if cam.dof_blur > 0.0 && !cam.orthographic {
                    // Thin lens: aperture ∝ blur, focused at dof_distance along the view axis.
                    let fwd = (cam.target - cam.eye()).normalize();
                    let focus = o + d * (cam.dof_distance / d.dot(fwd).max(1e-3));
                    let up = cam.up();
                    let right = fwd.cross(up).normalize();
                    let a = rnd(id, sidx, 3) * std::f32::consts::TAU;
                    let r = rnd(id, sidx, 4).sqrt() * cam.dof_blur * 0.08 * cam.distance;
                    o = o + (right * a.cos() + up * a.sin()) * r;
                    d = (focus - o).normalize();
                }
                let cx = Ctx { px: id, sample: sidx };
                let (c, a) = tracer.radiance(o, d, 0, &cx, true);
                // Premultiplied accumulation.
                acc += c * a;
                acc_a += a;
            }
            let n = samples as f32;
            let a = acc_a / n;
            let c = if acc_a > 1e-6 { finish(acc / acc_a, scene.render.exposure) } else { [0.0, 0.0, 0.0] };
            *px = [c[0], c[1], c[2], a.clamp(0.0, 1.0)];
        }
        let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        progress(d as f32 / height as f32);
    });
    if stop.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    Some(Output { width, height, pixels })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::MeshPreset;
    use crate::scene::{Geometry, Material};

    #[test]
    fn ray_traced_sphere_matches_preview_coverage() {
        let mut s = Scene::default();
        s.add_object("Sphere", Geometry::Preset { preset: MeshPreset::Sphere, aspect: 1.0 }, vec![Material { diffuse: [0.1, 0.8, 0.1], ..Default::default() }]);
        s.render.samples = 4;
        s.fit_ground();
        let t = render_traced(&s, 40, 40, &|| false, &|_| {}).unwrap();
        let p = render_preview_ref(&s);
        let centre = t.pixels[20 * 40 + 20];
        assert!(centre[3] > 0.99 && centre[1] > centre[0]);
        let cov_t: f32 = t.pixels.iter().map(|p| if p[1] > p[0] && p[3] > 0.5 { 1.0 } else { 0.0 }).sum();
        let cov_p: f32 = p.pixels.iter().map(|p| if p[1] > p[0] && p[3] > 0.5 { 1.0 } else { 0.0 }).sum();
        assert!((cov_t - cov_p).abs() / cov_p < 0.15, "{cov_t} vs {cov_p}");
    }

    fn render_preview_ref(s: &Scene) -> Output {
        crate::render::render_preview(s, 40, 40, 1)
    }

    #[test]
    fn cancel_stops_the_render() {
        let mut s = Scene::default();
        s.add_object("Cube", Geometry::Preset { preset: MeshPreset::Cube, aspect: 1.0 }, vec![Material::default()]);
        assert!(render_traced(&s, 32, 32, &|| true, &|_| {}).is_none());
    }

    #[test]
    fn point_light_casts_ground_shadow() {
        let mut s = Scene::default();
        s.lights.clear();
        let id = s.fresh_id();
        s.lights.push(crate::scene::Light { id, kind: LightKind::Point, position: v3(0.0, 3.0, 0.0), softness: 0.0, ..Default::default() });
        s.add_object("Cube", Geometry::Preset { preset: MeshPreset::Cube, aspect: 1.0 }, vec![Material::default()]);
        s.camera.pitch = 60.0;
        s.camera.distance = 6.0;
        s.render.samples = 1;
        s.fit_ground();
        let t = render_traced(&s, 32, 32, &|| false, &|_| {}).unwrap();
        // Somewhere there's a semi-transparent black shadow on the ground.
        assert!(t.pixels.iter().any(|p| p[3] > 0.2 && p[3] < 0.99 && p[0] < 0.05));
    }
}

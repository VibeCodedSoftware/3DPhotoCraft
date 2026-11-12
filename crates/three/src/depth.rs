//! 3D › New Mesh from Layer › Depth Map to: a grayscale image displaces a plane, a two-sided
//! plane, a cylinder or a sphere (light = near / raised).

use std::f32::consts::{PI, TAU};

use crate::math::{Vec3, v3};
use crate::mesh::Mesh;
use crate::scene::DepthMapKind;

/// Largest grid side kept for a depth map.
pub const MAX_GRID: usize = 256;

/// Downsample a `w × h` gray image (0–1, row-major) to at most [`MAX_GRID`] on its long side.
pub fn grid_from(gray: &[f32], w: usize, h: usize) -> (u32, u32, Vec<f32>) {
    if w == 0 || h == 0 || gray.len() < w * h {
        return (2, 2, vec![0.0; 4]);
    }
    let s = (w.max(h) as f32 / MAX_GRID as f32).max(1.0);
    let (gw, gh) = (((w as f32 / s).round() as usize).max(2), ((h as f32 / s).round() as usize).max(2));
    let mut out = vec![0.0; gw * gh];
    for gy in 0..gh {
        for gx in 0..gw {
            // Box average of the source pixels this sample covers.
            let x0 = (gx as f32 * w as f32 / gw as f32) as usize;
            let x1 = (((gx + 1) as f32 * w as f32 / gw as f32) as usize).clamp(x0 + 1, w);
            let y0 = (gy as f32 * h as f32 / gh as f32) as usize;
            let y1 = (((gy + 1) as f32 * h as f32 / gh as f32) as usize).clamp(y0 + 1, h);
            let mut acc = 0.0;
            for y in y0..y1 {
                for x in x0..x1 {
                    acc += gray[y * w + x];
                }
            }
            out[gy * gw + gx] = acc / ((x1 - x0) * (y1 - y0)) as f32;
        }
    }
    (gw as u32, gh as u32, out)
}

pub fn build(kind: DepthMapKind, w: u32, h: u32, heights: &[f32], strength: f32, aspect: f32) -> Mesh {
    let (w, h) = (w.max(2) as usize, h.max(2) as usize);
    let at = |x: usize, y: usize| heights.get(y * w + x).copied().unwrap_or(0.0).clamp(0.0, 1.0);
    let strength = if strength.is_finite() { strength.clamp(-10.0, 10.0) } else { 0.3 };
    let aspect = if aspect.is_finite() && aspect > 0.0 { aspect.clamp(0.01, 100.0) } else { 1.0 };
    let mut m = Mesh::default();
    let grid = |m: &mut Mesh, f: &dyn Fn(f32, f32, f32) -> Vec3, flip: bool, slot: u16| {
        let base = m.positions.len() as u32;
        for y in 0..h {
            for x in 0..w {
                let (u, v) = (x as f32 / (w - 1) as f32, y as f32 / (h - 1) as f32);
                let p = f(u, v, at(x, y) * strength);
                m.vertex(p, Vec3::ZERO, [if flip { 1.0 - u } else { u }, v]);
            }
        }
        for y in 0..h - 1 {
            for x in 0..w - 1 {
                let a = base + (y * w + x) as u32;
                let (b, c, d) = (a + 1, a + w as u32, a + w as u32 + 1);
                if flip {
                    m.tri(a, b, c, slot);
                    m.tri(b, d, c, slot);
                } else {
                    m.tri(a, c, b, slot);
                    m.tri(b, c, d, slot);
                }
            }
        }
    };
    match kind {
        DepthMapKind::Plane => grid(&mut m, &|u, v, d| v3((u - 0.5) * aspect, 0.5 - v, d), false, 0),
        DepthMapKind::TwoSidedPlane => {
            grid(&mut m, &|u, v, d| v3((u - 0.5) * aspect, 0.5 - v, d * 0.5), false, 0);
            grid(&mut m, &|u, v, d| v3((u - 0.5) * aspect, 0.5 - v, -d * 0.5), true, 1);
        }
        DepthMapKind::Cylinder => grid(
            &mut m,
            &|u, v, d| {
                let th = u * TAU - PI;
                let r = 0.4 + d * 0.4;
                v3(r * th.sin(), 0.5 - v, r * th.cos())
            },
            false,
            0,
        ),
        DepthMapKind::Sphere => grid(
            &mut m,
            &|u, v, d| {
                let th = u * TAU - PI;
                let ph = v * PI;
                let r = 0.5 + d * 0.5;
                v3(r * ph.sin() * th.sin(), r * ph.cos(), r * ph.sin() * th.cos())
            },
            false,
            0,
        ),
    }
    m.compute_normals(80.0);
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_meshes_build() {
        let (w, h, g) = grid_from(&vec![0.5; 600 * 300], 600, 300);
        assert!(w <= 256 && h <= 256);
        for k in DepthMapKind::ALL {
            let m = build(k, w, h, &g, 0.3, 2.0);
            assert!(m.triangle_count() > 100, "{k:?}");
            // Plane faces the camera (+z).
            if k == DepthMapKind::Plane {
                assert!(m.face_normal(0).z > 0.5);
            }
        }
    }
}

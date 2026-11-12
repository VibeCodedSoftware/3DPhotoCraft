//! Small linear-algebra kit: 3-vectors and 4×4 matrices (column vectors, row-major storage),
//! enough for transforms, cameras and shading. `f32` throughout.

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub const fn v3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub const ZERO: Vec3 = v3(0.0, 0.0, 0.0);
    pub const ONE: Vec3 = v3(1.0, 1.0, 1.0);
    pub const X: Vec3 = v3(1.0, 0.0, 0.0);
    pub const Y: Vec3 = v3(0.0, 1.0, 0.0);
    pub const Z: Vec3 = v3(0.0, 0.0, 1.0);

    pub fn splat(v: f32) -> Self {
        v3(v, v, v)
    }
    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: Vec3) -> Vec3 {
        v3(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    /// Unit vector, or `fallback` for a (near-)zero vector.
    pub fn normalize_or(self, fallback: Vec3) -> Vec3 {
        let l = self.length();
        if l > 1e-12 && l.is_finite() { self / l } else { fallback }
    }
    pub fn normalize(self) -> Vec3 {
        self.normalize_or(Vec3::Z)
    }
    pub fn mul_elem(self, o: Vec3) -> Vec3 {
        v3(self.x * o.x, self.y * o.y, self.z * o.z)
    }
    pub fn min(self, o: Vec3) -> Vec3 {
        v3(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }
    pub fn max(self, o: Vec3) -> Vec3 {
        v3(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }
    pub fn lerp(self, o: Vec3, t: f32) -> Vec3 {
        self + (o - self) * t
    }
    pub fn max_elem(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }
    pub fn get(self, axis: usize) -> f32 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    /// Mirror `self` (pointing away from the surface) about the normal `n`.
    pub fn reflect(self, n: Vec3) -> Vec3 {
        self - n * (2.0 * self.dot(n))
    }
    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
    pub fn from_array(a: [f32; 3]) -> Vec3 {
        v3(a[0], a[1], a[2])
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}
impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f32) -> Vec3 {
        v3(self.x / s, self.y / s, self.z / s)
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        v3(-self.x, -self.y, -self.z)
    }
}

/// 4×4 matrix, row-major: `m[r][c]`. Points are column vectors (`M · p`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    pub m: [[f32; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4 { m: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] };

    pub fn translate(t: Vec3) -> Mat4 {
        let mut r = Mat4::IDENTITY;
        r.m[0][3] = t.x;
        r.m[1][3] = t.y;
        r.m[2][3] = t.z;
        r
    }
    pub fn scale(s: Vec3) -> Mat4 {
        let mut r = Mat4::IDENTITY;
        r.m[0][0] = s.x;
        r.m[1][1] = s.y;
        r.m[2][2] = s.z;
        r
    }
    pub fn rot_x(deg: f32) -> Mat4 {
        let (s, c) = deg.to_radians().sin_cos();
        Mat4 { m: [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]] }
    }
    pub fn rot_y(deg: f32) -> Mat4 {
        let (s, c) = deg.to_radians().sin_cos();
        Mat4 { m: [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]] }
    }
    pub fn rot_z(deg: f32) -> Mat4 {
        let (s, c) = deg.to_radians().sin_cos();
        Mat4 { m: [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] }
    }
    /// Rotation by Euler angles in degrees, applied X, then Y, then Z.
    pub fn euler(r: Vec3) -> Mat4 {
        Mat4::rot_z(r.z) * Mat4::rot_y(r.y) * Mat4::rot_x(r.x)
    }
    /// Translation · rotation · scale.
    pub fn trs(t: Vec3, r: Vec3, s: Vec3) -> Mat4 {
        Mat4::translate(t) * Mat4::euler(r) * Mat4::scale(s)
    }
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let m = &self.m;
        let x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
        let y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
        let z = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
        let w = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];
        if (w - 1.0).abs() > 1e-9 && w.abs() > 1e-12 { v3(x / w, y / w, z / w) } else { v3(x, y, z) }
    }
    pub fn transform_vector(&self, v: Vec3) -> Vec3 {
        let m = &self.m;
        v3(m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z, m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z, m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z)
    }
    pub fn transpose(&self) -> Mat4 {
        let mut r = Mat4::IDENTITY;
        for i in 0..4 {
            for j in 0..4 {
                r.m[i][j] = self.m[j][i];
            }
        }
        r
    }
    /// General inverse; the identity for a singular matrix.
    pub fn inverse(&self) -> Mat4 {
        let a = &self.m;
        let mut inv = [[0.0f32; 4]; 4];
        // Cofactor expansion (Gauss-Jordan would do as well; this is branch-free).
        let s0 = a[0][0] * a[1][1] - a[1][0] * a[0][1];
        let s1 = a[0][0] * a[1][2] - a[1][0] * a[0][2];
        let s2 = a[0][0] * a[1][3] - a[1][0] * a[0][3];
        let s3 = a[0][1] * a[1][2] - a[1][1] * a[0][2];
        let s4 = a[0][1] * a[1][3] - a[1][1] * a[0][3];
        let s5 = a[0][2] * a[1][3] - a[1][2] * a[0][3];
        let c5 = a[2][2] * a[3][3] - a[3][2] * a[2][3];
        let c4 = a[2][1] * a[3][3] - a[3][1] * a[2][3];
        let c3 = a[2][1] * a[3][2] - a[3][1] * a[2][2];
        let c2 = a[2][0] * a[3][3] - a[3][0] * a[2][3];
        let c1 = a[2][0] * a[3][2] - a[3][0] * a[2][2];
        let c0 = a[2][0] * a[3][1] - a[3][0] * a[2][1];
        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        if det.abs() < 1e-20 || !det.is_finite() {
            return Mat4::IDENTITY;
        }
        let d = 1.0 / det;
        inv[0][0] = (a[1][1] * c5 - a[1][2] * c4 + a[1][3] * c3) * d;
        inv[0][1] = (-a[0][1] * c5 + a[0][2] * c4 - a[0][3] * c3) * d;
        inv[0][2] = (a[3][1] * s5 - a[3][2] * s4 + a[3][3] * s3) * d;
        inv[0][3] = (-a[2][1] * s5 + a[2][2] * s4 - a[2][3] * s3) * d;
        inv[1][0] = (-a[1][0] * c5 + a[1][2] * c2 - a[1][3] * c1) * d;
        inv[1][1] = (a[0][0] * c5 - a[0][2] * c2 + a[0][3] * c1) * d;
        inv[1][2] = (-a[3][0] * s5 + a[3][2] * s2 - a[3][3] * s1) * d;
        inv[1][3] = (a[2][0] * s5 - a[2][2] * s2 + a[2][3] * s1) * d;
        inv[2][0] = (a[1][0] * c4 - a[1][1] * c2 + a[1][3] * c0) * d;
        inv[2][1] = (-a[0][0] * c4 + a[0][1] * c2 - a[0][3] * c0) * d;
        inv[2][2] = (a[3][0] * s4 - a[3][1] * s2 + a[3][3] * s0) * d;
        inv[2][3] = (-a[2][0] * s4 + a[2][1] * s2 - a[2][3] * s0) * d;
        inv[3][0] = (-a[1][0] * c3 + a[1][1] * c1 - a[1][2] * c0) * d;
        inv[3][1] = (a[0][0] * c3 - a[0][1] * c1 + a[0][2] * c0) * d;
        inv[3][2] = (-a[3][0] * s3 + a[3][1] * s1 - a[3][2] * s0) * d;
        inv[3][3] = (a[2][0] * s3 - a[2][1] * s1 + a[2][2] * s0) * d;
        Mat4 { m: inv }
    }
    /// Matrix that maps world points into a camera at `eye` looking at `target` (camera looks
    /// down its −Z axis, +Y up).
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let f = (target - eye).normalize_or(-Vec3::Z);
        let mut s = f.cross(up);
        if s.length() < 1e-6 {
            s = f.cross(Vec3::Z);
        }
        let s = s.normalize_or(Vec3::X);
        let u = s.cross(f);
        Mat4 { m: [[s.x, s.y, s.z, -s.dot(eye)], [u.x, u.y, u.z, -u.dot(eye)], [-f.x, -f.y, -f.z, f.dot(eye)], [0.0, 0.0, 0.0, 1.0]] }
    }
    /// Matrix for transforming normals (inverse transpose of the upper 3×3).
    pub fn normal_matrix(&self) -> Mat4 {
        let mut m = *self;
        m.m[0][3] = 0.0;
        m.m[1][3] = 0.0;
        m.m[2][3] = 0.0;
        m.inverse().transpose()
    }
}

impl Mul for Mat4 {
    type Output = Mat4;
    fn mul(self, o: Mat4) -> Mat4 {
        let mut r = [[0.0f32; 4]; 4];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (0..4).map(|k| self.m[i][k] * o.m[k][j]).sum();
            }
        }
        Mat4 { m: r }
    }
}

/// sRGB transfer function (encoded → linear).
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// sRGB transfer function (linear → encoded).
pub fn linear_to_srgb(c: f32) -> f32 {
    let c = c.max(0.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Clamp a user number into a range, mapping NaN to `lo`.
pub fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_nan() { lo } else { v.clamp(lo, hi) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_round_trips() {
        let m = Mat4::trs(v3(1.0, -2.0, 3.0), v3(10.0, 20.0, 30.0), v3(2.0, 1.0, 0.5));
        let p = v3(0.3, 0.7, -1.1);
        let q = m.inverse().transform_point(m.transform_point(p));
        assert!((q - p).length() < 1e-4);
    }

    #[test]
    fn look_at_puts_target_in_front() {
        let v = Mat4::look_at(v3(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let p = v.transform_point(Vec3::ZERO);
        assert!((p.z + 5.0).abs() < 1e-5);
    }
}

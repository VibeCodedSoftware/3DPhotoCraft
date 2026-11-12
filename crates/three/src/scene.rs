//! The 3D scene of a 3D layer: objects, materials, textures, lights, camera, environment and
//! render settings (Photoshop's 3D panel and its Properties).
//!
//! Conventions: right-handed, +Y up, the camera looks at `camera.target`. A layer turned into 3D
//! keeps its pixels where they were: the default camera maps the plane z = 0 onto the canvas,
//! with the canvas 2 units tall.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::extrude::{ExtrudeParams, Shape};
use crate::math::{Mat4, Vec3, clampf, v3};
use crate::mesh::{Aabb, Mesh, MeshPreset};

/// RGBA8 image (straight alpha, sRGB) used as a texture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    #[serde(with = "crate::b64")]
    pub rgba: Vec<u8>,
}

impl Image {
    /// A `w × h` image, or `None` when the data doesn't match.
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Option<Image> {
        let n = (width as usize).checked_mul(height as usize)?.checked_mul(4)?;
        (width > 0 && height > 0 && rgba.len() == n).then_some(Image { width, height, rgba })
    }
    pub fn solid(c: [u8; 4]) -> Image {
        Image { width: 1, height: 1, rgba: c.to_vec() }
    }
    /// Bilinear, repeating sample at `uv`, returned as linear RGB + alpha.
    pub fn sample(&self, uv: [f32; 2]) -> [f32; 4] {
        let (w, h) = (self.width as usize, self.height as usize);
        if w == 0 || h == 0 || self.rgba.len() < w * h * 4 {
            return [1.0, 1.0, 1.0, 1.0];
        }
        let fx = (uv[0].rem_euclid(1.0)) * w as f32 - 0.5;
        let fy = (uv[1].rem_euclid(1.0)) * h as f32 - 0.5;
        if !fx.is_finite() || !fy.is_finite() {
            return [1.0, 1.0, 1.0, 1.0];
        }
        let x0 = fx.floor();
        let y0 = fy.floor();
        let (tx, ty) = (fx - x0, fy - y0);
        let wrap = |v: f32, n: usize| -> usize { (v as i64).rem_euclid(n as i64) as usize };
        let px = |x: usize, y: usize| -> [f32; 4] {
            let i = (y * w + x) * 4;
            let lin = &crate::render::SRGB_LUT;
            [lin[self.rgba[i] as usize], lin[self.rgba[i + 1] as usize], lin[self.rgba[i + 2] as usize], self.rgba[i + 3] as f32 / 255.0]
        };
        let (xa, xb, ya, yb) = (wrap(x0, w), wrap(x0 + 1.0, w), wrap(y0, h), wrap(y0 + 1.0, h));
        let (a, b, c, d) = (px(xa, ya), px(xb, ya), px(xa, yb), px(xb, yb));
        let mut o = [0.0; 4];
        for k in 0..4 {
            o[k] = (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty;
        }
        o
    }
    /// Gray value (0–1) at `uv`, for bump and depth maps.
    pub fn luminance(&self, uv: [f32; 2]) -> f32 {
        let c = self.sample(uv);
        (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) * c[3]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Texture {
    pub id: u32,
    pub name: String,
    pub image: Arc<Image>,
}

/// Linear-free colour: sRGB components 0–1 as the user picks them.
pub type Rgb = [f32; 3];

/// Photoshop's material model: diffuse, specular, illumination, ambient, shine, reflection,
/// roughness, bump, opacity, refraction, normal and environment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Material {
    pub name: String,
    pub diffuse: Rgb,
    pub diffuse_texture: Option<u32>,
    pub specular: Rgb,
    /// Self-illumination (emission).
    pub illumination: Rgb,
    pub ambient: Rgb,
    /// Glossiness, 0–1 (higher = tighter highlight).
    pub shine: f32,
    /// Mirror reflection, 0–1.
    pub reflection: f32,
    /// Blur of reflections, 0–1 (ray tracer).
    pub roughness: f32,
    /// Bump strength, 0–10, with its height texture.
    pub bump: f32,
    pub bump_texture: Option<u32>,
    pub opacity: f32,
    pub opacity_texture: Option<u32>,
    /// Index of refraction for transparent materials.
    pub refraction: f32,
    pub normal_texture: Option<u32>,
    /// Texture repeat and offset (Edit UV Properties).
    pub uv_scale: [f32; 2],
    pub uv_offset: [f32; 2],
    pub double_sided: bool,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            name: "Default Material".into(),
            diffuse: [0.8, 0.8, 0.8],
            diffuse_texture: None,
            specular: [0.25, 0.25, 0.25],
            illumination: [0.0, 0.0, 0.0],
            ambient: [0.0, 0.0, 0.0],
            shine: 0.3,
            reflection: 0.0,
            roughness: 0.0,
            bump: 1.0,
            bump_texture: None,
            opacity: 1.0,
            opacity_texture: None,
            refraction: 1.0,
            normal_texture: None,
            uv_scale: [1.0, 1.0],
            uv_offset: [0.0, 0.0],
            double_sided: true,
        }
    }
}

impl Material {
    pub fn named(name: &str) -> Material {
        Material { name: name.into(), ..Default::default() }
    }
    pub fn sanitize(&mut self) {
        let c3 = |c: &mut Rgb| c.iter_mut().for_each(|v| *v = clampf(*v, 0.0, 1.0));
        c3(&mut self.diffuse);
        c3(&mut self.specular);
        c3(&mut self.ambient);
        self.illumination.iter_mut().for_each(|v| *v = clampf(*v, 0.0, 10.0));
        self.shine = clampf(self.shine, 0.0, 1.0);
        self.reflection = clampf(self.reflection, 0.0, 1.0);
        self.roughness = clampf(self.roughness, 0.0, 1.0);
        self.bump = clampf(self.bump, 0.0, 10.0);
        self.opacity = clampf(self.opacity, 0.0, 1.0);
        self.refraction = clampf(self.refraction, 1.0, 3.0);
        for v in self.uv_scale.iter_mut() {
            *v = if v.is_finite() && v.abs() > 1e-4 { v.clamp(-1000.0, 1000.0) } else { 1.0 };
        }
        for v in self.uv_offset.iter_mut() {
            *v = clampf(*v, -1000.0, 1000.0);
        }
    }
}

/// Named material presets (Photoshop's Material Picker library).
pub const MATERIAL_PRESETS: &[&str] = &[
    "default", "plastic", "glossyPlastic", "matte", "metalGold", "metalChrome", "metalCopper", "glass", "rubber", "wood", "stone", "neon",
];

pub fn material_preset(id: &str) -> Option<Material> {
    let m = |d: Rgb, s: Rgb, shine: f32, refl: f32| Material { name: id.into(), diffuse: d, specular: s, shine, reflection: refl, ..Default::default() };
    Some(match id {
        "default" => Material { name: id.into(), ..Default::default() },
        "plastic" => m([0.75, 0.12, 0.1], [0.5, 0.5, 0.5], 0.55, 0.05),
        "glossyPlastic" => m([0.1, 0.35, 0.85], [0.9, 0.9, 0.9], 0.9, 0.15),
        "matte" => m([0.7, 0.7, 0.68], [0.02, 0.02, 0.02], 0.05, 0.0),
        "metalGold" => m([0.83, 0.62, 0.22], [1.0, 0.85, 0.5], 0.8, 0.55),
        "metalChrome" => m([0.55, 0.56, 0.58], [1.0, 1.0, 1.0], 0.95, 0.85),
        "metalCopper" => m([0.72, 0.4, 0.24], [0.95, 0.65, 0.5], 0.75, 0.45),
        "glass" => Material { name: id.into(), diffuse: [0.9, 0.95, 1.0], specular: [1.0, 1.0, 1.0], shine: 0.95, reflection: 0.15, opacity: 0.2, refraction: 1.5, ..Default::default() },
        "rubber" => m([0.08, 0.08, 0.08], [0.1, 0.1, 0.1], 0.15, 0.0),
        "wood" => m([0.55, 0.36, 0.2], [0.15, 0.12, 0.1], 0.25, 0.0),
        "stone" => m([0.5, 0.48, 0.45], [0.05, 0.05, 0.05], 0.1, 0.0),
        "neon" => Material { name: id.into(), diffuse: [0.2, 1.0, 0.9], illumination: [0.2, 1.0, 0.9], ..Default::default() },
        _ => return None,
    })
}

/// Where an object's geometry comes from; the mesh is rebuilt from it, so parametric objects
/// stay editable (extrusion depth, bevel, preset).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Geometry {
    Preset { preset: MeshPreset, aspect: f32 },
    Extrusion { shapes: Vec<Shape>, params: ExtrudeParams, uv_box: [f32; 4] },
    DepthMap { map: DepthMapKind, width: u32, height: u32, heights: Vec<f32>, strength: f32, aspect: f32 },
    Mesh { mesh: Arc<Mesh> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DepthMapKind {
    Plane,
    TwoSidedPlane,
    Cylinder,
    Sphere,
}

impl DepthMapKind {
    pub const ALL: [DepthMapKind; 4] = [DepthMapKind::Plane, DepthMapKind::TwoSidedPlane, DepthMapKind::Cylinder, DepthMapKind::Sphere];
    pub fn id(self) -> &'static str {
        match self {
            DepthMapKind::Plane => "plane",
            DepthMapKind::TwoSidedPlane => "twoSidedPlane",
            DepthMapKind::Cylinder => "cylinder",
            DepthMapKind::Sphere => "sphere",
        }
    }
    pub fn from_id(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.id().eq_ignore_ascii_case(s))
    }
}

impl Geometry {
    pub fn build(&self) -> Mesh {
        match self {
            Geometry::Preset { preset, aspect } => preset.build(*aspect),
            Geometry::Extrusion { shapes, params, uv_box } => {
                let [x0, y0, x1, y1] = *uv_box;
                let (dx, dy) = ((x1 - x0).abs().max(1e-6), (y1 - y0).abs().max(1e-6));
                crate::extrude::extrude(shapes, params, &|p| [(p[0] - x0) / dx, (y1 - p[1]) / dy])
            }
            Geometry::DepthMap { map, width, height, heights, strength, aspect } => crate::depth::build(*map, *width, *height, heights, *strength, *aspect),
            Geometry::Mesh { mesh } => (**mesh).clone(),
        }
    }
    pub fn kind_name(&self) -> &'static str {
        match self {
            Geometry::Preset { preset, .. } => preset.label(),
            Geometry::Extrusion { .. } => "Extrusion",
            Geometry::DepthMap { .. } => "Depth Map",
            Geometry::Mesh { .. } => "Mesh",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Transform {
    pub position: Vec3,
    /// Euler angles, degrees (X, then Y, then Z).
    pub rotation: Vec3,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Transform { position: Vec3::ZERO, rotation: Vec3::ZERO, scale: Vec3::ONE }
    }
}

impl Transform {
    pub fn matrix(&self) -> Mat4 {
        Mat4::trs(self.position, self.rotation, self.scale)
    }
    pub fn sanitize(&mut self) {
        let f = |v: &mut Vec3, lo: f32, hi: f32| {
            v.x = clampf(v.x, lo, hi);
            v.y = clampf(v.y, lo, hi);
            v.z = clampf(v.z, lo, hi);
        };
        f(&mut self.position, -1e5, 1e5);
        f(&mut self.rotation, -36000.0, 36000.0);
        f(&mut self.scale, -1e4, 1e4);
        for s in [&mut self.scale.x, &mut self.scale.y, &mut self.scale.z] {
            if s.abs() < 1e-4 {
                *s = 1e-4_f32.copysign(*s);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Object {
    pub id: u32,
    pub name: String,
    #[serde(default = "yes")]
    pub visible: bool,
    pub geometry: Geometry,
    #[serde(default)]
    pub transform: Transform,
    /// One material per mesh slot.
    pub materials: Vec<Material>,
    #[serde(default = "yes")]
    pub cast_shadows: bool,
    #[serde(default = "yes")]
    pub catch_shadows: bool,
    /// Rendered invisible but still casting and catching shadows (Photoshop's Invisible).
    #[serde(default)]
    pub invisible: bool,
    /// Built from `geometry`; not saved.
    #[serde(skip)]
    pub mesh: MeshCache,
}

fn yes() -> bool {
    true
}

/// The object's built mesh, compared by identity so scene comparisons stay cheap.
#[derive(Clone, Debug, Default)]
pub struct MeshCache(pub Option<Arc<Mesh>>);

impl PartialEq for MeshCache {
    fn eq(&self, o: &Self) -> bool {
        match (&self.0, &o.0) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b) || a == b,
            (None, None) => true,
            _ => false,
        }
    }
}

impl Object {
    pub fn new(id: u32, name: impl Into<String>, geometry: Geometry, materials: Vec<Material>) -> Object {
        let mut o = Object {
            id,
            name: name.into(),
            visible: true,
            geometry,
            transform: Transform::default(),
            materials,
            cast_shadows: true,
            catch_shadows: true,
            invisible: false,
            mesh: MeshCache::default(),
        };
        o.rebuild();
        o
    }
    /// Rebuild the mesh from the geometry, and make sure every slot has a material.
    pub fn rebuild(&mut self) {
        let mesh = self.geometry.build();
        let slots = mesh.slot_count();
        while self.materials.len() < slots {
            let n = self.materials.len() + 1;
            self.materials.push(Material::named(&format!("Material {n}")));
        }
        self.mesh = MeshCache(Some(Arc::new(mesh)));
    }
    pub fn mesh(&self) -> Arc<Mesh> {
        match &self.mesh.0 {
            Some(m) => m.clone(),
            None => Arc::new(self.geometry.build()),
        }
    }
    pub fn world_bounds(&self) -> Aabb {
        self.mesh().bounds_with(&self.transform.matrix())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LightKind {
    /// Directional light from infinitely far away.
    Infinite,
    Spot,
    Point,
}

impl LightKind {
    pub fn id(self) -> &'static str {
        match self {
            LightKind::Infinite => "infinite",
            LightKind::Spot => "spot",
            LightKind::Point => "point",
        }
    }
    pub fn from_id(s: &str) -> Option<Self> {
        [LightKind::Infinite, LightKind::Spot, LightKind::Point].into_iter().find(|k| k.id().eq_ignore_ascii_case(s))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Light {
    pub id: u32,
    pub name: String,
    pub kind: LightKind,
    pub visible: bool,
    pub color: Rgb,
    /// Intensity multiplier, 0–10.
    pub intensity: f32,
    pub shadow: bool,
    /// Shadow softness, 0–1.
    pub softness: f32,
    /// Position (point and spot).
    pub position: Vec3,
    /// Direction the light shines, as yaw/pitch in degrees (infinite and spot).
    pub yaw: f32,
    pub pitch: f32,
    /// Spot hotspot and cone, full angles in degrees.
    pub hotspot: f32,
    pub cone: f32,
    /// Distance falloff for point and spot lights.
    pub falloff: bool,
    pub inner_radius: f32,
    pub outer_radius: f32,
}

impl Default for Light {
    fn default() -> Self {
        Light {
            id: 0,
            name: "Infinite Light 1".into(),
            kind: LightKind::Infinite,
            visible: true,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            shadow: true,
            softness: 0.2,
            position: v3(2.0, 3.0, 4.0),
            yaw: -30.0,
            pitch: -45.0,
            hotspot: 30.0,
            cone: 45.0,
            falloff: false,
            inner_radius: 2.0,
            outer_radius: 10.0,
        }
    }
}

impl Light {
    /// Unit vector the light travels along.
    pub fn direction(&self) -> Vec3 {
        let (sy, cy) = self.yaw.to_radians().sin_cos();
        let (sp, cp) = self.pitch.to_radians().sin_cos();
        // yaw 0 / pitch 0 shines along −Z (into the scene, away from the default camera).
        v3(sy * cp, sp, -cy * cp).normalize()
    }
    /// Point the light at `target` from its position.
    pub fn aim_at(&mut self, target: Vec3) {
        let d = (target - self.position).normalize_or(-Vec3::Z);
        self.pitch = d.y.clamp(-1.0, 1.0).asin().to_degrees();
        self.yaw = d.x.atan2(-d.z).to_degrees();
    }
    pub fn sanitize(&mut self) {
        self.color.iter_mut().for_each(|v| *v = clampf(*v, 0.0, 1.0));
        self.intensity = clampf(self.intensity, 0.0, 10.0);
        self.softness = clampf(self.softness, 0.0, 1.0);
        self.hotspot = clampf(self.hotspot, 0.0, 179.0);
        self.cone = clampf(self.cone, 0.1, 179.0).max(self.hotspot);
        self.yaw = clampf(self.yaw, -36000.0, 36000.0);
        self.pitch = clampf(self.pitch, -90.0, 90.0);
        self.inner_radius = clampf(self.inner_radius, 0.0, 1e4);
        self.outer_radius = clampf(self.outer_radius, self.inner_radius + 1e-3, 1e5);
        let p = &mut self.position;
        p.x = clampf(p.x, -1e5, 1e5);
        p.y = clampf(p.y, -1e5, 1e5);
        p.z = clampf(p.z, -1e5, 1e5);
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    /// Orbit angles, degrees.
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    /// Vertical field of view, degrees (perspective).
    pub fov: f32,
    pub orthographic: bool,
    /// Visible height in world units (orthographic).
    pub ortho_height: f32,
    /// Depth of field: blur amount 0–1 (0 = off) and focus distance from the camera.
    pub dof_blur: f32,
    pub dof_distance: f32,
}

/// Field of view of the default camera.
pub const DEFAULT_FOV: f32 = 30.0;

impl Default for Camera {
    fn default() -> Self {
        let d = 1.0 / (DEFAULT_FOV.to_radians() * 0.5).tan();
        Camera { target: Vec3::ZERO, distance: d, yaw: 0.0, pitch: 0.0, roll: 0.0, fov: DEFAULT_FOV, orthographic: false, ortho_height: 2.0, dof_blur: 0.0, dof_distance: d }
    }
}

impl Camera {
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.to_radians().sin_cos();
        let (sp, cp) = self.pitch.to_radians().sin_cos();
        self.target + v3(sy * cp, sp, cy * cp) * self.distance
    }
    pub fn up(&self) -> Vec3 {
        let (sy, cy) = self.yaw.to_radians().sin_cos();
        let (sp, cp) = self.pitch.to_radians().sin_cos();
        // Orthogonal to the view direction, rolled.
        let up = v3(-sy * sp, cp, -cy * sp);
        let fwd = (self.target - self.eye()).normalize();
        let right = fwd.cross(up).normalize();
        let (sr, cr) = self.roll.to_radians().sin_cos();
        (up * cr + right * sr).normalize()
    }
    pub fn view(&self) -> Mat4 {
        Mat4::look_at(self.eye(), self.target, self.up())
    }
    pub fn sanitize(&mut self) {
        self.distance = clampf(self.distance, 0.01, 1e5);
        self.yaw = clampf(self.yaw, -36000.0, 36000.0);
        self.pitch = clampf(self.pitch, -89.9, 89.9);
        self.roll = clampf(self.roll, -36000.0, 36000.0);
        self.fov = clampf(self.fov, 1.0, 170.0);
        self.ortho_height = clampf(self.ortho_height, 1e-3, 1e5);
        self.dof_blur = clampf(self.dof_blur, 0.0, 1.0);
        self.dof_distance = clampf(self.dof_distance, 0.01, 1e5);
        let t = &mut self.target;
        t.x = clampf(t.x, -1e5, 1e5);
        t.y = clampf(t.y, -1e5, 1e5);
        t.z = clampf(t.z, -1e5, 1e5);
    }
    /// Standard views (View › 3D views; the camera keeps its target and distance).
    pub fn set_view(&mut self, name: &str) -> bool {
        let (yaw, pitch) = match name {
            "front" | "default" => (0.0, 0.0),
            "back" => (180.0, 0.0),
            "left" => (-90.0, 0.0),
            "right" => (90.0, 0.0),
            "top" => (0.0, 89.9),
            "bottom" => (0.0, -89.9),
            _ => return false,
        };
        self.yaw = yaw;
        self.pitch = pitch;
        self.roll = 0.0;
        true
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Background {
    /// Transparent: the layer shows what's beneath it.
    #[default]
    Transparent,
    Color,
    Environment,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Environment {
    /// Global ambient colour.
    pub ambient: Rgb,
    /// Image-based light: sky and ground colours, or an equirectangular texture.
    pub sky: Rgb,
    pub ground: Rgb,
    pub ibl_texture: Option<u32>,
    pub ibl_intensity: f32,
    pub ibl_rotation: f32,
    /// Ground plane height (world y) and whether the editor shows its grid.
    pub ground_y: f32,
    pub show_ground: bool,
    /// Ground plane shadow catcher opacity, 0–1.
    pub ground_shadow_opacity: f32,
    /// Ground plane reflections opacity, 0–1 (ray tracer).
    pub ground_reflection_opacity: f32,
    pub background: Background,
    pub background_color: Rgb,
}

impl Default for Environment {
    fn default() -> Self {
        Environment {
            ambient: [0.08, 0.08, 0.09],
            sky: [0.62, 0.68, 0.76],
            ground: [0.3, 0.28, 0.26],
            ibl_texture: None,
            ibl_intensity: 0.6,
            ibl_rotation: 0.0,
            ground_y: -1.0,
            show_ground: true,
            ground_shadow_opacity: 0.6,
            ground_reflection_opacity: 0.0,
            background: Background::Transparent,
            background_color: [0.2, 0.2, 0.2],
        }
    }
}

/// How the scene is drawn (Photoshop's Scene › Presets / Surface Style).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Style {
    #[default]
    Solid,
    Unlit,
    Wireframe,
    ShadedWireframe,
    Points,
    Normals,
    DepthMap,
    LineIllustration,
}

impl Style {
    pub const ALL: [Style; 8] = [Style::Solid, Style::Unlit, Style::Wireframe, Style::ShadedWireframe, Style::Points, Style::Normals, Style::DepthMap, Style::LineIllustration];
    pub fn id(self) -> &'static str {
        match self {
            Style::Solid => "solid",
            Style::Unlit => "unlitTexture",
            Style::Wireframe => "wireframe",
            Style::ShadedWireframe => "shadedWireframe",
            Style::Points => "points",
            Style::Normals => "normals",
            Style::DepthMap => "depthMap",
            Style::LineIllustration => "lineIllustration",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Style::Solid => "Default",
            Style::Unlit => "Unlit Texture",
            Style::Wireframe => "Wireframe",
            Style::ShadedWireframe => "Shaded Wireframe",
            Style::Points => "Points",
            Style::Normals => "Normals",
            Style::DepthMap => "Depth Map",
            Style::LineIllustration => "Line Illustration",
        }
    }
    pub fn from_id(s: &str) -> Option<Style> {
        Style::ALL.into_iter().find(|k| k.id().eq_ignore_ascii_case(s))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CrossAxis {
    X,
    Y,
    Z,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RenderSettings {
    pub style: Style,
    /// Rays per pixel for Render 3D Layer (ray tracer), 1–256.
    pub samples: u32,
    /// Reflection / refraction bounces, 0–8.
    pub bounces: u32,
    /// Interactive (rasterized) preview antialiasing: supersampling factor 1–3.
    pub preview_aa: u32,
    pub line_color: Rgb,
    pub exposure: f32,
    /// Cross section (Apply Cross Section to Scene).
    pub cross_section: bool,
    pub cross_axis: CrossAxis,
    pub cross_offset: f32,
    pub cross_flip: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        RenderSettings {
            style: Style::Solid,
            samples: 16,
            bounces: 3,
            preview_aa: 1,
            line_color: [0.0, 0.0, 0.0],
            exposure: 0.0,
            cross_section: false,
            cross_axis: CrossAxis::X,
            cross_offset: 0.0,
            cross_flip: false,
        }
    }
}

/// How the layer's pixels were last produced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RenderState {
    /// Interactive rasterized preview.
    #[default]
    Preview,
    /// Ray traced (3D › Render 3D Layer); any edit returns to Preview.
    RayTraced,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub lights: Vec<Light>,
    pub textures: Vec<Texture>,
    pub camera: Camera,
    /// Saved camera views (Properties › Camera › View › Save).
    pub saved_views: Vec<(String, Camera)>,
    pub environment: Environment,
    pub render: RenderSettings,
    pub state: RenderState,
    next_id: u32,
}

impl Default for Scene {
    fn default() -> Self {
        let mut s = Scene {
            objects: Vec::new(),
            lights: Vec::new(),
            textures: Vec::new(),
            camera: Camera::default(),
            saved_views: Vec::new(),
            environment: Environment::default(),
            render: RenderSettings::default(),
            state: RenderState::Preview,
            next_id: 1,
        };
        let id = s.fresh_id();
        s.lights.push(Light { id, ..Default::default() });
        s
    }
}

impl Scene {
    pub fn fresh_id(&mut self) -> u32 {
        let used = self.objects.iter().map(|o| o.id).chain(self.lights.iter().map(|l| l.id)).chain(self.textures.iter().map(|t| t.id)).max().unwrap_or(0);
        self.next_id = self.next_id.max(used + 1);
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    /// Rebuild meshes after loading (they aren't saved) and clamp every value into range.
    pub fn rebuild(&mut self) {
        for o in &mut self.objects {
            o.transform.sanitize();
            for m in &mut o.materials {
                m.sanitize();
            }
            o.rebuild();
        }
        for l in &mut self.lights {
            l.sanitize();
        }
        self.camera.sanitize();
        self.render.samples = self.render.samples.clamp(1, 256);
        self.render.bounces = self.render.bounces.min(8);
        self.render.preview_aa = self.render.preview_aa.clamp(1, 3);
        self.render.exposure = clampf(self.render.exposure, -10.0, 10.0);
        self.environment.ibl_intensity = clampf(self.environment.ibl_intensity, 0.0, 10.0);
        self.environment.ground_shadow_opacity = clampf(self.environment.ground_shadow_opacity, 0.0, 1.0);
        self.environment.ground_reflection_opacity = clampf(self.environment.ground_reflection_opacity, 0.0, 1.0);
        self.environment.ground_y = clampf(self.environment.ground_y, -1e5, 1e5);
        let _ = self.fresh_id();
    }

    pub fn texture(&self, id: Option<u32>) -> Option<&Image> {
        let id = id?;
        self.textures.iter().find(|t| t.id == id).map(|t| &*t.image)
    }

    pub fn add_texture(&mut self, name: &str, image: Image) -> u32 {
        let id = self.fresh_id();
        self.textures.push(Texture { id, name: name.into(), image: Arc::new(image) });
        id
    }

    pub fn object(&self, id: u32) -> Option<&Object> {
        self.objects.iter().find(|o| o.id == id)
    }
    pub fn object_mut(&mut self, id: u32) -> Option<&mut Object> {
        self.objects.iter_mut().find(|o| o.id == id)
    }
    pub fn light_mut(&mut self, id: u32) -> Option<&mut Light> {
        self.lights.iter_mut().find(|l| l.id == id)
    }

    /// Add an object with a fresh id, returning the id.
    pub fn add_object(&mut self, name: &str, geometry: Geometry, materials: Vec<Material>) -> u32 {
        let id = self.fresh_id();
        self.objects.push(Object::new(id, name, geometry, materials));
        id
    }

    /// Bounds of every visible object.
    pub fn bounds(&self) -> Aabb {
        self.objects.iter().filter(|o| o.visible).fold(Aabb::EMPTY, |b, o| b.union(&o.world_bounds()))
    }

    /// Move an object down (or up) so it rests on the ground plane (3D › Move Object to Ground Plane).
    pub fn drop_to_ground(&mut self, id: u32) -> bool {
        let gy = self.environment.ground_y;
        let Some(o) = self.object_mut(id) else { return false };
        let b = o.world_bounds();
        if b.is_empty() {
            return false;
        }
        o.transform.position.y += gy - b.min.y;
        true
    }

    /// Ground plane level just under every object (used when a scene is created).
    pub fn fit_ground(&mut self) {
        let b = self.bounds();
        if !b.is_empty() {
            self.environment.ground_y = b.min.y;
        }
    }

    /// Frame the camera on `bounds` (View › Fit / Frame Selected), keeping the view direction.
    pub fn frame(&mut self, bounds: Aabb) {
        if bounds.is_empty() {
            return;
        }
        let r = (bounds.size().length() * 0.5).max(1e-3);
        self.camera.target = bounds.center();
        let half = (self.camera.fov.to_radians() * 0.5).tan().max(1e-3);
        self.camera.distance = (r / half * 1.15).max(r * 1.2);
        self.camera.ortho_height = r * 2.3;
    }

    /// Merge another scene's objects, lights and textures into this one (3D › Merge 3D Layers).
    pub fn merge(&mut self, other: &Scene) {
        let mut remap = std::collections::HashMap::new();
        for t in &other.textures {
            let id = self.fresh_id();
            remap.insert(t.id, id);
            self.textures.push(Texture { id, ..t.clone() });
        }
        let fix = |t: &mut Option<u32>| {
            if let Some(v) = t {
                *t = remap.get(v).copied();
            }
        };
        for o in &other.objects {
            let id = self.fresh_id();
            let mut o = o.clone();
            o.id = id;
            for m in &mut o.materials {
                fix(&mut m.diffuse_texture);
                fix(&mut m.bump_texture);
                fix(&mut m.opacity_texture);
                fix(&mut m.normal_texture);
            }
            self.objects.push(o);
        }
        for l in &other.lights {
            let id = self.fresh_id();
            self.lights.push(Light { id, ..l.clone() });
        }
        self.state = RenderState::Preview;
    }

    /// Drop textures no material uses.
    pub fn prune_textures(&mut self) {
        let used: std::collections::HashSet<u32> = self
            .objects
            .iter()
            .flat_map(|o| o.materials.iter())
            .flat_map(|m| [m.diffuse_texture, m.bump_texture, m.opacity_texture, m.normal_texture])
            .flatten()
            .chain(self.environment.ibl_texture)
            .collect();
        self.textures.retain(|t| used.contains(&t.id));
    }

    pub fn triangle_count(&self) -> usize {
        self.objects.iter().map(|o| o.mesh().triangle_count()).sum()
    }
}

/// Light presets (Properties › Lights › Preset).
pub const LIGHT_PRESETS: &[&str] = &["default", "studio", "softbox", "dramatic", "productPhotography", "threePoint", "sunset", "neutral", "night"];

/// Replace the scene's lights with a preset (keeps the environment's colours in step).
pub fn apply_light_preset(scene: &mut Scene, id: &str) -> bool {
    let inf = |name: &str, yaw: f32, pitch: f32, c: Rgb, i: f32, shadow: bool, soft: f32| Light { name: name.into(), kind: LightKind::Infinite, yaw, pitch, color: c, intensity: i, shadow, softness: soft, ..Default::default() };
    let spot = |name: &str, pos: Vec3, c: Rgb, i: f32, cone: f32| {
        let mut l = Light { name: name.into(), kind: LightKind::Spot, position: pos, color: c, intensity: i, hotspot: cone * 0.6, cone, shadow: true, softness: 0.3, ..Default::default() };
        l.aim_at(Vec3::ZERO);
        l
    };
    let (lights, ambient, sky, ground): (Vec<Light>, Rgb, Rgb, Rgb) = match id {
        "default" => (vec![inf("Infinite Light 1", -30.0, -45.0, [1.0, 1.0, 1.0], 1.0, true, 0.2)], [0.08, 0.08, 0.09], [0.62, 0.68, 0.76], [0.3, 0.28, 0.26]),
        "studio" => (
            vec![inf("Key Light", -40.0, -35.0, [1.0, 0.98, 0.95], 1.0, true, 0.3), inf("Fill Light", 50.0, -15.0, [0.85, 0.9, 1.0], 0.4, false, 0.0), inf("Top Light", 0.0, -85.0, [1.0, 1.0, 1.0], 0.3, false, 0.0)],
            [0.1, 0.1, 0.1],
            [0.7, 0.7, 0.72],
            [0.4, 0.4, 0.4],
        ),
        "softbox" => (vec![inf("Softbox", -20.0, -50.0, [1.0, 1.0, 1.0], 1.1, true, 0.9), inf("Bounce", 160.0, -10.0, [1.0, 1.0, 1.0], 0.25, false, 0.0)], [0.15, 0.15, 0.15], [0.8, 0.8, 0.8], [0.5, 0.5, 0.5]),
        "dramatic" => (vec![spot("Spot Light", v3(-3.0, 4.0, 2.5), [1.0, 0.95, 0.85], 2.2, 35.0), inf("Rim Light", 160.0, -20.0, [0.6, 0.7, 1.0], 0.5, false, 0.0)], [0.02, 0.02, 0.03], [0.1, 0.1, 0.12], [0.03, 0.03, 0.03]),
        "productPhotography" => (
            vec![inf("Key Light", -35.0, -40.0, [1.0, 1.0, 1.0], 0.9, true, 0.7), inf("Fill Light", 45.0, -25.0, [1.0, 1.0, 1.0], 0.45, false, 0.0), inf("Back Light", 180.0, -30.0, [1.0, 1.0, 1.0], 0.6, false, 0.0)],
            [0.18, 0.18, 0.18],
            [0.95, 0.95, 0.95],
            [0.85, 0.85, 0.85],
        ),
        "threePoint" => (
            vec![inf("Key Light", -45.0, -35.0, [1.0, 0.97, 0.92], 1.0, true, 0.25), inf("Fill Light", 45.0, -20.0, [0.8, 0.85, 1.0], 0.45, false, 0.0), inf("Rim Light", 170.0, -35.0, [1.0, 1.0, 1.0], 0.8, false, 0.0)],
            [0.06, 0.06, 0.07],
            [0.5, 0.55, 0.6],
            [0.25, 0.24, 0.22],
        ),
        "sunset" => (vec![inf("Sun", -70.0, -12.0, [1.0, 0.62, 0.32], 1.3, true, 0.15)], [0.12, 0.06, 0.08], [0.95, 0.55, 0.4], [0.25, 0.15, 0.15]),
        "neutral" => (vec![inf("Infinite Light 1", -20.0, -60.0, [1.0, 1.0, 1.0], 0.6, true, 0.6)], [0.25, 0.25, 0.25], [0.7, 0.7, 0.7], [0.7, 0.7, 0.7]),
        "night" => (vec![spot("Moon", v3(2.0, 5.0, 3.0), [0.6, 0.7, 1.0], 1.0, 60.0)], [0.02, 0.02, 0.05], [0.05, 0.07, 0.15], [0.02, 0.02, 0.03]),
        _ => return false,
    };
    scene.lights.clear();
    for l in lights {
        let id = scene.fresh_id();
        scene.lights.push(Light { id, ..l });
    }
    scene.environment.ambient = ambient;
    scene.environment.sky = sky;
    scene.environment.ground = ground;
    scene.state = RenderState::Preview;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_round_trips_through_json() {
        let mut s = Scene::default();
        let tex = s.add_texture("t", Image::solid([255, 0, 0, 255]));
        let mut m = Material::default();
        m.diffuse_texture = Some(tex);
        s.add_object("Cube", Geometry::Preset { preset: MeshPreset::Cube, aspect: 1.0 }, vec![m]);
        let json = serde_json::to_string(&s).unwrap();
        let mut back: Scene = serde_json::from_str(&json).unwrap();
        back.rebuild();
        assert_eq!(back.objects.len(), 1);
        assert_eq!(back.objects[0].mesh().triangle_count(), s.objects[0].mesh().triangle_count());
        assert_eq!(back.textures[0].image.rgba, vec![255, 0, 0, 255]);
        assert_eq!(back.objects[0].materials.len(), 6, "a cube has six slots");
    }

    #[test]
    fn default_camera_maps_unit_plane_to_canvas() {
        let c = Camera::default();
        let half = (c.fov.to_radians() * 0.5).tan() * c.distance;
        assert!((half - 1.0).abs() < 1e-4);
    }

    #[test]
    fn every_light_preset_applies() {
        for p in LIGHT_PRESETS {
            let mut s = Scene::default();
            assert!(apply_light_preset(&mut s, p), "{p}");
            assert!(!s.lights.is_empty());
        }
        assert!(!apply_light_preset(&mut Scene::default(), "nope"));
    }
}

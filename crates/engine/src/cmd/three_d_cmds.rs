//! The 3D menu: Photoshop's 3D workspace (removed from Photoshop in 2023), rebuilt on
//! `photocraft-three`.
//!
//! A 3D layer is a pixel layer with a scene ([`photocraft_doc::Layer::scene3d`]). Every edit to
//! the scene re-renders the layer's pixels with the interactive rasterizer, so the canvas, the
//! compositor, PSD export and the pixel tools always see the current render; 3D › Render 3D Layer
//! replaces them with a ray-traced render (a background job). Rasterize drops the scene.
//!
//! ```text
//! layer / selection / path ──trace──► outlines ──extrude──┐
//! layer pixels ──────────────────────────────► postcard / mesh preset / depth map ─┤──► Scene ──render──► layer pixels
//! OBJ / STL / glTF file ──import──────────────────────────┘
//! ```

use std::sync::Arc;

use photocraft_doc::{Document, Layer, LayerContent, LayerId};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use photocraft_three::extrude::{self, Contour, ExtrudeParams, Shape};
use photocraft_three::io::{ImageRef, Imported};
use photocraft_three::scene::{self, DepthMapKind, Geometry, Image, Light, LightKind, Material, RenderState, Scene, Style};
use photocraft_three::{MeshPreset, Vec3, v3};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::{EngineError, Result, Session};

const MENU: &[&str] = &["3D"];
const MENU_PRESET: &[&str] = &["3D", "New Mesh from Layer", "Mesh Preset"];
const MENU_DEPTH: &[&str] = &["3D", "New Mesh from Layer", "Depth Map to"];
const MENU_MESH: &[&str] = &["3D", "New Mesh from Layer"];
const MENU_ADD: &[&str] = &["3D", "Add Object"];
const MENU_LIGHT: &[&str] = &["3D", "Add Light"];
const MENU_LIGHT_PRESET: &[&str] = &["3D", "Light Presets"];
const MENU_STYLE: &[&str] = &["3D", "Render Style"];
const MENU_VIEW: &[&str] = &["3D", "View"];

/// Longest side of the coverage grid traced for an extrusion.
const TRACE_MAX: usize = 1024;

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn other(msg: impl Into<String>) -> EngineError {
    EngineError::Other(msg.into())
}

// ---------------------------------------------------------------- enablement

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn has_layer(s: &Session) -> std::result::Result<(), String> {
    crate::active_layer_of(s).map(|_| ())
}

fn has_3d(s: &Session) -> std::result::Result<(), String> {
    let l = crate::active_layer_of(s)?;
    if l.is_3d() { Ok(()) } else { Err("the active layer isn't a 3D layer".into()) }
}

fn has_selection(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    d.doc.selection.as_ref().map(|_| ()).ok_or_else(|| "there's no selection".into())
}

fn has_path(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.work_path.is_some() || !d.doc.paths.is_empty() { Ok(()) } else { Err("there's no path".into()) }
}

fn two_3d(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let n = d.selected_layers().iter().filter(|id| d.doc.layer(**id).is_some_and(Layer::is_3d)).count();
    if n >= 2 { Ok(()) } else { Err("select two or more 3D layers".into()) }
}

// ---------------------------------------------------------------- coordinates

/// Canvas pixels → world units on the z = 0 plane the default camera maps onto the canvas.
#[derive(Clone, Copy)]
struct Canvas {
    w: f32,
    h: f32,
}

impl Canvas {
    fn of(doc: &Document) -> Canvas {
        Canvas { w: doc.size.width.max(1) as f32, h: doc.size.height.max(1) as f32 }
    }
    fn to_world(self, x: f32, y: f32) -> [f32; 2] {
        let k = 2.0 / self.h;
        [(x - self.w * 0.5) * k, (self.h * 0.5 - y) * k]
    }
    /// World units per canvas pixel.
    fn unit(self) -> f32 {
        2.0 / self.h
    }
}

// ---------------------------------------------------------------- rendering into the layer

/// Re-render a 3D layer's pixels from its scene with the interactive renderer.
pub fn render_layer(doc: &mut Document, id: LayerId) -> Result<()> {
    let size = doc.size;
    let fmt = doc.pixel_format();
    let layer = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
    let Some(scene) = layer.scene3d.clone() else { return Err(other("the layer isn't a 3D layer")) };
    let out = photocraft_three::render_preview(&scene, size.width, size.height, scene.render.preview_aa);
    layer.content = LayerContent::Raster(surface_from(&out, fmt)?);
    Ok(())
}

/// Rendered RGBA (straight, sRGB) → a layer surface in the document's pixel format. Only tiles
/// with visible pixels are allocated.
fn surface_from(out: &photocraft_three::Output, fmt: photocraft_color::PixelFormat) -> Result<Surface> {
    let mut s = Surface::new(fmt);
    let (w, h) = (out.width as i32, out.height as i32);
    let bounds = Rect::new(0, 0, w, h);
    let n = fmt.channels();
    let mut data = Vec::new();
    let mut px = vec![0.0f32; n.max(4)];
    for tc in bounds.tiles() {
        let r = tc.rect().intersect(&bounds);
        if r.is_empty() {
            continue;
        }
        let visible = (r.y0..r.y1).any(|y| (r.x0..r.x1).any(|x| out.pixels.get((y * w + x) as usize).is_some_and(|p| p[3] > 0.0)));
        if !visible {
            continue;
        }
        data.clear();
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let p = out.pixels.get((y * w + x) as usize).copied().unwrap_or([0.0; 4]);
                let k = photocraft_raster::from_rgba_into(&fmt, p, &mut px);
                data.extend_from_slice(px.get(..k).unwrap_or(&[]));
            }
        }
        s.try_write_region(r, &data).map_err(|e| other(format!("not enough memory for the 3D render: {e:?}")))?;
    }
    Ok(s)
}

/// Change a 3D layer's scene in one history step and re-render it.
fn edit_scene(s: &mut Session, id: LayerId, label: &str, f: impl FnOnce(&mut Scene, Canvas) -> Result<Value>) -> Result<Value> {
    s.edit(label, |doc, _| {
        let canvas = Canvas::of(doc);
        let layer = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let scene = layer.scene3d.as_mut().ok_or_else(|| other("the layer isn't a 3D layer"))?;
        let scene = Arc::make_mut(scene);
        let r = f(scene, canvas)?;
        scene.state = RenderState::Preview;
        render_layer(doc, id)?;
        Ok(r)
    })
}

/// The 3D layer the params name (`layer`), else the active layer.
fn layer_3d(s: &Session, p: &Value) -> Result<LayerId> {
    let id = layer_param(s, p)?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    if st.doc.layer(id).is_some_and(Layer::is_3d) { Ok(id) } else { Err(other("the layer isn't a 3D layer")) }
}

/// The scene of the active (or named) 3D layer, for reading.
pub fn scene_of(s: &Session, layer: Option<LayerId>) -> Option<Arc<Scene>> {
    let st = s.active()?;
    let id = layer.or(st.active_layer)?;
    st.doc.layer(id)?.scene3d.clone()
}

/// Make `id` (or a new layer above the active one when `id` is None) a 3D layer showing `scene`.
fn install(s: &mut Session, label: &str, id: Option<LayerId>, name: &str, scene: Scene) -> Result<Value> {
    let scene = Arc::new(scene);
    let lid = s.edit(label, |doc, active| {
        let lid = match id.and_then(|i| doc.layer(i).map(|_| i)) {
            Some(i) => i,
            None => {
                let l = Layer::raster(name, doc.pixel_format());
                let lid = l.id;
                doc.insert_above(*active, l);
                lid
            }
        };
        let layer = doc.layer_mut(lid).ok_or(EngineError::NoLayer(lid))?;
        layer.scene3d = Some(scene);
        // A converted layer becomes a plain pixel layer showing the render (as Photoshop did).
        layer.content = LayerContent::Raster(Surface::new(doc_pixel(layer)));
        layer.mask = None;
        layer.vector_mask = None;
        render_layer(doc, lid)?;
        *active = Some(lid);
        Ok(lid)
    })?;
    if let Some(st) = s.active_mut() {
        st.active_layer = Some(lid);
    }
    let summary = scene_of(s, Some(lid)).map(|sc| summary(&sc)).unwrap_or(Value::Null);
    Ok(json!({"layer": lid.0, "scene": summary}))
}

fn doc_pixel(layer: &Layer) -> photocraft_color::PixelFormat {
    match &layer.content {
        LayerContent::Raster(s) => s.format(),
        _ => photocraft_color::PixelFormat::new(photocraft_color::ColorMode::Rgb, photocraft_color::SampleType::U8, true),
    }
}

/// Short JSON description of a scene.
pub fn summary(s: &Scene) -> Value {
    json!({
        "objects": s.objects.iter().map(|o| json!({
            "id": o.id, "name": o.name, "visible": o.visible, "kind": o.geometry.kind_name(),
            "triangles": o.mesh().triangle_count(), "materials": o.materials.iter().map(|m| m.name.clone()).collect::<Vec<_>>(),
            "position": o.transform.position.to_array(), "rotation": o.transform.rotation.to_array(), "scale": o.transform.scale.to_array(),
        })).collect::<Vec<_>>(),
        "lights": s.lights.iter().map(|l| json!({"id": l.id, "name": l.name, "kind": l.kind.id(), "visible": l.visible})).collect::<Vec<_>>(),
        "textures": s.textures.iter().map(|t| json!({"id": t.id, "name": t.name, "width": t.image.width, "height": t.image.height})).collect::<Vec<_>>(),
        "triangles": s.triangle_count(),
        "style": s.render.style.id(),
        "rayTraced": s.state == RenderState::RayTraced,
    })
}

// ---------------------------------------------------------------- layer pixels

/// The layer rendered alone (any kind: pixels, type, shapes, smart objects), cropped to its
/// visible bounds: (bounds, straight RGBA).
fn layer_pixels(doc: &Document, id: LayerId) -> Result<(Rect, Vec<[f32; 4]>)> {
    let layer = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let full = photocraft_compose::render_layer_in(doc, layer, doc.bounds());
    let (w, h) = (full.rect.width() as i32, full.rect.height() as i32);
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for y in 0..h {
        for x in 0..w {
            if full.px.get((y * w + x) as usize).is_some_and(|p| p[3] > 0.004) {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x0 > x1 {
        return Ok((Rect::EMPTY, Vec::new()));
    }
    let r = Rect::new(x0 + full.rect.x0, y0 + full.rect.y0, x1 + full.rect.x0, y1 + full.rect.y0);
    let mut px = Vec::with_capacity((r.width() * r.height()) as usize);
    for y in y0..y1 {
        for x in x0..x1 {
            px.push(full.px.get((y * w + x) as usize).copied().unwrap_or([0.0; 4]));
        }
    }
    Ok((r, px))
}

fn image_of(w: u32, h: u32, px: &[[f32; 4]]) -> Option<Image> {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    Image::new(w, h, px.iter().flat_map(|p| [q(p[0]), q(p[1]), q(p[2]), q(p[3])]).collect())
}

/// Average colour of the opaque pixels (for an extrusion's sides).
fn average_color(px: &[[f32; 4]]) -> [f32; 3] {
    let (mut acc, mut n) = ([0.0f32; 3], 0.0f32);
    for p in px.iter().filter(|p| p[3] > 0.5) {
        for k in 0..3 {
            acc[k] += p[k];
        }
        n += 1.0;
    }
    if n == 0.0 { [0.7, 0.7, 0.7] } else { [acc[0] / n, acc[1] / n, acc[2] / n] }
}

// ---------------------------------------------------------------- extrusion

/// Trace `coverage` (`w × h` at canvas offset `origin`, one grid cell = `step` canvas pixels)
/// into world-space shapes.
fn shapes_from_coverage(coverage: &[f32], w: usize, h: usize, origin: (f32, f32), step: f32, canvas: Canvas) -> Vec<Shape> {
    let loops = extrude::trace(coverage, w, h);
    let world: Vec<Contour> = loops
        .iter()
        .map(|c| extrude::simplify(c, 0.6))
        .map(|c| c.iter().map(|p| canvas.to_world(origin.0 + p[0] * step, origin.1 + p[1] * step)).collect())
        .collect();
    let min_area = (2.0 * step * canvas.unit()).powi(2);
    extrude::shapes(&world, min_area)
}

/// Downsample coverage to at most [`TRACE_MAX`] on its long side: (grid, w, h, step).
fn coverage_grid(r: Rect, sample: &dyn Fn(i32, i32) -> f32) -> (Vec<f32>, usize, usize, f32) {
    let (rw, rh) = (r.width() as usize, r.height() as usize);
    let step = (rw.max(rh) as f32 / TRACE_MAX as f32).max(1.0);
    let (w, h) = (((rw as f32 / step).ceil() as usize).max(1), ((rh as f32 / step).ceil() as usize).max(1));
    let mut g = vec![0.0; w * h];
    for gy in 0..h {
        for gx in 0..w {
            let x = r.x0 + ((gx as f32 + 0.5) * step) as i32;
            let y = r.y0 + ((gy as f32 + 0.5) * step) as i32;
            g[gy * w + gx] = sample(x.min(r.x1 - 1), y.min(r.y1 - 1));
        }
    }
    (g, w, h, step)
}

/// Default extrusion for an outline `height` world units tall.
fn default_params(height: f32) -> ExtrudeParams {
    ExtrudeParams { depth: (height * 0.4).clamp(0.03, 1.5), bevel_width: (height * 0.02).clamp(0.001, 0.05), ..Default::default() }
}

fn extrusion_scene(shapes: Vec<Shape>, bounds: Rect, canvas: Canvas, materials: Vec<Material>, name: &str) -> Result<Scene> {
    if shapes.is_empty() {
        return Err(other("there's nothing to extrude: the outline is empty"));
    }
    let [x0, y1] = canvas.to_world(bounds.x0 as f32, bounds.y0 as f32);
    let [x1, y0] = canvas.to_world(bounds.x1 as f32, bounds.y1 as f32);
    let params = default_params((y1 - y0).abs());
    let depth = params.depth;
    let mut sc = Scene::default();
    let id = sc.add_object(name, Geometry::Extrusion { shapes, params, uv_box: [x0, y0, x1, y1] }, materials);
    if let Some(o) = sc.object_mut(id) {
        // The front face stays exactly where the outline was on the canvas.
        o.transform.position.z = -depth * 0.5;
    }
    sc.fit_ground();
    Ok(sc)
}

fn extrusion_materials(front: Option<u32>, color: [f32; 3]) -> Vec<Material> {
    extrude::EXTRUSION_SLOTS
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let mut m = Material { name: (*n).into(), diffuse: color, ..Default::default() };
            if i == extrude::SLOT_FRONT as usize {
                m.diffuse_texture = front;
            }
            m
        })
        .collect()
}

fn new_extrusion_from_layer(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = st.doc.clone();
    let canvas = Canvas::of(&doc);
    let (r, px) = layer_pixels(&doc, id)?;
    if r.is_empty() {
        return Err(other("the layer is empty: there's nothing to extrude"));
    }
    let rw = r.width() as i32;
    let alpha = |x: i32, y: i32| px.get(((y - r.y0) * rw + (x - r.x0)) as usize).map_or(0.0, |c| c[3]);
    let (g, w, h, step) = coverage_grid(r, &alpha);
    let shapes = shapes_from_coverage(&g, w, h, (r.x0 as f32, r.y0 as f32), step, canvas);
    let name = doc.layer(id).map_or_else(|| "Extrusion".to_string(), |l| l.name.clone());
    let mut sc = Scene::default();
    let tex = image_of(r.width(), r.height(), &px).map(|img| sc.add_texture(&format!("{name} pixels"), img));
    let color = average_color(&px);
    let mut built = extrusion_scene(shapes, r, canvas, extrusion_materials(tex, color), &name)?;
    built.textures = sc.textures;
    install(s, "New 3D Extrusion", Some(id), &name, built)
}

fn new_extrusion_from_selection(s: &mut Session, _p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = st.doc.clone();
    let canvas = Canvas::of(&doc);
    let sel = doc.selection.as_ref().ok_or_else(|| other("there's no selection"))?;
    let r = sel.content_bounds().intersect(&doc.bounds());
    if r.is_empty() {
        return Err(other("the selection is empty"));
    }
    let (g, w, h, step) = coverage_grid(r, &|x, y| sel.sample_channel(x, y, 0));
    let shapes = shapes_from_coverage(&g, w, h, (r.x0 as f32, r.y0 as f32), step, canvas);
    let sc = extrusion_scene(shapes, r, canvas, extrusion_materials(None, [0.75, 0.75, 0.78]), "Selection Extrusion")?;
    install(s, "New 3D Extrusion from Selection", None, "Selection Extrusion", sc)
}

fn new_extrusion_from_path(s: &mut Session, _p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = st.doc.clone();
    let canvas = Canvas::of(&doc);
    let path = doc.work_path.clone().or_else(|| doc.paths.first().map(|n| n.path.clone())).ok_or_else(|| other("there's no path"))?;
    let bounds = doc.bounds();
    let cov = photocraft_vector::path_coverage(&path, bounds);
    let bw = bounds.width() as i32;
    let at = |x: i32, y: i32| cov.get((y * bw + x) as usize).copied().unwrap_or(0.0);
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for y in 0..bounds.height() as i32 {
        for x in 0..bw {
            if at(x, y) > 0.0 {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            }
        }
    }
    if x0 > x1 {
        return Err(other("the path encloses no area"));
    }
    let r = Rect::new(x0, y0, x1, y1);
    let (g, w, h, step) = coverage_grid(r, &at);
    let shapes = shapes_from_coverage(&g, w, h, (r.x0 as f32, r.y0 as f32), step, canvas);
    let sc = extrusion_scene(shapes, r, canvas, extrusion_materials(None, [0.75, 0.75, 0.78]), "Path Extrusion")?;
    install(s, "New 3D Extrusion from Path", None, "Path Extrusion", sc)
}

/// Split Extrusion: each separate outline of an extrusion becomes its own object.
fn split_extrusion(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_3d(s, p)?;
    let obj = p.get("object").and_then(Value::as_u64).map(|v| v as u32);
    edit_scene(s, id, "Split Extrusion", |sc, _| {
        let target = obj.or_else(|| sc.objects.iter().find(|o| matches!(&o.geometry, Geometry::Extrusion { shapes, .. } if shapes.len() > 1)).map(|o| o.id)).ok_or_else(|| other("no extrusion with more than one outline"))?;
        let o = sc.object(target).cloned().ok_or_else(|| other("no such object"))?;
        let Geometry::Extrusion { shapes, params, uv_box } = &o.geometry else { return Err(other("the object isn't an extrusion")) };
        if shapes.len() < 2 {
            return Err(other("the extrusion has a single outline"));
        }
        sc.objects.retain(|x| x.id != target);
        let mut made = Vec::new();
        for (i, sh) in shapes.iter().enumerate() {
            let mut n = o.clone();
            n.id = sc.fresh_id();
            n.name = format!("{} {}", o.name, i + 1);
            n.geometry = Geometry::Extrusion { shapes: vec![sh.clone()], params: params.clone(), uv_box: *uv_box };
            n.rebuild();
            made.push(n.id);
            sc.objects.push(n);
        }
        Ok(json!({"objects": made}))
    })
}

// ---------------------------------------------------------------- meshes from a layer

/// The layer's pixels as a texture and where they sit, or None for an empty layer.
fn layer_texture(doc: &Document, id: LayerId) -> Result<Option<(Rect, Image)>> {
    let (r, px) = layer_pixels(doc, id)?;
    if r.is_empty() {
        return Ok(None);
    }
    Ok(image_of(r.width(), r.height(), &px).map(|i| (r, i)))
}

fn new_postcard(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let canvas = Canvas::of(&doc);
    let (r, img) = layer_texture(&doc, id)?.ok_or_else(|| other("the layer is empty"))?;
    let name = doc.layer(id).map_or_else(|| "Postcard".to_string(), |l| l.name.clone());
    let mut sc = Scene::default();
    let tex = sc.add_texture(&name, img);
    let aspect = r.width() as f32 / r.height().max(1) as f32;
    let mats = vec![Material { name: "Postcard Material".into(), diffuse_texture: Some(tex), specular: [0.1, 0.1, 0.1], ..Default::default() }];
    let oid = sc.add_object(&name, Geometry::Preset { preset: MeshPreset::Plane, aspect }, mats);
    let [cx, cy] = canvas.to_world((r.x0 + r.x1) as f32 * 0.5, (r.y0 + r.y1) as f32 * 0.5);
    if let Some(o) = sc.object_mut(oid) {
        let s = r.height() as f32 * canvas.unit();
        o.transform.position = v3(cx, cy, 0.0);
        o.transform.scale = v3(s, s, s);
    }
    sc.fit_ground();
    install(s, "New 3D Postcard", Some(id), &name, sc)
}

fn new_mesh_preset(s: &mut Session, p: &Value, preset: MeshPreset) -> Result<Value> {
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let canvas = Canvas::of(&doc);
    let id = layer_param(s, p).ok();
    let tex = match id {
        Some(i) => layer_texture(&doc, i)?,
        None => None,
    };
    let name = id.and_then(|i| doc.layer(i)).map_or_else(|| preset.label().to_string(), |l| l.name.clone());
    let mut sc = Scene::default();
    let tex_id = tex.as_ref().map(|(_, img)| sc.add_texture(&name, img.clone()));
    let mats: Vec<Material> = preset
        .slot_names()
        .iter()
        .map(|n| Material { name: (*n).into(), diffuse_texture: tex_id, ..Default::default() })
        .collect();
    let aspect = tex.as_ref().map_or(1.0, |(r, _)| r.width() as f32 / r.height().max(1) as f32);
    let oid = sc.add_object(preset.label(), Geometry::Preset { preset, aspect }, mats);
    let centre = tex.as_ref().map(|(r, _)| canvas.to_world((r.x0 + r.x1) as f32 * 0.5, (r.y0 + r.y1) as f32 * 0.5)).unwrap_or([0.0, 0.0]);
    if let Some(o) = sc.object_mut(oid) {
        o.transform.position = v3(centre[0], centre[1], 0.0);
        o.transform.rotation = match preset {
            MeshPreset::Cube | MeshPreset::CubeWrap | MeshPreset::Pyramid => v3(18.0, -32.0, 0.0),
            MeshPreset::Plane | MeshPreset::TwoSidedPlane => Vec3::ZERO,
            _ => v3(12.0, 0.0, 0.0),
        };
    }
    sc.fit_ground();
    install(s, &format!("New Mesh: {}", preset.label()), id, &name, sc)
}

fn new_depth_map(s: &mut Session, p: &Value, kind: DepthMapKind) -> Result<Value> {
    let id = layer_param(s, p)?;
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let canvas = Canvas::of(&doc);
    let (r, px) = layer_pixels(&doc, id)?;
    if r.is_empty() {
        return Err(other("the layer is empty"));
    }
    let gray: Vec<f32> = px.iter().map(|c| (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) * c[3]).collect();
    let (gw, gh, heights) = photocraft_three::depth::grid_from(&gray, r.width() as usize, r.height() as usize);
    let name = doc.layer(id).map_or_else(|| "Depth Map".to_string(), |l| l.name.clone());
    let mut sc = Scene::default();
    let tex = image_of(r.width(), r.height(), &px).map(|i| sc.add_texture(&name, i));
    let aspect = r.width() as f32 / r.height().max(1) as f32;
    let strength = p.get("strength").and_then(Value::as_f64).map_or(0.25, |v| v as f32);
    let mats = vec![Material { name: "Depth Map Material".into(), diffuse_texture: tex, ..Default::default() }, Material { name: "Back Material".into(), diffuse_texture: tex, ..Default::default() }];
    let oid = sc.add_object(&name, Geometry::DepthMap { map: kind, width: gw, height: gh, heights, strength, aspect }, mats);
    let [cx, cy] = canvas.to_world((r.x0 + r.x1) as f32 * 0.5, (r.y0 + r.y1) as f32 * 0.5);
    if let Some(o) = sc.object_mut(oid) {
        let k = r.height() as f32 * canvas.unit();
        o.transform.position = v3(cx, cy, 0.0);
        o.transform.scale = v3(k, k, k);
        if matches!(kind, DepthMapKind::Plane | DepthMapKind::TwoSidedPlane) {
            o.transform.rotation = v3(-25.0, 20.0, 0.0);
        }
    }
    sc.fit_ground();
    install(s, "New Mesh from Depth Map", Some(id), &name, sc)
}

// ---------------------------------------------------------------- files

/// Decode an image file (any format PhotoCraft opens) into a texture.
pub fn decode_image(name: &str, bytes: &[u8]) -> Option<Image> {
    let doc = crate::file_cmds::import(name, bytes).ok()?;
    let buf = photocraft_compose::flatten(&doc);
    image_of(buf.rect.width(), buf.rect.height(), &buf.px)
}

fn encode_png(img: &Image) -> Option<Vec<u8>> {
    let raw = photocraft_codecs::Image::from_raw(img.width, img.height, photocraft_codecs::ChannelLayout::Rgba, photocraft_codecs::SampleType::U8, img.rgba.clone()).ok()?;
    photocraft_codecs::encode(&raw, photocraft_codecs::Format::Png, &photocraft_codecs::EncodeOptions::default()).ok()
}

/// Build a scene from an imported model: centred, about the height of the canvas's middle
/// third, standing on the ground plane, textures decoded through `resolve`.
pub fn scene_from_import(imported: Imported, resolve: &dyn Fn(&str) -> Option<Vec<u8>>) -> (Scene, Vec<String>) {
    let mut sc = Scene::default();
    let mut warnings = imported.warnings;
    let mut cache: std::collections::HashMap<String, Option<u32>> = std::collections::HashMap::new();
    let mut texture = |sc: &mut Scene, r: &Option<ImageRef>, warnings: &mut Vec<String>| -> Option<u32> {
        match r.as_ref()? {
            ImageRef::Path(p) => {
                if let Some(t) = cache.get(p) {
                    return *t;
                }
                let t = match resolve(p) {
                    Some(b) => match decode_image(p, &b) {
                        Some(img) => Some(sc.add_texture(p, img)),
                        None => {
                            warnings.push(format!("Texture \"{p}\" couldn't be decoded."));
                            None
                        }
                    },
                    None => {
                        warnings.push(format!("Texture \"{p}\" wasn't found next to the model."));
                        None
                    }
                };
                cache.insert(p.clone(), t);
                t
            }
            ImageRef::Bytes(b) => match decode_image("texture.png", b) {
                Some(img) => Some(sc.add_texture("Embedded texture", img)),
                None => {
                    warnings.push("An embedded texture couldn't be decoded.".into());
                    None
                }
            },
        }
    };
    let mut ids = Vec::new();
    for o in imported.objects {
        let mats: Vec<Material> = o
            .materials
            .iter()
            .map(|im| {
                let mut m = im.material.clone();
                m.diffuse_texture = texture(&mut sc, &im.diffuse, &mut warnings);
                m.bump_texture = texture(&mut sc, &im.bump, &mut warnings);
                m.normal_texture = texture(&mut sc, &im.normal, &mut warnings);
                m.opacity_texture = texture(&mut sc, &im.opacity, &mut warnings);
                m.sanitize();
                m
            })
            .collect();
        ids.push(sc.add_object(&o.name, Geometry::Mesh { mesh: Arc::new(o.mesh) }, mats));
    }
    // Normalise: centre and scale the whole model together.
    let b = sc.bounds();
    if !b.is_empty() {
        let size = b.size().max_elem().max(1e-6);
        let k = 1.4 / size;
        let c = b.center();
        for o in &mut sc.objects {
            o.transform.position = (o.transform.position - c) * k;
            o.transform.scale = o.transform.scale * k;
            // Geometry is baked in world space by the importer, so the scale/offset apply to it.
        }
        sc.fit_ground();
        let fitted = sc.bounds();
        sc.frame(fitted);
        sc.camera.yaw = 25.0;
        sc.camera.pitch = 15.0;
    }
    (sc, warnings)
}

/// 3D › New 3D Layer from File with the file's name and bytes (the file pickers' entry point).
/// `resolve` fetches files next to the model (MTL, textures, buffers).
pub fn new_layer_from_bytes(s: &mut Session, name: &str, bytes: &[u8], resolve: &dyn Fn(&str) -> Option<Vec<u8>>) -> Result<Value> {
    if s.active().is_none() {
        return Err(EngineError::NoDocument);
    }
    let imported = photocraft_three::io::import(name, bytes, resolve).map_err(|e| other(e.to_string()))?;
    let (sc, warnings) = scene_from_import(imported, resolve);
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let mut v = install(s, "New 3D Layer from File", None, base, sc)?;
    v["warnings"] = json!(warnings);
    Ok(v)
}

/// Sibling-file reader for a model at `path`: relative names, no escaping the folder upwards.
#[cfg(not(target_arch = "wasm32"))]
pub fn sibling_resolver(path: &str) -> impl Fn(&str) -> Option<Vec<u8>> {
    let dir = std::path::Path::new(path).parent().map(std::path::Path::to_path_buf).unwrap_or_default();
    move |rel: &str| {
        let rel = rel.replace('\\', "/");
        if rel.split('/').any(|c| c == "..") || std::path::Path::new(&rel).is_absolute() {
            return None;
        }
        let p = dir.join(&rel);
        photocraft_format::read_file(&p).ok().or_else(|| {
            // Models often name textures by a path from the author's machine: try the bare name.
            let file = rel.rsplit('/').next()?;
            photocraft_format::read_file(&dir.join(file)).ok()
        })
    }
}

fn path_param<'a>(cmd: &str, p: &'a Value) -> Result<&'a str> {
    p.get("path").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or_else(|| bad(cmd, "`path` (a file path) is required"))
}

#[cfg(not(target_arch = "wasm32"))]
fn new_layer_from_file(s: &mut Session, p: &Value) -> Result<Value> {
    let path = path_param("3d.newLayerFromFile", p)?.to_string();
    let bytes = photocraft_format::read_file(std::path::Path::new(&path)).map_err(|e| other(format!("can't read {path}: {e}")))?;
    let resolve = sibling_resolver(&path);
    new_layer_from_bytes(s, &path, &bytes, &resolve)
}

#[cfg(target_arch = "wasm32")]
fn new_layer_from_file(_s: &mut Session, _p: &Value) -> Result<Value> {
    Err(other("open a model with the file picker on the web"))
}

/// The files 3D › Export 3D Layer writes for `name` (format from its extension).
pub fn export_files(scene: &Scene, name: &str) -> Result<photocraft_three::io::Files> {
    photocraft_three::io::export(scene, name, &encode_png).map_err(|e| other(e.to_string()))
}

#[cfg(not(target_arch = "wasm32"))]
fn export_layer(s: &mut Session, p: &Value) -> Result<Value> {
    let path = path_param("3d.exportLayer", p)?.to_string();
    let id = layer_3d(s, p)?;
    let scene = scene_of(s, Some(id)).ok_or_else(|| other("the layer isn't a 3D layer"))?;
    let files = export_files(&scene, &path)?;
    let dir = std::path::Path::new(&path).parent().map(std::path::Path::to_path_buf).unwrap_or_default();
    let mut written = Vec::new();
    for (i, (name, bytes)) in files.iter().enumerate() {
        let target = if i == 0 { std::path::PathBuf::from(&path) } else { dir.join(name) };
        photocraft_format::atomic_write(&target, bytes).map_err(|e| other(format!("{}: {e}", target.display())))?;
        written.push(target.to_string_lossy().into_owned());
    }
    Ok(json!({"files": written}))
}

#[cfg(target_arch = "wasm32")]
fn export_layer(_s: &mut Session, _p: &Value) -> Result<Value> {
    Err(other("export with the download dialog on the web"))
}

// ---------------------------------------------------------------- render, rasterize, merge

/// 3D › Render 3D Layer: ray trace the scene (a background job when started from the UI) and
/// put the result in the layer's pixels. Any later scene edit returns to the live preview.
fn render(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_3d(s, p)?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let (w, h) = (st.doc.size.width, st.doc.size.height);
    let fmt = st.doc.pixel_format();
    let scene = st.doc.layer(id).and_then(|l| l.scene3d.clone()).ok_or_else(|| other("the layer isn't a 3D layer"))?;
    let samples = p.get("samples").and_then(Value::as_u64).map(|v| v.clamp(1, 256) as u32);
    crate::jobs::run(
        s,
        "Render 3D Layer",
        true,
        move |ctx| {
            let mut sc = (*scene).clone();
            if let Some(n) = samples {
                sc.render.samples = n;
            }
            ctx.progress(0.0, "Ray tracing");
            let out = photocraft_three::render_traced(&sc, w, h, &|| ctx.cancelled(), &|f| ctx.progress(f * 0.98, ""));
            let out = out.ok_or(EngineError::Cancelled)?;
            surface_from(&out, fmt)
        },
        move |s, surface| {
            s.edit("Render 3D Layer", |doc, _| {
                let layer = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
                let sc = layer.scene3d.as_mut().ok_or_else(|| other("the layer isn't a 3D layer"))?;
                Arc::make_mut(sc).state = RenderState::RayTraced;
                layer.content = LayerContent::Raster(surface);
                Ok(())
            })?;
            Ok(json!({"layer": id.0, "rayTraced": true}))
        },
    )
}

fn rasterize(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_3d(s, p)?;
    s.edit("Rasterize 3D", |doc, _| {
        doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.scene3d = None;
        Ok(())
    })?;
    Ok(json!({"layer": id.0}))
}

/// 3D › Merge 3D Layers: the selected 3D layers' scenes become one, in the topmost.
fn merge_layers(s: &mut Session, _p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let ids: Vec<LayerId> = st.selected_layers().into_iter().filter(|id| st.doc.layer(*id).is_some_and(Layer::is_3d)).collect();
    if ids.len() < 2 {
        return Err(other("select two or more 3D layers"));
    }
    // The active layer keeps the merged scene (else the first selected).
    let keep = st.active_layer.filter(|a| ids.contains(a)).or_else(|| ids.first().copied()).ok_or_else(|| other("no layers"))?;
    let keep_v = keep;
    s.edit("Merge 3D Layers", move |doc, active| {
        let mut merged: Scene = doc.layer(keep_v).and_then(|l| l.scene3d.as_deref().cloned()).ok_or(EngineError::NoLayer(keep_v))?;
        for id in ids.iter().filter(|i| **i != keep_v) {
            if let Some(sc) = doc.layer(*id).and_then(|l| l.scene3d.clone()) {
                merged.merge(&sc);
            }
            doc.remove(*id);
        }
        doc.layer_mut(keep_v).ok_or(EngineError::NoLayer(keep_v))?.scene3d = Some(Arc::new(merged));
        render_layer(doc, keep_v)?;
        *active = Some(keep_v);
        Ok(())
    })?;
    Ok(json!({"layer": keep.0}))
}

// ---------------------------------------------------------------- scene editing

/// Deep-merge `patch` into `target` (objects merge key by key; anything else replaces).
fn merge_json(target: &mut Value, patch: &Value) {
    match (target, patch) {
        (Value::Object(t), Value::Object(p)) => {
            for (k, v) in p {
                merge_json(t.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        (t, p) => *t = p.clone(),
    }
}

fn patch<T: serde::Serialize + serde::de::DeserializeOwned>(cmd: &str, value: &mut T, values: &Value) -> Result<()> {
    let mut v = serde_json::to_value(&*value).map_err(|e| other(e.to_string()))?;
    merge_json(&mut v, values);
    *value = serde_json::from_value(v).map_err(|e| bad(cmd, format!("bad value: {e}")))?;
    Ok(())
}

const SET: &str = "3d.set";

/// `3d.set`: change any scene property (the 3D panel and Properties drive this).
fn set(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_3d(s, p)?;
    let target = p.get("target").and_then(Value::as_str).ok_or_else(|| bad(SET, "`target` is required: object, material, light, camera, environment or render"))?.to_string();
    let values = p.get("values").cloned().filter(Value::is_object).ok_or_else(|| bad(SET, "`values` must be an object"))?;
    let node = p.get("id").and_then(Value::as_u64).map(|v| v as u32);
    let slot = p.get("slot").and_then(Value::as_u64).unwrap_or(0) as usize;
    let label = match target.as_str() {
        "object" => "3D Object Properties",
        "material" => "3D Material",
        "light" => "3D Light",
        "camera" => "3D Camera",
        "environment" => "3D Environment",
        "render" => "3D Render Settings",
        _ => return Err(bad(SET, format!("unknown target `{target}`"))),
    };
    edit_scene(s, id, label, |sc, _| {
        match target.as_str() {
            "object" => {
                let oid = node.ok_or_else(|| bad(SET, "`id` (object id) is required"))?;
                let o = sc.object_mut(oid).ok_or_else(|| bad(SET, format!("no object {oid}")))?;
                // The mesh cache isn't serialized; geometry edits rebuild it.
                let geometry_before = o.geometry.clone();
                patch(SET, o, &values)?;
                o.transform.sanitize();
                if o.geometry != geometry_before || o.mesh.0.is_none() {
                    if let Geometry::Extrusion { params, .. } = &mut o.geometry {
                        *params = params.sanitized();
                    }
                    o.rebuild();
                }
                for m in &mut o.materials {
                    m.sanitize();
                }
            }
            "material" => {
                let oid = node.ok_or_else(|| bad(SET, "`id` (object id) is required"))?;
                let o = sc.object_mut(oid).ok_or_else(|| bad(SET, format!("no object {oid}")))?;
                let m = o.materials.get_mut(slot).ok_or_else(|| bad(SET, format!("no material slot {slot}")))?;
                patch(SET, m, &values)?;
                m.sanitize();
            }
            "light" => {
                let lid = node.ok_or_else(|| bad(SET, "`id` (light id) is required"))?;
                let l = sc.light_mut(lid).ok_or_else(|| bad(SET, format!("no light {lid}")))?;
                patch(SET, l, &values)?;
                l.id = lid;
                l.sanitize();
            }
            "camera" => {
                patch(SET, &mut sc.camera, &values)?;
                sc.camera.sanitize();
            }
            "environment" => patch(SET, &mut sc.environment, &values)?,
            _ => patch(SET, &mut sc.render, &values)?,
        }
        sc.rebuild_light();
        Ok(json!({"ok": true}))
    })
}

trait RebuildLight {
    fn rebuild_light(&mut self);
}

impl RebuildLight for Scene {
    /// Clamp the values every patch can reach without rebuilding meshes.
    fn rebuild_light(&mut self) {
        self.render.samples = self.render.samples.clamp(1, 256);
        self.render.bounces = self.render.bounces.min(8);
        self.render.preview_aa = self.render.preview_aa.clamp(1, 3);
        let e = &mut self.environment;
        e.ibl_intensity = e.ibl_intensity.clamp(0.0, 10.0);
        e.ground_shadow_opacity = e.ground_shadow_opacity.clamp(0.0, 1.0);
        e.ground_reflection_opacity = e.ground_reflection_opacity.clamp(0.0, 1.0);
        if !e.ground_y.is_finite() {
            e.ground_y = 0.0;
        }
    }
}

/// `3d.navigate`: the 3D Orbit / Roll / Pan / Slide / Scale tools, on the camera or an object.
/// `dx`, `dy` are canvas pixels dragged.
fn navigate(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "3d.navigate";
    let id = layer_3d(s, p)?;
    let mode = p.get("mode").and_then(Value::as_str).ok_or_else(|| bad(CMD, "`mode` is required: orbit, roll, pan, slide or scale"))?.to_string();
    let f = |k: &str| p.get(k).and_then(Value::as_f64).filter(|v| v.is_finite()).map_or(0.0, |v| v.clamp(-1e5, 1e5) as f32);
    let (dx, dy) = (f("dx"), f("dy"));
    let object = p.get("object").and_then(Value::as_u64).map(|v| v as u32);
    edit_scene(s, id, "3D Navigate", |sc, canvas| {
        let cam = sc.camera.clone();
        let fwd = (cam.target - cam.eye()).normalize();
        let up = cam.up();
        let right = fwd.cross(up).normalize();
        // World units per canvas pixel at the target's distance.
        let k = if cam.orthographic { cam.ortho_height / canvas.h } else { 2.0 * cam.distance * (cam.fov.to_radians() * 0.5).tan() / canvas.h };
        match object {
            Some(oid) => {
                let o = sc.object_mut(oid).ok_or_else(|| bad(CMD, format!("no object {oid}")))?;
                let t = &mut o.transform;
                match mode.as_str() {
                    "orbit" => {
                        t.rotation.y += dx * 0.5;
                        t.rotation.x += dy * 0.5;
                    }
                    "roll" => t.rotation.z -= dx * 0.5,
                    "pan" => t.position = t.position + right * (dx * k) - up * (dy * k),
                    "slide" => t.position = t.position - fwd * (dy * k),
                    "scale" => t.scale = t.scale * (-dy * 0.005).exp(),
                    _ => return Err(bad(CMD, format!("unknown mode `{mode}`"))),
                }
                t.sanitize();
            }
            None => {
                let c = &mut sc.camera;
                match mode.as_str() {
                    "orbit" => {
                        c.yaw -= dx * 0.4;
                        c.pitch += dy * 0.4;
                    }
                    "roll" => c.roll += dx * 0.4,
                    "pan" => c.target = c.target - right * (dx * k) + up * (dy * k),
                    "slide" => {
                        let z = (dy * 0.005).exp();
                        c.distance *= z;
                        c.ortho_height *= z;
                    }
                    "scale" => c.fov *= (dy * 0.004).exp(),
                    _ => return Err(bad(CMD, format!("unknown mode `{mode}`"))),
                }
                c.sanitize();
            }
        }
        Ok(json!({"ok": true}))
    })
}

fn add_object(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "3d.addObject";
    let id = layer_3d(s, p)?;
    let preset = p.get("preset").and_then(Value::as_str).and_then(MeshPreset::from_id).ok_or_else(|| bad(CMD, "`preset` is required: cube, sphere, cylinder, cone, donut, ring, hat, pyramid, sodaCan, wineBottle, cubeWrap, plane"))?;
    edit_scene(s, id, "Add 3D Object", |sc, _| {
        let mats = preset.slot_names().iter().map(|n| Material::named(n)).collect();
        let n = sc.objects.len() + 1;
        let oid = sc.add_object(&format!("{} {n}", preset.label()), Geometry::Preset { preset, aspect: 1.0 }, mats);
        let target = sc.camera.target;
        if let Some(o) = sc.object_mut(oid) {
            o.transform.position = target;
            o.transform.scale = Vec3::splat(0.6);
        }
        sc.drop_to_ground(oid);
        Ok(json!({"object": oid}))
    })
}

fn add_light(s: &mut Session, p: &Value, kind: LightKind) -> Result<Value> {
    let id = layer_3d(s, p)?;
    edit_scene(s, id, "Add 3D Light", |sc, _| {
        let n = sc.lights.iter().filter(|l| l.kind == kind).count() + 1;
        let lid = sc.fresh_id();
        let name = match kind {
            LightKind::Infinite => format!("Infinite Light {n}"),
            LightKind::Spot => format!("Spot Light {n}"),
            LightKind::Point => format!("Point Light {n}"),
        };
        let b = sc.bounds();
        let c = if b.is_empty() { Vec3::ZERO } else { b.center() };
        let mut l = Light { id: lid, name, kind, position: c + v3(1.5, 2.0, 2.0), falloff: kind == LightKind::Point, ..Default::default() };
        if kind != LightKind::Infinite {
            l.aim_at(c);
        }
        sc.lights.push(l);
        Ok(json!({"light": lid}))
    })
}

fn delete_node(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "3d.deleteNode";
    let id = layer_3d(s, p)?;
    let node = p.get("id").and_then(Value::as_u64).map(|v| v as u32).ok_or_else(|| bad(CMD, "`id` (object or light id) is required"))?;
    edit_scene(s, id, "Delete 3D Node", |sc, _| {
        let (no, nl) = (sc.objects.len(), sc.lights.len());
        sc.objects.retain(|o| o.id != node);
        sc.lights.retain(|l| l.id != node);
        if sc.objects.len() == no && sc.lights.len() == nl {
            return Err(bad(CMD, format!("no object or light {node}")));
        }
        sc.prune_textures();
        Ok(json!({"deleted": node}))
    })
}

fn duplicate_object(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "3d.duplicateObject";
    let id = layer_3d(s, p)?;
    let node = p.get("id").and_then(Value::as_u64).map(|v| v as u32);
    edit_scene(s, id, "Duplicate 3D Object", |sc, _| {
        let src = node.or_else(|| sc.objects.last().map(|o| o.id)).ok_or_else(|| bad(CMD, "the scene has no objects"))?;
        let mut o = sc.object(src).cloned().ok_or_else(|| bad(CMD, format!("no object {src}")))?;
        o.id = sc.fresh_id();
        o.name = format!("{} copy", o.name);
        let w = o.world_bounds().size().x.max(0.1);
        o.transform.position.x += w * 1.1;
        let nid = o.id;
        sc.objects.push(o);
        Ok(json!({"object": nid}))
    })
}

fn move_to_ground(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_3d(s, p)?;
    let node = p.get("id").and_then(Value::as_u64).map(|v| v as u32);
    edit_scene(s, id, "Move Object to Ground Plane", |sc, _| {
        let ids: Vec<u32> = match node {
            Some(n) => vec![n],
            None => sc.objects.iter().map(|o| o.id).collect(),
        };
        for i in &ids {
            sc.drop_to_ground(*i);
        }
        Ok(json!({"objects": ids}))
    })
}

fn light_preset(s: &mut Session, p: &Value, preset: &'static str) -> Result<Value> {
    let id = layer_3d(s, p)?;
    edit_scene(s, id, "3D Light Preset", |sc, _| {
        if scene::apply_light_preset(sc, preset) { Ok(json!({"preset": preset})) } else { Err(other("unknown light preset")) }
    })
}

fn render_style(s: &mut Session, p: &Value, style: Style) -> Result<Value> {
    let id = layer_3d(s, p)?;
    edit_scene(s, id, "3D Render Style", |sc, _| {
        sc.render.style = style;
        Ok(json!({"style": style.id()}))
    })
}

fn camera_view(s: &mut Session, p: &Value, view: &'static str) -> Result<Value> {
    let id = layer_3d(s, p)?;
    edit_scene(s, id, "3D Camera View", |sc, _| {
        if view == "default" {
            sc.camera = scene::Camera::default();
        } else if view == "fit" {
            let b = sc.bounds();
            sc.frame(b);
        } else {
            sc.camera.set_view(view);
        }
        Ok(json!({"view": view}))
    })
}

fn toggle_ground(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_3d(s, p)?;
    edit_scene(s, id, "Ground Plane Shadows", |sc, _| {
        let e = &mut sc.environment;
        e.ground_shadow_opacity = if e.ground_shadow_opacity > 0.0 { 0.0 } else { 0.6 };
        Ok(json!({"groundShadowOpacity": e.ground_shadow_opacity}))
    })
}

fn cross_section(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_3d(s, p)?;
    edit_scene(s, id, "Cross Section", |sc, _| {
        sc.render.cross_section = !sc.render.cross_section;
        Ok(json!({"crossSection": sc.render.cross_section}))
    })
}

/// `3d.setTexture`: map a layer's pixels or an image file onto a material (or the environment's
/// image-based light), or remove the map.
fn set_texture(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "3d.setTexture";
    let id = layer_3d(s, p)?;
    let map = p.get("map").and_then(Value::as_str).unwrap_or("diffuse").to_string();
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let image: Option<(String, Image)> = if let Some(src) = p.get("fromLayer").and_then(Value::as_u64) {
        let lid = LayerId(src);
        let name = doc.layer(lid).map_or_else(|| "Layer".to_string(), |l| l.name.clone());
        Some((name, layer_texture(&doc, lid)?.ok_or_else(|| other("that layer is empty"))?.1))
    } else if let Some(path) = p.get("path").and_then(Value::as_str) {
        let bytes = photocraft_format::read_file(std::path::Path::new(path)).map_err(|e| other(format!("can't read {path}: {e}")))?;
        Some((path.rsplit(['/', '\\']).next().unwrap_or(path).to_string(), decode_image(path, &bytes).ok_or_else(|| other(format!("{path} isn't an image PhotoCraft reads")))?))
    } else {
        None
    };
    set_texture_image(s, id, p, &map, image).map_err(|e| match e {
        EngineError::BadParams { msg, .. } => bad(CMD, msg),
        e => e,
    })
}

/// [`set_texture`] with an already decoded image (`None` removes the map).
pub fn set_texture_image(s: &mut Session, id: LayerId, p: &Value, map: &str, image: Option<(String, Image)>) -> Result<Value> {
    const CMD: &str = "3d.setTexture";
    let obj = p.get("id").and_then(Value::as_u64).map(|v| v as u32);
    let slot = p.get("slot").and_then(Value::as_u64).unwrap_or(0) as usize;
    let map = map.to_string();
    edit_scene(s, id, "3D Texture", move |sc, _| {
        let tex = image.map(|(n, img)| sc.add_texture(&n, img));
        if map == "environment" {
            sc.environment.ibl_texture = tex;
        } else {
            let oid = obj.ok_or_else(|| bad(CMD, "`id` (object id) is required"))?;
            let m = sc.object_mut(oid).and_then(|o| o.materials.get_mut(slot)).ok_or_else(|| bad(CMD, "no such object or material slot"))?;
            match map.as_str() {
                "diffuse" => m.diffuse_texture = tex,
                "bump" => m.bump_texture = tex,
                "opacity" => m.opacity_texture = tex,
                "normal" => m.normal_texture = tex,
                other => return Err(bad(CMD, format!("unknown map `{other}`: diffuse, bump, opacity, normal or environment"))),
            }
        }
        sc.prune_textures();
        Ok(json!({"texture": tex}))
    })
}

fn material_preset(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "3d.materialPreset";
    let id = layer_3d(s, p)?;
    let preset = p.get("preset").and_then(Value::as_str).ok_or_else(|| bad(CMD, format!("`preset` is required: {}", scene::MATERIAL_PRESETS.join(", "))))?.to_string();
    let obj = p.get("id").and_then(Value::as_u64).map(|v| v as u32);
    let slot = p.get("slot").and_then(Value::as_u64).map(|v| v as usize);
    edit_scene(s, id, "3D Material Preset", |sc, _| {
        let m = scene::material_preset(&preset).ok_or_else(|| bad(CMD, format!("unknown preset `{preset}`")))?;
        let oid = obj.or_else(|| sc.objects.first().map(|o| o.id)).ok_or_else(|| bad(CMD, "the scene has no objects"))?;
        let o = sc.object_mut(oid).ok_or_else(|| bad(CMD, format!("no object {oid}")))?;
        let slots: Vec<usize> = match slot {
            Some(s) => vec![s],
            None => (0..o.materials.len()).collect(),
        };
        for k in slots {
            if let Some(dst) = o.materials.get_mut(k) {
                let (name, tex) = (dst.name.clone(), dst.diffuse_texture);
                *dst = Material { name, diffuse_texture: if preset == "default" { tex } else { None }, ..m.clone() };
            }
        }
        Ok(json!({"preset": preset}))
    })
}

fn save_view(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_3d(s, p)?;
    let name = p.get("name").and_then(Value::as_str).unwrap_or("View").to_string();
    edit_scene(s, id, "Save 3D View", |sc, _| {
        sc.saved_views.retain(|(n, _)| *n != name);
        sc.saved_views.push((name.clone(), sc.camera.clone()));
        Ok(json!({"saved": name}))
    })
}

fn restore_view(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "3d.restoreView";
    let id = layer_3d(s, p)?;
    let name = p.get("name").and_then(Value::as_str).ok_or_else(|| bad(CMD, "`name` is required"))?.to_string();
    edit_scene(s, id, "Restore 3D View", |sc, _| {
        let cam = sc.saved_views.iter().find(|(n, _)| *n == name).map(|(_, c)| c.clone()).ok_or_else(|| bad(CMD, format!("no saved view `{name}`")))?;
        sc.camera = cam;
        Ok(json!({"restored": name}))
    })
}

fn inspect(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_3d(s, p)?;
    let sc = scene_of(s, Some(id)).ok_or_else(|| other("the layer isn't a 3D layer"))?;
    let mut full = serde_json::to_value(&*sc).map_err(|e| other(e.to_string()))?;
    // Texture pixels are large; list them instead.
    if let Some(t) = full.get_mut("textures") {
        *t = summary(&sc)["textures"].clone();
    }
    if let Some(objs) = full.get_mut("objects").and_then(Value::as_array_mut) {
        for o in objs {
            if let Some(g) = o.get_mut("geometry").and_then(Value::as_object_mut) {
                for k in ["mesh", "heights", "shapes"] {
                    if g.contains_key(k) {
                        g.insert(k.into(), json!("…"));
                    }
                }
            }
        }
    }
    Ok(json!({"layer": id.0, "summary": summary(&sc), "scene": full}))
}

// ---------------------------------------------------------------- registry

macro_rules! spec {
    ($id:literal, $label:literal, $menu:expr, $params:literal, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: $menu, shortcut: None, params: $params, enabled: $en, run: $run, journal: true }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    let mut v = vec![
        spec!("3d.newLayerFromFile", "New 3D Layer from File…", MENU, r#"{"path":str} (OBJ+MTL, STL, glTF, GLB; textures and MTL are read next to the model)"#, has_doc, new_layer_from_file),
        spec!("3d.mergeLayers", "Merge 3D Layers", MENU, "{} (the selected 3D layers)", two_3d, merge_layers),
        spec!("3d.exportLayer", "Export 3D Layer…", MENU, r#"{"layer":id?,"path":str} (.obj + .mtl + textures, .stl, .gltf or .glb)"#, has_3d, export_layer),
        spec!("3d.newExtrusionFromLayer", "New 3D Extrusion from Selected Layer", MENU, r#"{"layer":id?} (any layer kind: type, shapes, pixels)"#, has_layer, new_extrusion_from_layer),
        spec!("3d.newExtrusionFromSelection", "New 3D Extrusion from Selection", MENU, "{}", has_selection, new_extrusion_from_selection),
        spec!("3d.newExtrusionFromPath", "New 3D Extrusion from Current Path", MENU, "{} (the work path, else the first saved path)", has_path, new_extrusion_from_path),
        spec!("3d.newMeshFromLayer.postcard", "Postcard", MENU_MESH, r#"{"layer":id?}"#, has_layer, new_postcard),
        spec!("3d.newMeshFromLayer.depthMap.plane", "Plane", MENU_DEPTH, r#"{"layer":id?,"strength":num?}"#, has_layer, |s, p| new_depth_map(s, p, DepthMapKind::Plane)),
        spec!("3d.newMeshFromLayer.depthMap.twoSidedPlane", "Two-Sided Plane", MENU_DEPTH, r#"{"layer":id?,"strength":num?}"#, has_layer, |s, p| new_depth_map(s, p, DepthMapKind::TwoSidedPlane)),
        spec!("3d.newMeshFromLayer.depthMap.cylinder", "Cylinder", MENU_DEPTH, r#"{"layer":id?,"strength":num?}"#, has_layer, |s, p| new_depth_map(s, p, DepthMapKind::Cylinder)),
        spec!("3d.newMeshFromLayer.depthMap.sphere", "Sphere", MENU_DEPTH, r#"{"layer":id?,"strength":num?}"#, has_layer, |s, p| new_depth_map(s, p, DepthMapKind::Sphere)),
        spec!("3d.splitExtrusion", "Split Extrusion", MENU, r#"{"layer":id?,"object":id?}"#, has_3d, split_extrusion),
        spec!("3d.moveObjectToGroundPlane", "Move Object to Ground Plane", MENU, r#"{"layer":id?,"id":id? (default: every object)}"#, has_3d, move_to_ground),
        spec!("3d.crossSection", "Apply Cross Section to Scene", MENU, r#"{"layer":id?} (toggles; set axis/offset with 3d.set target render)"#, has_3d, cross_section),
        spec!("3d.toggleGroundShadows", "Ground Plane Shadows", MENU, r#"{"layer":id?}"#, has_3d, toggle_ground),
        CommandSpec {
            id: "3d.render",
            label: "Render 3D Layer",
            menu: MENU,
            shortcut: Some("Cmd+Alt+Shift+R"),
            params: r#"{"layer":id?,"samples":int?} (ray traced; a background job from the UI)"#,
            enabled: has_3d,
            run: render,
            journal: true,
        },
        spec!("3d.rasterize", "Rasterize 3D Layer", MENU, r#"{"layer":id?} (keeps the pixels, drops the scene)"#, has_3d, rasterize),
        // Panel-driven commands (no menu item of their own).
        spec!(
            "3d.set",
            "Set 3D Property",
            &[],
            r#"{"layer":id?,"target":"object"|"material"|"light"|"camera"|"environment"|"render","id":id? (object/light),"slot":int? (material),"values":{…}} (deep-merges values into the target)"#,
            has_3d,
            set
        ),
        spec!("3d.navigate", "3D Navigate", &[], r#"{"layer":id?,"mode":"orbit"|"roll"|"pan"|"slide"|"scale","dx":num,"dy":num,"object":id? (else the camera)}"#, has_3d, navigate),
        spec!("3d.addObject", "Add 3D Object", &[], r#"{"layer":id?,"preset":str}"#, has_3d, add_object),
        spec!("3d.deleteNode", "Delete 3D Object or Light", &[], r#"{"layer":id?,"id":id}"#, has_3d, delete_node),
        spec!("3d.duplicateObject", "Duplicate 3D Object", MENU, r#"{"layer":id?,"id":id?}"#, has_3d, duplicate_object),
        spec!("3d.setTexture", "Set 3D Texture", &[], r#"{"layer":id?,"id":id? (object),"slot":int?,"map":"diffuse"|"bump"|"opacity"|"normal"|"environment","fromLayer":id?|"path":str? (neither: remove)}"#, has_3d, set_texture),
        spec!("3d.materialPreset", "3D Material Preset", &[], r#"{"layer":id?,"id":id?,"slot":int? (default every slot),"preset":str}"#, has_3d, material_preset),
        spec!("3d.saveView", "Save 3D View", &[], r#"{"layer":id?,"name":str}"#, has_3d, save_view),
        spec!("3d.restoreView", "Restore 3D View", &[], r#"{"layer":id?,"name":str}"#, has_3d, restore_view),
        spec!("3d.inspect", "Inspect 3D Scene", &[], r#"{"layer":id?} → {"summary","scene"}"#, has_3d, inspect),
    ];
    // Journal only edits.
    if let Some(i) = v.iter_mut().find(|c| c.id == "3d.inspect") {
        i.journal = false;
    }
    macro_rules! preset {
        ($($id:literal => $p:expr),*) => { $( v.push(spec!($id, "", MENU_PRESET, r#"{"layer":id?}"#, has_doc, |s, p| new_mesh_preset(s, p, $p))); )* };
    }
    preset!(
        "3d.newMeshFromLayer.meshPreset.cone" => MeshPreset::Cone,
        "3d.newMeshFromLayer.meshPreset.cube" => MeshPreset::Cube,
        "3d.newMeshFromLayer.meshPreset.cubeWrap" => MeshPreset::CubeWrap,
        "3d.newMeshFromLayer.meshPreset.cylinder" => MeshPreset::Cylinder,
        "3d.newMeshFromLayer.meshPreset.donut" => MeshPreset::Donut,
        "3d.newMeshFromLayer.meshPreset.hat" => MeshPreset::Hat,
        "3d.newMeshFromLayer.meshPreset.pyramid" => MeshPreset::Pyramid,
        "3d.newMeshFromLayer.meshPreset.ring" => MeshPreset::Ring,
        "3d.newMeshFromLayer.meshPreset.sodaCan" => MeshPreset::SodaCan,
        "3d.newMeshFromLayer.meshPreset.sphere" => MeshPreset::Sphere,
        "3d.newMeshFromLayer.meshPreset.wineBottle" => MeshPreset::WineBottle
    );
    // Labels from the preset table (the macro can't build &'static labels from expressions).
    for c in v.iter_mut().filter(|c| c.id.starts_with("3d.newMeshFromLayer.meshPreset.")) {
        c.label = c.id.rsplit('.').next().and_then(MeshPreset::from_id).map_or("Mesh", MeshPreset::label);
    }
    macro_rules! add {
        ($($id:literal, $label:literal => $p:literal),*) => { $( v.push(spec!($id, $label, MENU_ADD, r#"{"layer":id?}"#, has_3d, |s, p| { let mut q = p.clone(); if let Some(m) = q.as_object_mut() { m.insert("preset".into(), json!($p)); } else { q = json!({"preset": $p}); } add_object(s, &q) })); )* };
    }
    add!(
        "3d.addObject.cube", "Cube" => "cube",
        "3d.addObject.sphere", "Sphere" => "sphere",
        "3d.addObject.cylinder", "Cylinder" => "cylinder",
        "3d.addObject.cone", "Cone" => "cone",
        "3d.addObject.donut", "Donut" => "donut",
        "3d.addObject.pyramid", "Pyramid" => "pyramid",
        "3d.addObject.plane", "Plane" => "plane"
    );
    v.push(spec!("3d.addLight.infinite", "Infinite Light", MENU_LIGHT, r#"{"layer":id?}"#, has_3d, |s, p| add_light(s, p, LightKind::Infinite)));
    v.push(spec!("3d.addLight.spot", "Spot Light", MENU_LIGHT, r#"{"layer":id?}"#, has_3d, |s, p| add_light(s, p, LightKind::Spot)));
    v.push(spec!("3d.addLight.point", "Point Light", MENU_LIGHT, r#"{"layer":id?}"#, has_3d, |s, p| add_light(s, p, LightKind::Point)));
    macro_rules! lp {
        ($($id:literal, $label:literal => $p:literal),*) => { $( v.push(spec!($id, $label, MENU_LIGHT_PRESET, r#"{"layer":id?}"#, has_3d, |s, p| light_preset(s, p, $p))); )* };
    }
    lp!(
        "3d.lightPreset.default", "Default Lights" => "default",
        "3d.lightPreset.studio", "Studio" => "studio",
        "3d.lightPreset.softbox", "Softbox" => "softbox",
        "3d.lightPreset.dramatic", "Dramatic" => "dramatic",
        "3d.lightPreset.productPhotography", "Product Photography" => "productPhotography",
        "3d.lightPreset.threePoint", "Three-Point Lighting" => "threePoint",
        "3d.lightPreset.sunset", "Sunset" => "sunset",
        "3d.lightPreset.neutral", "Neutral Environment" => "neutral",
        "3d.lightPreset.night", "Night" => "night"
    );
    macro_rules! st {
        ($($id:literal, $label:literal => $p:expr),*) => { $( v.push(spec!($id, $label, MENU_STYLE, r#"{"layer":id?}"#, has_3d, |s, p| render_style(s, p, $p))); )* };
    }
    st!(
        "3d.style.solid", "Default" => Style::Solid,
        "3d.style.unlitTexture", "Unlit Texture" => Style::Unlit,
        "3d.style.wireframe", "Wireframe" => Style::Wireframe,
        "3d.style.shadedWireframe", "Shaded Wireframe" => Style::ShadedWireframe,
        "3d.style.points", "Points" => Style::Points,
        "3d.style.normals", "Normals" => Style::Normals,
        "3d.style.depthMap", "Depth Map" => Style::DepthMap,
        "3d.style.lineIllustration", "Line Illustration" => Style::LineIllustration
    );
    macro_rules! cv {
        ($($id:literal, $label:literal => $p:literal),*) => { $( v.push(spec!($id, $label, MENU_VIEW, r#"{"layer":id?}"#, has_3d, |s, p| camera_view(s, p, $p))); )* };
    }
    cv!(
        "3d.view.default", "Default View" => "default",
        "3d.view.fit", "Fit to Scene" => "fit",
        "3d.view.front", "Front" => "front",
        "3d.view.back", "Back" => "back",
        "3d.view.left", "Left" => "left",
        "3d.view.right", "Right" => "right",
        "3d.view.top", "Top" => "top",
        "3d.view.bottom", "Bottom" => "bottom"
    );
    v
}

#[cfg(test)]
mod tests;

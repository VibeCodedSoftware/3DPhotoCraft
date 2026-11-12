//! Window › 3D: Photoshop's 3D panel and its Properties, in one floating panel.
//!
//! - With an ordinary layer active: the panel's "Create New 3D Object" section (extrusion from
//!   the layer, selection or path, postcard, mesh presets, depth maps, a model from a file).
//! - With a 3D layer active: an interactive view of the scene (drag to orbit, roll, pan, slide
//!   or scale the camera or the selected object, as the old 3D tools did), the scene tree
//!   (environment, camera, objects, lights) and the selected node's properties.
//!
//! Every change runs an engine `3d.*` command, so it is undoable and the control channel drives
//! the panel the same way. Slider and drag edits coalesce into one history step per gesture.

use std::sync::Arc;

use egui::{RichText, vec2};
use photocraft_three::extrude::{BevelContour, BevelSides};
use photocraft_three::scene::{Background, CrossAxis, Geometry, LightKind, Scene, Style};
use photocraft_three::{MeshPreset, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

/// What the panel has selected in the scene tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Node {
    #[default]
    Scene,
    Environment,
    Camera,
    Object(u32),
    Light(u32),
}

/// The drag tools of the old 3D toolbar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NavMode {
    #[default]
    Orbit,
    Roll,
    Pan,
    Slide,
    Scale,
}

impl NavMode {
    const ALL: [NavMode; 5] = [NavMode::Orbit, NavMode::Roll, NavMode::Pan, NavMode::Slide, NavMode::Scale];
    fn id(self) -> &'static str {
        match self {
            NavMode::Orbit => "orbit",
            NavMode::Roll => "roll",
            NavMode::Pan => "pan",
            NavMode::Slide => "slide",
            NavMode::Scale => "scale",
        }
    }
    fn label(self) -> &'static str {
        match self {
            NavMode::Orbit => "Orbit",
            NavMode::Roll => "Roll",
            NavMode::Pan => "Pan",
            NavMode::Slide => "Slide",
            NavMode::Scale => "Scale",
        }
    }
    fn tip(self) -> &'static str {
        match self {
            NavMode::Orbit => "3D Orbit: drag to turn around the scene (or the selected object)",
            NavMode::Roll => "3D Roll: drag sideways to roll around the view axis",
            NavMode::Pan => "3D Pan: drag to move in the view plane",
            NavMode::Slide => "3D Slide: drag up and down to move nearer or farther",
            NavMode::Scale => "3D Scale: drag up and down to zoom (camera) or resize (object)",
        }
    }
}

/// Cached render of the panel's scene view.
#[derive(Clone, Default)]
pub struct ViewCache {
    key: Option<(usize, u32, u32)>,
    texture: Option<egui::TextureHandle>,
    swatch_key: Option<(usize, u32, usize)>,
    swatch: Option<egui::TextureHandle>,
}

impl std::fmt::Debug for ViewCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ViewCache")
    }
}

impl PartialEq for ViewCache {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// 3D panel view state.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThreeDUi {
    pub open: bool,
    #[serde(default)]
    pub node: Node,
    #[serde(default)]
    pub mode: NavMode,
    /// Drags move the selected object instead of the camera.
    #[serde(default)]
    pub move_object: bool,
    #[serde(default)]
    pub slot: usize,
    /// The 3D tools stay in the panel's view only (the canvas keeps the normal tools).
    #[serde(default)]
    pub canvas_off: bool,
    /// Bumped after each gesture, so edits coalesce per drag.
    #[serde(skip)]
    pub gesture: u64,
    #[serde(skip)]
    pub cache: ViewCache,
}

pub fn checked(app: &PhotocraftApp, id: &str) -> Option<bool> {
    (id == "window.panel.3d").then_some(app.ui.three_d.open)
}

pub fn handles(id: &str) -> bool {
    id == "window.panel.3d"
}

pub fn menu(app: &mut PhotocraftApp, id: &str, _params: &Value) -> Option<Result<Value, String>> {
    if id == "window.panel.3d" {
        app.ui.three_d.open = !app.ui.three_d.open;
        return Some(Ok(json!({ "open": app.ui.three_d.open })));
    }
    // Making a 3D layer from a menu shows the panel, as Photoshop switched to its 3D workspace.
    // The command itself still runs (None).
    if id.starts_with("3d.new") {
        app.ui.three_d.open = true;
        app.ui.three_d.node = Node::Scene;
    }
    None
}

/// Do the 3D tools own drags on the canvas? (The panel is open, on, and a 3D layer is active.)
pub fn canvas_owns(app: &PhotocraftApp) -> bool {
    app.ui.three_d.open && !app.ui.three_d.canvas_off && active_scene(app).is_some()
}

/// Canvas drags for the 3D tools. `zoom` is screen points per document pixel.
pub fn canvas_input(app: &mut PhotocraftApp, ctx: &egui::Context, zoom: f32, response: &egui::Response) {
    let st = &app.ui.three_d;
    if response.hovered() {
        ctx.set_cursor_icon(match st.mode {
            NavMode::Pan => egui::CursorIcon::Grab,
            NavMode::Slide | NavMode::Scale => egui::CursorIcon::ResizeVertical,
            NavMode::Roll => egui::CursorIcon::ResizeHorizontal,
            NavMode::Orbit => egui::CursorIcon::AllScroll,
        });
    }
    let zoom = if zoom.is_finite() && zoom > 1e-6 { zoom } else { 1.0 };
    let d = response.drag_delta() / zoom;
    let object = match (st.move_object, st.node) {
        (true, Node::Object(id)) => Some(id),
        _ => None,
    };
    let key = format!("3d-canvas-{}", st.gesture);
    let mode = st.mode.id();
    if d != egui::Vec2::ZERO {
        let mut p = json!({"mode": mode, "dx": d.x, "dy": d.y, "coalesce": key});
        if let Some(o) = object {
            p["object"] = json!(o);
        }
        if let Err(e) = app.run("3d.navigate", p) {
            app.ui.status = e;
        }
    }
    if response.drag_stopped() {
        app.ui.three_d.gesture = app.ui.three_d.gesture.wrapping_add(1);
    }
    if response.hovered() {
        let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.5 && !ctx.input(|i| i.modifiers.command || i.modifiers.alt) {
            let _ = app.run("3d.navigate", json!({"mode": "slide", "dx": 0, "dy": -scroll * 0.6, "coalesce": "3d-wheel"}));
        }
    }
}

fn active_scene(app: &PhotocraftApp) -> Option<Arc<Scene>> {
    let st = app.session.active()?;
    let l = st.doc.layer(st.active_layer?)?;
    if l.is_3d() { l.scene3d.clone() } else { None }
}

enum Act {
    Run(&'static str, Value),
    /// Run through the menu path (file pickers, dialogs).
    Menu(&'static str),
    Close,
    LoadTexture { id: u32, slot: usize, map: &'static str },
    LayerTexture { id: u32, slot: usize, map: &'static str, layer: u64 },
}

pub fn windows(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if !app.ui.three_d.open {
        return;
    }
    let scene = active_scene(app);
    let have_doc = app.session.active().is_some();
    let doc_size = app.session.active().map(|d| (d.doc.size.width, d.doc.size.height)).unwrap_or((1, 1));
    let layers: Vec<(u64, String)> = app.session.active().map(|d| d.doc.walk().into_iter().map(|(_, _, l)| (l.id.0, l.name.clone())).collect()).unwrap_or_default();
    let t = Tokens::get(ctx);
    let mut acts: Vec<Act> = Vec::new();
    let mut ui_state = app.ui.three_d.clone();
    // Scene view texture.
    let view_w = 300u32;
    let view_h = ((view_w as f32 * doc_size.1 as f32 / doc_size.0.max(1) as f32).round() as u32).clamp(80, 340);
    if let Some(sc) = &scene {
        let key = (Arc::as_ptr(sc) as usize, view_w, view_h);
        if ui_state.cache.key != Some(key) {
            let mut preview = (**sc).clone();
            // The panel always shows the ground and a neutral backdrop.
            if preview.environment.background == Background::Transparent {
                preview.environment.background = Background::Color;
                preview.environment.background_color = [0.16, 0.16, 0.17];
            }
            let out = photocraft_three::render_preview(&preview, view_w, view_h, 1);
            let img = egui::ColorImage::from_rgba_unmultiplied([view_w as usize, view_h as usize], &out.to_rgba8());
            ui_state.cache.texture = Some(ctx.load_texture("3d-view", img, egui::TextureOptions::LINEAR));
            ui_state.cache.key = Some(key);
        }
    }
    let gesture_key = format!("3d-gesture-{}", ui_state.gesture);
    // Docked against the canvas's right edge, like Photoshop's 3D panel next to Properties.
    crate::analysis_ui::panel_window(app, ctx, "3d", tl!("3D"), vec2(0.0, 8.0), 330.0, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(tl!("3D")).strong().color(t.text));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if crate::icons::button(ui, "x", 20.0, false, tl!("Close")).clicked() {
                    acts.push(Act::Close);
                }
            });
        });
        ui.separator();
        if !have_doc {
            ui.weak(tl!("Open or create a document to work in 3D."));
            return;
        }
        let Some(sc) = scene.as_deref() else {
            create_section(ui, &mut acts);
            return;
        };
        let tall = (ctx.content_rect().height() - 160.0).clamp(300.0, 900.0);
        egui::ScrollArea::vertical().min_scrolled_height(tall).max_height(tall).show(ui, |ui| {
            // Navigation tools.
            ui.horizontal_wrapped(|ui| {
                for m in NavMode::ALL {
                    if ui.selectable_label(ui_state.mode == m, m.label()).on_hover_text(m.tip()).clicked() {
                        ui_state.mode = m;
                    }
                }
            });
            let can_move_object = matches!(ui_state.node, Node::Object(_));
            ui.horizontal(|ui| {
                ui.label(tl!("Drag moves:"));
                ui.selectable_value(&mut ui_state.move_object, false, tl!("Camera"));
                ui.add_enabled_ui(can_move_object, |ui| {
                    ui.selectable_value(&mut ui_state.move_object, true, tl!("Selected Object"));
                });
            });
            let mut on_canvas = !ui_state.canvas_off;
            if ui.checkbox(&mut on_canvas, tl!("Use 3D tools on the canvas")).on_hover_text(tl!("Drag on the image to navigate; turn off to paint or select on the 3D layer")).changed() {
                ui_state.canvas_off = !on_canvas;
            }
            // Scene view.
            if let Some(tex) = &ui_state.cache.texture {
                let size = vec2(view_w as f32, view_h as f32);
                let r = ui.add(egui::Image::new((tex.id(), size)).sense(egui::Sense::click_and_drag())).on_hover_text(tl!("Drag to navigate; the scroll wheel slides the camera"));
                let d = r.drag_delta();
                let k = doc_size.1 as f32 / view_h as f32;
                let object = match (ui_state.move_object, ui_state.node) {
                    (true, Node::Object(id)) => Some(id),
                    _ => None,
                };
                if d != egui::Vec2::ZERO {
                    let mut p = json!({"mode": ui_state.mode.id(), "dx": d.x * k, "dy": d.y * k, "coalesce": gesture_key});
                    if let Some(o) = object {
                        p["object"] = json!(o);
                    }
                    acts.push(Act::Run("3d.navigate", p));
                }
                if r.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.5 {
                        acts.push(Act::Run("3d.navigate", json!({"mode": "slide", "dx": 0, "dy": -scroll * 0.6, "coalesce": "3d-wheel"})));
                    }
                }
            }
            ui.horizontal_wrapped(|ui| {
                for (id, label) in [("3d.view.default", "Reset View"), ("3d.view.fit", "Fit"), ("3d.view.front", "Front"), ("3d.view.top", "Top"), ("3d.view.left", "Left"), ("3d.view.right", "Right")] {
                    if ui.small_button(label).clicked() {
                        acts.push(Act::Run(id, json!({})));
                    }
                }
            });
            ui.horizontal(|ui| {
                if crate::widgets::primary_button(ui, tl!("Render"), 90.0).on_hover_text(tl!("3D › Render 3D Layer: ray traced")).clicked() {
                    acts.push(Act::Run("3d.render", json!({})));
                }
                if ui.button(tl!("Export…")).clicked() {
                    acts.push(Act::Menu("3d.exportLayer"));
                }
                if ui.button(tl!("Rasterize")).clicked() {
                    acts.push(Act::Run("3d.rasterize", json!({})));
                }
            });
            ui.label(
                RichText::new(format!(
                    "{} objects · {} lights · {} triangles{}",
                    sc.objects.len(),
                    sc.lights.len(),
                    sc.triangle_count(),
                    if sc.state == photocraft_three::scene::RenderState::RayTraced { " · ray traced" } else { " · preview" }
                ))
                .color(t.text_dim)
                .small(),
            );
            ui.separator();
            tree(ui, sc, &mut ui_state, &mut acts);
            ui.separator();
            properties(ui, sc, &mut ui_state, &mut acts, &gesture_key, &layers, &t);
        });
    });
    // A gesture ends when no button is held.
    if !ctx.input(|i| i.pointer.any_down()) {
        ui_state.gesture = ui_state.gesture.wrapping_add(1);
    }
    app.ui.three_d = ui_state;
    for a in acts {
        let r = match a {
            Act::Run(id, p) => app.run(id, p),
            Act::Menu(id) => crate::menus::invoke(app, ctx, id, json!({})),
            Act::Close => {
                app.ui.three_d.open = false;
                Ok(Value::Null)
            }
            Act::LoadTexture { id, slot, map } => {
                let doc = app.session.active().map(|d| d.doc.id);
                let layer = app.session.active().and_then(|d| d.active_layer);
                match (doc, layer) {
                    (Some(doc), Some(layer)) => app.pick_file_bytes(move |app, name, bytes| {
                        app.with_document(doc, |app| {
                            let img = decode(&name, &bytes).ok_or_else(|| format!("{name} isn't an image PhotoCraft reads"))?;
                            let r = photocraft_engine::three_d_cmds::set_texture_image(&mut app.session, layer, &json!({"id": id, "slot": slot}), map, Some((name.clone(), img))).map_err(|e| e.to_string());
                            app.sync_views();
                            r
                        })
                    }),
                    _ => Err("no 3D layer".into()),
                }
            }
            Act::LayerTexture { id, slot, map, layer } => app.run("3d.setTexture", json!({"id": id, "slot": slot, "map": map, "fromLayer": layer})),
        };
        if let Err(e) = r {
            app.ui.status = e;
        }
    }
}

fn decode(name: &str, bytes: &[u8]) -> Option<photocraft_three::Image> {
    photocraft_engine::three_d_cmds::decode_image(name, bytes)
}

fn create_section(ui: &mut egui::Ui, acts: &mut Vec<Act>) {
    ui.label(RichText::new(tl!("Create New 3D Object")).strong());
    ui.weak(tl!("The active layer isn't a 3D layer. Make one from it:"));
    ui.add_space(4.0);
    let w = 290.0;
    if crate::widgets::primary_button(ui, tl!("3D Extrusion from Selected Layer"), w).clicked() {
        acts.push(Act::Run("3d.newExtrusionFromLayer", json!({})));
    }
    if ui.add_sized([w, 22.0], egui::Button::new(tl!("3D Extrusion from Selection"))).clicked() {
        acts.push(Act::Run("3d.newExtrusionFromSelection", json!({})));
    }
    if ui.add_sized([w, 22.0], egui::Button::new(tl!("3D Extrusion from Current Path"))).clicked() {
        acts.push(Act::Run("3d.newExtrusionFromPath", json!({})));
    }
    if ui.add_sized([w, 22.0], egui::Button::new(tl!("Postcard"))).clicked() {
        acts.push(Act::Run("3d.newMeshFromLayer.postcard", json!({})));
    }
    if ui.add_sized([w, 22.0], egui::Button::new(tl!("New 3D Layer from File… (OBJ, STL, glTF, GLB)"))).clicked() {
        acts.push(Act::Menu("3d.newLayerFromFile"));
    }
    ui.add_space(4.0);
    ui.label(tl!("Mesh from Preset"));
    ui.horizontal_wrapped(|ui| {
        for p in MeshPreset::ALL.iter().filter(|p| !matches!(p, MeshPreset::Plane | MeshPreset::TwoSidedPlane)) {
            if ui.small_button(p.label()).clicked() {
                acts.push(Act::Run(preset_cmd(*p), json!({})));
            }
        }
    });
    ui.label(tl!("Depth Map to"));
    ui.horizontal_wrapped(|ui| {
        for (id, label) in [
            ("3d.newMeshFromLayer.depthMap.plane", "Plane"),
            ("3d.newMeshFromLayer.depthMap.twoSidedPlane", "Two-Sided Plane"),
            ("3d.newMeshFromLayer.depthMap.cylinder", "Cylinder"),
            ("3d.newMeshFromLayer.depthMap.sphere", "Sphere"),
        ] {
            if ui.small_button(label).clicked() {
                acts.push(Act::Run(id, json!({})));
            }
        }
    });
}

fn preset_cmd(p: MeshPreset) -> &'static str {
    match p {
        MeshPreset::Cone => "3d.newMeshFromLayer.meshPreset.cone",
        MeshPreset::Cube => "3d.newMeshFromLayer.meshPreset.cube",
        MeshPreset::CubeWrap => "3d.newMeshFromLayer.meshPreset.cubeWrap",
        MeshPreset::Cylinder => "3d.newMeshFromLayer.meshPreset.cylinder",
        MeshPreset::Donut => "3d.newMeshFromLayer.meshPreset.donut",
        MeshPreset::Hat => "3d.newMeshFromLayer.meshPreset.hat",
        MeshPreset::Pyramid => "3d.newMeshFromLayer.meshPreset.pyramid",
        MeshPreset::Ring => "3d.newMeshFromLayer.meshPreset.ring",
        MeshPreset::SodaCan => "3d.newMeshFromLayer.meshPreset.sodaCan",
        MeshPreset::Sphere => "3d.newMeshFromLayer.meshPreset.sphere",
        MeshPreset::WineBottle => "3d.newMeshFromLayer.meshPreset.wineBottle",
        MeshPreset::Plane | MeshPreset::TwoSidedPlane => "3d.newMeshFromLayer.postcard",
    }
}

fn tree(ui: &mut egui::Ui, sc: &Scene, st: &mut ThreeDUi, acts: &mut Vec<Act>) {
    // A node that no longer exists falls back to the scene.
    match st.node {
        Node::Object(id) if sc.object(id).is_none() => st.node = Node::Scene,
        Node::Light(id) if !sc.lights.iter().any(|l| l.id == id) => st.node = Node::Scene,
        _ => {}
    }
    for (n, label) in [(Node::Environment, "Environment"), (Node::Scene, "Scene"), (Node::Camera, "Current View")] {
        if ui.selectable_label(st.node == n, label).clicked() {
            st.node = n;
        }
    }
    for o in &sc.objects {
        ui.horizontal(|ui| {
            let mut vis = o.visible;
            if ui.checkbox(&mut vis, "").on_hover_text(tl!("Show or hide the object")).changed() {
                acts.push(Act::Run("3d.set", json!({"target": "object", "id": o.id, "values": {"visible": vis}})));
            }
            if ui.selectable_label(st.node == Node::Object(o.id), format!("▣ {}", o.name)).clicked() {
                st.node = Node::Object(o.id);
                st.slot = 0;
            }
        });
    }
    for l in &sc.lights {
        ui.horizontal(|ui| {
            let mut vis = l.visible;
            if ui.checkbox(&mut vis, "").on_hover_text(tl!("Turn the light on or off")).changed() {
                acts.push(Act::Run("3d.set", json!({"target": "light", "id": l.id, "values": {"visible": vis}})));
            }
            let icon = match l.kind {
                LightKind::Infinite => "☀",
                LightKind::Spot => "◭",
                LightKind::Point => "✱",
            };
            if ui.selectable_label(st.node == Node::Light(l.id), format!("{icon} {}", l.name)).clicked() {
                st.node = Node::Light(l.id);
            }
        });
    }
    ui.horizontal_wrapped(|ui| {
        ui.menu_button(tl!("+ Object"), |ui| {
            for (id, label) in [
                ("3d.addObject.cube", "Cube"),
                ("3d.addObject.sphere", "Sphere"),
                ("3d.addObject.cylinder", "Cylinder"),
                ("3d.addObject.cone", "Cone"),
                ("3d.addObject.donut", "Donut"),
                ("3d.addObject.pyramid", "Pyramid"),
                ("3d.addObject.plane", "Plane"),
            ] {
                if ui.button(label).clicked() {
                    acts.push(Act::Run(id, json!({})));
                    ui.close();
                }
            }
        });
        ui.menu_button(tl!("+ Light"), |ui| {
            for (id, label) in [("3d.addLight.infinite", "Infinite Light"), ("3d.addLight.spot", "Spot Light"), ("3d.addLight.point", "Point Light")] {
                if ui.button(label).clicked() {
                    acts.push(Act::Run(id, json!({})));
                    ui.close();
                }
            }
        });
        let node_id = match st.node {
            Node::Object(id) | Node::Light(id) => Some(id),
            _ => None,
        };
        if ui.add_enabled(matches!(st.node, Node::Object(_)), egui::Button::new(tl!("Duplicate"))).clicked()
            && let Some(id) = node_id
        {
            acts.push(Act::Run("3d.duplicateObject", json!({"id": id})));
        }
        if ui.add_enabled(node_id.is_some(), egui::Button::new(tl!("Delete"))).clicked()
            && let Some(id) = node_id
        {
            acts.push(Act::Run("3d.deleteNode", json!({"id": id})));
            st.node = Node::Scene;
        }
    });
}

/// A numeric field that sends `3d.set` with `values` built from the new number.
fn num(ui: &mut egui::Ui, label: &str, v: f32, range: std::ops::RangeInclusive<f32>, speed: f32, mut make: impl FnMut(f32) -> Value, target: Value, acts: &mut Vec<Act>, key: &str) {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut x = v;
        if ui.add(egui::DragValue::new(&mut x).range(range).speed(speed).max_decimals(3)).changed() {
            let mut p = target;
            p["values"] = make(x);
            p["coalesce"] = json!(format!("{key}-{label}"));
            acts.push(Act::Run("3d.set", p));
        }
    });
}

fn slider(ui: &mut egui::Ui, label: &str, v: f32, range: std::ops::RangeInclusive<f32>, mut make: impl FnMut(f32) -> Value, target: Value, acts: &mut Vec<Act>, key: &str) {
    let mut x = v;
    if ui.add(egui::Slider::new(&mut x, range).text(label)).changed() {
        let mut p = target;
        p["values"] = make(x);
        p["coalesce"] = json!(format!("{key}-{label}"));
        acts.push(Act::Run("3d.set", p));
    }
}

fn color(ui: &mut egui::Ui, label: &str, c: [f32; 3], mut make: impl FnMut([f32; 3]) -> Value, target: Value, acts: &mut Vec<Act>, key: &str) {
    ui.horizontal(|ui| {
        let mut x = c;
        if ui.color_edit_button_rgb(&mut x).changed() {
            let mut p = target;
            p["values"] = make(x);
            p["coalesce"] = json!(format!("{key}-{label}"));
            acts.push(Act::Run("3d.set", p));
        }
        ui.label(label);
    });
}

fn check(ui: &mut egui::Ui, label: &str, v: bool, mut make: impl FnMut(bool) -> Value, target: Value, acts: &mut Vec<Act>) {
    let mut x = v;
    if ui.checkbox(&mut x, label).changed() {
        let mut p = target;
        p["values"] = make(x);
        acts.push(Act::Run("3d.set", p));
    }
}

fn vec3(ui: &mut egui::Ui, label: &str, v: Vec3, speed: f32, field: &'static str, sub: &'static str, target: &Value, acts: &mut Vec<Act>, key: &str) {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut a = [v.x, v.y, v.z];
        let mut changed = false;
        for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
            changed |= ui.add(egui::DragValue::new(&mut a[i]).speed(speed).prefix(format!("{axis} ")).max_decimals(2)).changed();
        }
        if changed {
            let mut p = target.clone();
            p["values"] = json!({field: {sub: {"x": a[0], "y": a[1], "z": a[2]}}});
            if field.is_empty() {
                p["values"] = json!({sub: {"x": a[0], "y": a[1], "z": a[2]}});
            }
            p["coalesce"] = json!(format!("{key}-{label}"));
            acts.push(Act::Run("3d.set", p));
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn properties(ui: &mut egui::Ui, sc: &Scene, st: &mut ThreeDUi, acts: &mut Vec<Act>, key: &str, layers: &[(u64, String)], t: &Tokens) {
    ui.label(RichText::new(tl!("Properties")).strong().color(t.text));
    match st.node {
        Node::Scene => {
            let r = &sc.render;
            let tg = json!({"target": "render"});
            egui::ComboBox::from_label(tl!("Style")).selected_text(r.style.label()).show_ui(ui, |ui| {
                for s in Style::ALL {
                    if ui.selectable_label(r.style == s, s.label()).clicked() {
                        acts.push(Act::Run("3d.set", json!({"target": "render", "values": {"style": s}})));
                    }
                }
            });
            color(ui, "Line Color", r.line_color, |c| json!({"lineColor": c}), tg.clone(), acts, key);
            slider(ui, "Exposure", r.exposure, -4.0..=4.0, |v| json!({"exposure": v}), tg.clone(), acts, key);
            num(ui, "Preview Anti-alias", r.preview_aa as f32, 1.0..=3.0, 0.05, |v| json!({"previewAa": v.round() as u32}), tg.clone(), acts, key);
            ui.label(RichText::new(tl!("Render (ray traced)")).color(t.text_dim));
            num(ui, "Samples per pixel", r.samples as f32, 1.0..=256.0, 0.5, |v| json!({"samples": v.round() as u32}), tg.clone(), acts, key);
            num(ui, "Reflection bounces", r.bounces as f32, 0.0..=8.0, 0.05, |v| json!({"bounces": v.round() as u32}), tg.clone(), acts, key);
            ui.separator();
            check(ui, "Cross Section", r.cross_section, |v| json!({"crossSection": v}), tg.clone(), acts);
            if r.cross_section {
                ui.horizontal(|ui| {
                    for (a, l) in [(CrossAxis::X, "X"), (CrossAxis::Y, "Y"), (CrossAxis::Z, "Z")] {
                        if ui.selectable_label(r.cross_axis == a, l).clicked() {
                            acts.push(Act::Run("3d.set", json!({"target": "render", "values": {"crossAxis": a}})));
                        }
                    }
                });
                num(ui, "Offset", r.cross_offset, -100.0..=100.0, 0.01, |v| json!({"crossOffset": v}), tg.clone(), acts, key);
                check(ui, "Flip", r.cross_flip, |v| json!({"crossFlip": v}), tg, acts);
            }
            ui.separator();
            ui.label(tl!("Light preset"));
            ui.horizontal_wrapped(|ui| {
                for (id, label) in [
                    ("3d.lightPreset.default", "Default"),
                    ("3d.lightPreset.studio", "Studio"),
                    ("3d.lightPreset.softbox", "Softbox"),
                    ("3d.lightPreset.dramatic", "Dramatic"),
                    ("3d.lightPreset.productPhotography", "Product"),
                    ("3d.lightPreset.threePoint", "Three-Point"),
                    ("3d.lightPreset.sunset", "Sunset"),
                    ("3d.lightPreset.neutral", "Neutral"),
                    ("3d.lightPreset.night", "Night"),
                ] {
                    if ui.small_button(label).clicked() {
                        acts.push(Act::Run(id, json!({})));
                    }
                }
            });
        }
        Node::Environment => {
            let e = &sc.environment;
            let tg = json!({"target": "environment"});
            color(ui, "Global Ambient", e.ambient, |c| json!({"ambient": c}), tg.clone(), acts, key);
            ui.label(RichText::new(tl!("Image Based Light")).color(t.text_dim));
            color(ui, "Sky", e.sky, |c| json!({"sky": c}), tg.clone(), acts, key);
            color(ui, "Ground", e.ground, |c| json!({"ground": c}), tg.clone(), acts, key);
            slider(ui, "Intensity", e.ibl_intensity, 0.0..=3.0, |v| json!({"iblIntensity": v}), tg.clone(), acts, key);
            slider(ui, "Rotation", e.ibl_rotation, -180.0..=180.0, |v| json!({"iblRotation": v}), tg.clone(), acts, key);
            ui.horizontal(|ui| {
                ui.label(tl!("IBL image:"));
                ui.label(if e.ibl_texture.is_some() { "set" } else { "none" });
                if ui.small_button(tl!("Load…")).clicked() {
                    acts.push(Act::LoadTexture { id: 0, slot: 0, map: "environment" });
                }
                if e.ibl_texture.is_some() && ui.small_button(tl!("Remove")).clicked() {
                    acts.push(Act::Run("3d.setTexture", json!({"map": "environment"})));
                }
            });
            ui.label(RichText::new(tl!("Ground Plane")).color(t.text_dim));
            slider(ui, "Shadows", e.ground_shadow_opacity, 0.0..=1.0, |v| json!({"groundShadowOpacity": v}), tg.clone(), acts, key);
            slider(ui, "Reflections", e.ground_reflection_opacity, 0.0..=1.0, |v| json!({"groundReflectionOpacity": v}), tg.clone(), acts, key);
            num(ui, "Height", e.ground_y, -100.0..=100.0, 0.01, |v| json!({"groundY": v}), tg.clone(), acts, key);
            if ui.button(tl!("Move Objects to Ground Plane")).clicked() {
                acts.push(Act::Run("3d.moveObjectToGroundPlane", json!({})));
            }
            ui.label(RichText::new(tl!("Background")).color(t.text_dim));
            ui.horizontal(|ui| {
                for (b, l) in [(Background::Transparent, "Transparent"), (Background::Color, "Color"), (Background::Environment, "Environment")] {
                    if ui.selectable_label(e.background == b, l).clicked() {
                        acts.push(Act::Run("3d.set", json!({"target": "environment", "values": {"background": b}})));
                    }
                }
            });
            if e.background == Background::Color {
                color(ui, "Background Color", e.background_color, |c| json!({"backgroundColor": c}), tg, acts, key);
            }
        }
        Node::Camera => {
            let c = &sc.camera;
            let tg = json!({"target": "camera"});
            check(ui, "Orthographic", c.orthographic, |v| json!({"orthographic": v}), tg.clone(), acts);
            if c.orthographic {
                num(ui, "Zoom (visible height)", c.ortho_height, 0.01..=1000.0, 0.01, |v| json!({"orthoHeight": v}), tg.clone(), acts, key);
            } else {
                slider(ui, "Field of View", c.fov, 5.0..=120.0, |v| json!({"fov": v}), tg.clone(), acts, key);
            }
            num(ui, "Distance", c.distance, 0.01..=1000.0, 0.01, |v| json!({"distance": v}), tg.clone(), acts, key);
            num(ui, "Orbit (yaw)", c.yaw, -3600.0..=3600.0, 0.5, |v| json!({"yaw": v}), tg.clone(), acts, key);
            num(ui, "Orbit (pitch)", c.pitch, -89.9..=89.9, 0.5, |v| json!({"pitch": v}), tg.clone(), acts, key);
            num(ui, "Roll", c.roll, -3600.0..=3600.0, 0.5, |v| json!({"roll": v}), tg.clone(), acts, key);
            vec3(ui, "Target", c.target, 0.01, "", "target", &tg, acts, key);
            ui.label(RichText::new(tl!("Depth of Field (ray traced)")).color(t.text_dim));
            slider(ui, "Blur", c.dof_blur, 0.0..=1.0, |v| json!({"dofBlur": v}), tg.clone(), acts, key);
            num(ui, "Focus distance", c.dof_distance, 0.01..=1000.0, 0.01, |v| json!({"dofDistance": v}), tg, acts, key);
            ui.horizontal(|ui| {
                if ui.button(tl!("Save View")).clicked() {
                    let n = format!("View {}", sc.saved_views.len() + 1);
                    acts.push(Act::Run("3d.saveView", json!({"name": n})));
                }
                for (n, _) in &sc.saved_views {
                    if ui.small_button(n).on_hover_text(tl!("Restore this view")).clicked() {
                        acts.push(Act::Run("3d.restoreView", json!({"name": n})));
                    }
                }
            });
        }
        Node::Light(id) => {
            let Some(l) = sc.lights.iter().find(|l| l.id == id) else { return };
            let tg = json!({"target": "light", "id": id});
            ui.horizontal(|ui| {
                ui.label(tl!("Type"));
                for k in [LightKind::Infinite, LightKind::Spot, LightKind::Point] {
                    let label = match k {
                        LightKind::Infinite => "Infinite",
                        LightKind::Spot => "Spot",
                        LightKind::Point => "Point",
                    };
                    if ui.selectable_label(l.kind == k, label).clicked() {
                        acts.push(Act::Run("3d.set", json!({"target": "light", "id": id, "values": {"kind": k}})));
                    }
                }
            });
            color(ui, "Color", l.color, |c| json!({"color": c}), tg.clone(), acts, key);
            slider(ui, "Intensity", l.intensity, 0.0..=5.0, |v| json!({"intensity": v}), tg.clone(), acts, key);
            check(ui, "Shadow", l.shadow, |v| json!({"shadow": v}), tg.clone(), acts);
            slider(ui, "Softness", l.softness, 0.0..=1.0, |v| json!({"softness": v}), tg.clone(), acts, key);
            if l.kind != LightKind::Infinite {
                vec3(ui, "Position", l.position, 0.02, "", "position", &tg, acts, key);
            }
            if l.kind != LightKind::Point {
                num(ui, "Direction (yaw)", l.yaw, -3600.0..=3600.0, 0.5, |v| json!({"yaw": v}), tg.clone(), acts, key);
                num(ui, "Direction (pitch)", l.pitch, -90.0..=90.0, 0.5, |v| json!({"pitch": v}), tg.clone(), acts, key);
            }
            if l.kind == LightKind::Spot {
                slider(ui, "Hotspot", l.hotspot, 0.0..=170.0, |v| json!({"hotspot": v}), tg.clone(), acts, key);
                slider(ui, "Cone", l.cone, 1.0..=170.0, |v| json!({"cone": v}), tg.clone(), acts, key);
            }
            if l.kind != LightKind::Infinite {
                check(ui, "Light Falloff", l.falloff, |v| json!({"falloff": v}), tg.clone(), acts);
                if l.falloff {
                    num(ui, "Inner", l.inner_radius, 0.0..=1000.0, 0.05, |v| json!({"innerRadius": v}), tg.clone(), acts, key);
                    num(ui, "Outer", l.outer_radius, 0.0..=1000.0, 0.05, |v| json!({"outerRadius": v}), tg, acts, key);
                }
            }
        }
        Node::Object(id) => {
            let Some(o) = sc.object(id) else { return };
            let tg = json!({"target": "object", "id": id});
            ui.horizontal(|ui| {
                ui.label(tl!("Name"));
                let mut name = o.name.clone();
                if ui.text_edit_singleline(&mut name).lost_focus() && name != o.name && !name.trim().is_empty() {
                    acts.push(Act::Run("3d.set", json!({"target": "object", "id": id, "values": {"name": name}})));
                }
            });
            ui.label(RichText::new(format!("{} · {} triangles", o.geometry.kind_name(), o.mesh().triangle_count())).color(t.text_dim).small());
            ui.label(RichText::new(tl!("Coordinates")).color(t.text_dim));
            vec3(ui, "Position", o.transform.position, 0.01, "transform", "position", &tg, acts, key);
            vec3(ui, "Rotation", o.transform.rotation, 0.5, "transform", "rotation", &tg, acts, key);
            vec3(ui, "Scale", o.transform.scale, 0.01, "transform", "scale", &tg, acts, key);
            ui.horizontal(|ui| {
                if ui.small_button(tl!("Reset")).on_hover_text(tl!("Reset position, rotation and scale")).clicked() {
                    acts.push(Act::Run("3d.set", json!({"target": "object", "id": id, "values": {"transform": {"position": {"x": 0, "y": 0, "z": 0}, "rotation": {"x": 0, "y": 0, "z": 0}, "scale": {"x": 1, "y": 1, "z": 1}}}})));
                }
                if ui.small_button(tl!("Drop to Ground")).clicked() {
                    acts.push(Act::Run("3d.moveObjectToGroundPlane", json!({"id": id})));
                }
            });
            ui.horizontal(|ui| {
                check(ui, "Cast Shadows", o.cast_shadows, |v| json!({"castShadows": v}), tg.clone(), acts);
                check(ui, "Catch Shadows", o.catch_shadows, |v| json!({"catchShadows": v}), tg.clone(), acts);
                check(ui, "Invisible", o.invisible, |v| json!({"invisible": v}), tg.clone(), acts);
            });
            if let Geometry::Extrusion { params, shapes, .. } = &o.geometry {
                ui.label(RichText::new(tl!("Deform and Cap")).color(t.text_dim));
                let g = |v: Value| json!({"geometry": {"kind": "extrusion", "params": v}});
                num(ui, "Extrusion Depth", params.depth, 0.0..=50.0, 0.005, |v| g(json!({"depth": v})), tg.clone(), acts, key);
                slider(ui, "Twist", params.twist, -360.0..=360.0, |v| g(json!({"twist": v})), tg.clone(), acts, key);
                slider(ui, "Taper", params.taper, 0.0..=200.0, |v| g(json!({"taper": v})), tg.clone(), acts, key);
                slider(ui, "Bend X", params.bend_x, -180.0..=180.0, |v| g(json!({"bendX": v})), tg.clone(), acts, key);
                slider(ui, "Bend Y", params.bend_y, -180.0..=180.0, |v| g(json!({"bendY": v})), tg.clone(), acts, key);
                slider(ui, "Shear X", params.shear_x, -60.0..=60.0, |v| g(json!({"shearX": v})), tg.clone(), acts, key);
                slider(ui, "Shear Y", params.shear_y, -60.0..=60.0, |v| g(json!({"shearY": v})), tg.clone(), acts, key);
                num(ui, "Bevel Width", params.bevel_width, 0.0..=5.0, 0.001, |v| g(json!({"bevelWidth": v})), tg.clone(), acts, key);
                slider(ui, "Bevel Angle", params.bevel_angle, 0.0..=3.0, |v| g(json!({"bevelAngle": v})), tg.clone(), acts, key);
                num(ui, "Bevel Segments", params.bevel_segments as f32, 1.0..=16.0, 0.05, |v| g(json!({"bevelSegments": v.round() as u32})), tg.clone(), acts, key);
                ui.horizontal_wrapped(|ui| {
                    ui.label(tl!("Contour"));
                    for c in BevelContour::ALL {
                        if ui.selectable_label(params.bevel_contour == c, c.id()).clicked() {
                            acts.push(Act::Run("3d.set", json!({"target": "object", "id": id, "values": g(json!({"bevelContour": c}))})));
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label(tl!("Sides"));
                    for (s, l) in [(BevelSides::Front, "Front"), (BevelSides::Back, "Back"), (BevelSides::FrontAndBack, "Front and Back")] {
                        if ui.selectable_label(params.bevel_sides == s, l).clicked() {
                            acts.push(Act::Run("3d.set", json!({"target": "object", "id": id, "values": g(json!({"bevelSides": s}))})));
                        }
                    }
                });
                slider(ui, "Smoothing", params.smoothing, 0.0..=90.0, |v| g(json!({"smoothing": v})), tg.clone(), acts, key);
                if shapes.len() > 1 && ui.button(tl!("Split Extrusion")).clicked() {
                    acts.push(Act::Run("3d.splitExtrusion", json!({"object": id})));
                }
            }
            if let Geometry::DepthMap { strength, .. } = &o.geometry {
                slider(ui, "Depth Strength", *strength, -2.0..=2.0, |v| json!({"geometry": {"kind": "depthMap", "strength": v}}), tg.clone(), acts, key);
            }
            material_editor(ui, sc, o, st, acts, key, layers, t);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn material_editor(ui: &mut egui::Ui, sc: &Scene, o: &photocraft_three::Object, st: &mut ThreeDUi, acts: &mut Vec<Act>, key: &str, layers: &[(u64, String)], t: &Tokens) {
    ui.separator();
    ui.label(RichText::new(tl!("Materials")).strong().color(t.text));
    if o.materials.is_empty() {
        return;
    }
    st.slot = st.slot.min(o.materials.len() - 1);
    egui::ComboBox::from_id_salt("3d-slot").selected_text(o.materials[st.slot].name.clone()).show_ui(ui, |ui| {
        for (i, m) in o.materials.iter().enumerate() {
            ui.selectable_value(&mut st.slot, i, m.name.clone());
        }
    });
    let slot = st.slot;
    let m = &o.materials[slot];
    // Material preview sphere.
    let skey = (sc as *const Scene as usize, o.id, slot);
    if st.cache.swatch_key != Some(skey) {
        let px = photocraft_three::render::material_swatch(m, &sc.textures, 72);
        st.cache.swatch = Some(ui.ctx().load_texture("3d-swatch", egui::ColorImage::from_rgba_unmultiplied([72, 72], &px), egui::TextureOptions::LINEAR));
        st.cache.swatch_key = Some(skey);
    }
    ui.horizontal(|ui| {
        if let Some(tex) = &st.cache.swatch {
            ui.add(egui::Image::new((tex.id(), vec2(72.0, 72.0))));
        }
        ui.vertical(|ui| {
            ui.label(tl!("Preset"));
            egui::ComboBox::from_id_salt("3d-mat-preset").selected_text(tl!("Choose…")).show_ui(ui, |ui| {
                for p in photocraft_three::scene::MATERIAL_PRESETS {
                    if ui.button(*p).clicked() {
                        acts.push(Act::Run("3d.materialPreset", json!({"id": o.id, "slot": slot, "preset": p})));
                    }
                }
            });
            if ui.small_button(tl!("Apply to all slots")).clicked() {
                // Copy this slot's material onto every slot.
                for i in 0..o.materials.len() {
                    if i != slot
                        && let Ok(mut v) = serde_json::to_value(m)
                    {
                        if let Some(name) = o.materials.get(i).map(|x| x.name.clone()) {
                            v["name"] = json!(name);
                        }
                        acts.push(Act::Run("3d.set", json!({"target": "material", "id": o.id, "slot": i, "values": v})));
                    }
                }
            }
        });
    });
    let tg = json!({"target": "material", "id": o.id, "slot": slot});
    color(ui, "Diffuse", m.diffuse, |c| json!({"diffuse": c}), tg.clone(), acts, key);
    texture_row(ui, "Diffuse texture", m.diffuse_texture, o.id, slot, "diffuse", acts, layers);
    color(ui, "Specular", m.specular, |c| json!({"specular": c}), tg.clone(), acts, key);
    color(ui, "Illumination", m.illumination, |c| json!({"illumination": c}), tg.clone(), acts, key);
    color(ui, "Ambient", m.ambient, |c| json!({"ambient": c}), tg.clone(), acts, key);
    slider(ui, "Shine", m.shine, 0.0..=1.0, |v| json!({"shine": v}), tg.clone(), acts, key);
    slider(ui, "Reflection", m.reflection, 0.0..=1.0, |v| json!({"reflection": v}), tg.clone(), acts, key);
    slider(ui, "Roughness", m.roughness, 0.0..=1.0, |v| json!({"roughness": v}), tg.clone(), acts, key);
    slider(ui, "Bump", m.bump, 0.0..=10.0, |v| json!({"bump": v}), tg.clone(), acts, key);
    texture_row(ui, "Bump texture", m.bump_texture, o.id, slot, "bump", acts, layers);
    slider(ui, "Opacity", m.opacity, 0.0..=1.0, |v| json!({"opacity": v}), tg.clone(), acts, key);
    texture_row(ui, "Opacity texture", m.opacity_texture, o.id, slot, "opacity", acts, layers);
    slider(ui, "Refraction", m.refraction, 1.0..=2.5, |v| json!({"refraction": v}), tg.clone(), acts, key);
    texture_row(ui, "Normal map", m.normal_texture, o.id, slot, "normal", acts, layers);
    ui.label(RichText::new(tl!("UV Properties")).color(t.text_dim));
    num(ui, "Scale U", m.uv_scale[0], -100.0..=100.0, 0.01, |v| json!({"uvScale": [v, m.uv_scale[1]]}), tg.clone(), acts, key);
    num(ui, "Scale V", m.uv_scale[1], -100.0..=100.0, 0.01, |v| json!({"uvScale": [m.uv_scale[0], v]}), tg.clone(), acts, key);
    num(ui, "Offset U", m.uv_offset[0], -100.0..=100.0, 0.005, |v| json!({"uvOffset": [v, m.uv_offset[1]]}), tg.clone(), acts, key);
    num(ui, "Offset V", m.uv_offset[1], -100.0..=100.0, 0.005, |v| json!({"uvOffset": [m.uv_offset[0], v]}), tg, acts, key);
}

#[allow(clippy::too_many_arguments)]
fn texture_row(ui: &mut egui::Ui, label: &str, tex: Option<u32>, obj: u32, slot: usize, map: &'static str, acts: &mut Vec<Act>, layers: &[(u64, String)]) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).small());
        ui.label(RichText::new(if tex.is_some() { "●" } else { "○" }).small());
        if ui.small_button(tl!("Load…")).on_hover_text(tl!("Load an image file as this texture")).clicked() {
            acts.push(Act::LoadTexture { id: obj, slot, map });
        }
        ui.menu_button(RichText::new(tl!("From Layer")).small(), |ui| {
            for (id, name) in layers {
                if ui.button(name).clicked() {
                    acts.push(Act::LayerTexture { id: obj, slot, map, layer: *id });
                    ui.close();
                }
            }
        });
        if tex.is_some() && ui.small_button(tl!("Remove")).clicked() {
            acts.push(Act::Run("3d.setTexture", json!({"id": obj, "slot": slot, "map": map})));
        }
    });
}

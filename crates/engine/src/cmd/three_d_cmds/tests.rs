use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_geom::Rect;
use serde_json::json;

/// A document with one pixel layer holding an opaque red square in the middle.
fn session() -> (Session, LayerId) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 96, "height": 64})).unwrap();
    let id = s
        .edit("Fixture", |doc, active| {
            let mut l = Layer::raster("Logo", doc.pixel_format());
            if let LayerContent::Raster(surf) = &mut l.content {
                let px = photocraft_raster::from_rgba(&surf.format(), [0.9, 0.1, 0.1, 1.0]);
                surf.fill_rect(Rect::new(32, 16, 64, 48), &px);
            }
            let id = l.id;
            doc.layers.push(l);
            *active = Some(id);
            Ok(id)
        })
        .unwrap();
    (s, id)
}

fn layer_alpha(s: &Session, id: LayerId, x: i32, y: i32) -> f32 {
    let l = s.active().unwrap().doc.layer(id).unwrap();
    match &l.content {
        LayerContent::Raster(surf) => surf.rgba(x, y)[3],
        _ => 0.0,
    }
}

#[test]
fn extrusion_from_layer_keeps_its_place_and_renders() {
    let (mut s, id) = session();
    let v = s.execute("3d.newExtrusionFromLayer", json!({})).unwrap();
    assert_eq!(v["layer"], json!(id.0), "the layer is converted in place");
    let l = s.active().unwrap().doc.layer(id).unwrap();
    assert!(l.is_3d());
    let sc = l.scene3d.as_ref().unwrap();
    assert_eq!(sc.objects.len(), 1);
    assert_eq!(sc.objects[0].materials.len(), 5, "front, bevel, sides, bevel, back");
    // The front face covers where the square was, and the corners stay clear.
    assert!(layer_alpha(&s, id, 48, 32) > 0.99);
    assert!(layer_alpha(&s, id, 2, 2) < 0.01);
    // Undo restores the plain layer.
    s.execute("edit.undo", json!({})).unwrap();
    assert!(!s.active().unwrap().doc.layer(id).unwrap().is_3d());
}

#[test]
fn extrusion_from_selection_and_empty_selection() {
    let (mut s, _) = session();
    s.edit("Select", |doc, _| {
        let mut m = Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::U8, false));
        m.fill_rect(Rect::new(10, 10, 40, 30), &[1.0]);
        doc.selection = Some(m);
        Ok(())
    })
    .unwrap();
    let v = s.execute("3d.newExtrusionFromSelection", json!({})).unwrap();
    let new = LayerId(v["layer"].as_u64().unwrap());
    assert!(layer_alpha(&s, new, 25, 20) > 0.99);
    s.edit("Empty", |doc, _| {
        doc.selection = Some(Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::U8, false)));
        Ok(())
    })
    .unwrap();
    assert!(s.execute("3d.newExtrusionFromSelection", json!({})).is_err());
}

#[test]
fn every_mesh_preset_postcard_and_depth_map() {
    for preset in MeshPreset::ALL.iter().filter(|p| !matches!(p, MeshPreset::Plane | MeshPreset::TwoSidedPlane)) {
        let (mut s, id) = session();
        s.execute(&format!("3d.newMeshFromLayer.meshPreset.{}", preset.id()), json!({})).unwrap();
        let l = s.active().unwrap().doc.layer(id).unwrap();
        assert!(l.is_3d(), "{preset:?}");
        // Rings and donuts have a hole in the middle: look for the mesh anywhere.
        let covered = (0..96).any(|x| (0..64).any(|y| layer_alpha(&s, id, x, y) > 0.5));
        let centre = layer_alpha(&s, id, 48, 32) > 0.5;
        assert!(covered && (centre || matches!(preset, MeshPreset::Ring | MeshPreset::Donut)), "{preset:?} renders in the middle");
    }
    let (mut s, id) = session();
    s.execute("3d.newMeshFromLayer.postcard", json!({})).unwrap();
    assert!(layer_alpha(&s, id, 48, 32) > 0.99);
    for kind in ["plane", "twoSidedPlane", "cylinder", "sphere"] {
        let (mut s, _) = session();
        s.execute(&format!("3d.newMeshFromLayer.depthMap.{kind}"), json!({})).unwrap();
    }
}

#[test]
fn scene_edits_lights_navigation_and_undo() {
    let (mut s, id) = session();
    s.execute("3d.newMeshFromLayer.meshPreset.cube", json!({})).unwrap();
    let sc = scene_of(&s, Some(id)).unwrap();
    let oid = sc.objects[0].id;
    s.execute("3d.set", json!({"target": "material", "id": oid, "slot": 0, "values": {"diffuse": [0.0, 0.0, 1.0]}})).unwrap();
    assert_eq!(scene_of(&s, Some(id)).unwrap().objects[0].materials[0].diffuse, [0.0, 0.0, 1.0]);
    s.execute("3d.set", json!({"target": "object", "id": oid, "values": {"transform": {"position": {"x": 0.25}}}})).unwrap();
    assert!((scene_of(&s, Some(id)).unwrap().objects[0].transform.position.x - 0.25).abs() < 1e-6);
    let before = scene_of(&s, Some(id)).unwrap().camera.yaw;
    // A drag coalesces into one history step.
    for _ in 0..5 {
        s.execute("3d.navigate", json!({"mode": "orbit", "dx": 10, "dy": 0, "coalesce": "orbit-1"})).unwrap();
    }
    assert!((scene_of(&s, Some(id)).unwrap().camera.yaw - (before - 20.0)).abs() < 1e-3);
    s.execute("edit.undo", json!({})).unwrap();
    assert!((scene_of(&s, Some(id)).unwrap().camera.yaw - before).abs() < 1e-3, "one undo reverts the whole drag");
    s.execute("3d.addLight.spot", json!({})).unwrap();
    s.execute("3d.addLight.point", json!({})).unwrap();
    assert_eq!(scene_of(&s, Some(id)).unwrap().lights.len(), 3);
    s.execute("3d.lightPreset.threePoint", json!({})).unwrap();
    assert_eq!(scene_of(&s, Some(id)).unwrap().lights.len(), 3);
    s.execute("3d.addObject.sphere", json!({})).unwrap();
    s.execute("3d.duplicateObject", json!({})).unwrap();
    assert_eq!(scene_of(&s, Some(id)).unwrap().objects.len(), 3);
    s.execute("3d.materialPreset", json!({"preset": "metalGold"})).unwrap();
    for st in ["wireframe", "normals", "depthMap", "lineIllustration", "points", "solid"] {
        s.execute(&format!("3d.style.{st}"), json!({})).unwrap();
    }
    for v in ["front", "top", "left", "fit", "default"] {
        s.execute(&format!("3d.view.{v}"), json!({})).unwrap();
    }
    s.execute("3d.saveView", json!({"name": "Mine"})).unwrap();
    s.execute("3d.navigate", json!({"mode": "slide", "dx": 0, "dy": 40})).unwrap();
    s.execute("3d.restoreView", json!({"name": "Mine"})).unwrap();
    s.execute("3d.crossSection", json!({})).unwrap();
    s.execute("3d.moveObjectToGroundPlane", json!({})).unwrap();
    let info = s.execute("3d.inspect", json!({})).unwrap();
    assert_eq!(info["summary"]["objects"].as_array().unwrap().len(), 3);
}

#[test]
fn ray_traced_render_rasterize_and_merge() {
    let (mut s, id) = session();
    s.execute("3d.newMeshFromLayer.meshPreset.sphere", json!({})).unwrap();
    s.execute("3d.render", json!({"samples": 2})).unwrap();
    assert_eq!(scene_of(&s, Some(id)).unwrap().state, RenderState::RayTraced);
    assert!(layer_alpha(&s, id, 48, 32) > 0.99);
    // A second 3D layer, then merge both.
    let v = s.execute("3d.newExtrusionFromLayer", json!({"layer": id.0})).unwrap();
    assert_eq!(v["layer"], json!(id.0));
    s.edit("Second", |doc, _| {
        let mut sc = Scene::default();
        sc.add_object("Cube", Geometry::Preset { preset: MeshPreset::Cube, aspect: 1.0 }, vec![]);
        let mut l = Layer::raster("Other 3D", doc.pixel_format());
        l.scene3d = Some(Arc::new(sc));
        let lid = l.id;
        doc.layers.push(l);
        render_layer(doc, lid)
    })
    .unwrap();
    let other_id = s.active().unwrap().doc.layers.last().unwrap().id;
    s.execute("layer.select", json!({"layers": [id.0, other_id.0]})).ok();
    if two_3d(&s).is_ok() {
        s.execute("3d.mergeLayers", json!({})).unwrap();
        assert!(s.active().unwrap().doc.layer(other_id).is_none() || s.active().unwrap().doc.layer(id).is_none());
    }
    s.execute("3d.rasterize", json!({"layer": id.0})).ok();
}

#[test]
fn export_import_and_native_format_round_trip() {
    let (mut s, id) = session();
    s.execute("3d.newMeshFromLayer.meshPreset.cube", json!({})).unwrap();
    let sc = scene_of(&s, Some(id)).unwrap();
    for ext in ["obj", "stl", "gltf", "glb"] {
        let files = export_files(&sc, &format!("model.{ext}")).unwrap();
        let lookup = |n: &str| files.iter().find(|(f, _)| f == n).map(|(_, b)| b.clone());
        let v = new_layer_from_bytes(&mut s, &files[0].0, &files[0].1, &lookup).unwrap();
        let lid = LayerId(v["layer"].as_u64().unwrap());
        assert!(s.active().unwrap().doc.layer(lid).unwrap().is_3d(), "{ext}");
        if ext == "glb" || ext == "obj" {
            // The cube's texture came along.
            assert!(!scene_of(&s, Some(lid)).unwrap().textures.is_empty(), "{ext} keeps the texture");
        }
    }
    let doc = s.active().unwrap().doc.clone();
    let bytes = photocraft_format::save_to_bytes(&doc, &Default::default()).unwrap();
    let back = photocraft_format::load_from_bytes(&bytes).unwrap();
    let l = back.layer(id).or_else(|| back.layers.iter().find(|l| l.name == "Logo")).unwrap();
    let sc2 = l.scene3d.as_ref().expect("the scene survives .pcraft");
    assert_eq!(sc2.objects.len(), sc.objects.len());
    assert_eq!(sc2.objects[0].mesh().triangle_count(), sc.objects[0].mesh().triangle_count());
}

#[test]
fn bad_params_are_errors_not_panics() {
    let (mut s, _) = session();
    assert!(s.execute("3d.render", json!({})).is_err(), "not a 3D layer");
    s.execute("3d.newMeshFromLayer.meshPreset.cone", json!({})).unwrap();
    for (cmd, p) in [
        ("3d.set", json!({"target": "nope", "values": {}})),
        ("3d.set", json!({"target": "object", "id": 999, "values": {}})),
        ("3d.set", json!({"target": "camera", "values": {"fov": "wide"}})),
        ("3d.set", json!({"target": "camera", "values": 4})),
        ("3d.navigate", json!({"mode": "spin", "dx": 1, "dy": 1})),
        ("3d.addObject", json!({"preset": "teapot"})),
        ("3d.deleteNode", json!({"id": 12345})),
        ("3d.setTexture", json!({"map": "glow", "id": 1})),
        ("3d.materialPreset", json!({"preset": "unobtainium"})),
        ("3d.restoreView", json!({"name": "missing"})),
        ("3d.splitExtrusion", json!({})),
        ("3d.newLayerFromFile", json!({"path": "/no/such/file.obj"})),
        ("3d.exportLayer", json!({"path": "x.dae"})),
    ] {
        assert!(s.execute(cmd, p.clone()).is_err(), "{cmd} {p}");
    }
    // Extreme but well-formed numbers clamp instead of breaking the renderer.
    s.execute("3d.set", json!({"target": "camera", "values": {"fov": 1e9, "distance": -5.0}})).unwrap();
    s.execute("3d.navigate", json!({"mode": "scale", "dx": 0, "dy": 1e30})).unwrap();
    assert!(new_layer_from_bytes(&mut s, "x.obj", b"not a model", &|_| None).is_err());
}

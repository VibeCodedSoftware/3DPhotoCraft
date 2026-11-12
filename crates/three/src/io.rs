//! 3D files: import OBJ (+MTL), STL (binary and ASCII), glTF 2.0 (.gltf, .glb); export OBJ+MTL,
//! STL, glTF and GLB. Collada (.dae), 3DS, U3D and KMZ, which Photoshop also read, are not
//! supported and say so.
//!
//! Images are decoded and encoded by the caller (this crate has no image codecs): imports return
//! texture references, exports take a PNG encoder.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::math::{Mat4, Vec3, linear_to_srgb, srgb_to_linear, v3};
use crate::mesh::Mesh;
use crate::scene::{Image, Material, Scene};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Unsupported(String),
    #[error("not a valid {format} file: {msg}")]
    Invalid { format: &'static str, msg: String },
}

fn invalid(format: &'static str, msg: impl Into<String>) -> Error {
    Error::Invalid { format, msg: msg.into() }
}

/// Most triangles accepted from one file.
pub const MAX_TRIANGLES: usize = 4_000_000;

/// Where a texture of an imported material comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum ImageRef {
    /// A file name relative to the model (OBJ `map_Kd`, glTF `uri`).
    Path(String),
    /// Encoded bytes embedded in the file (glTF).
    Bytes(Vec<u8>),
}

#[derive(Clone, Debug)]
pub struct ImportedMaterial {
    pub material: Material,
    pub diffuse: Option<ImageRef>,
    pub bump: Option<ImageRef>,
    pub normal: Option<ImageRef>,
    pub opacity: Option<ImageRef>,
}

impl ImportedMaterial {
    fn plain(material: Material) -> Self {
        ImportedMaterial { material, diffuse: None, bump: None, normal: None, opacity: None }
    }
}

#[derive(Clone, Debug)]
pub struct ImportedObject {
    pub name: String,
    pub mesh: Mesh,
    pub materials: Vec<ImportedMaterial>,
}

#[derive(Clone, Debug, Default)]
pub struct Imported {
    pub objects: Vec<ImportedObject>,
    pub warnings: Vec<String>,
}

/// File extensions [`import`] reads.
pub const IMPORT_EXTENSIONS: &[&str] = &["obj", "stl", "gltf", "glb"];
/// File extensions [`export`] writes.
pub const EXPORT_EXTENSIONS: &[&str] = &["obj", "stl", "gltf", "glb"];

fn ext_of(name: &str) -> String {
    name.rsplit('.').next().unwrap_or("").to_ascii_lowercase()
}

/// Read a 3D file. `resolve` fetches a sibling file by relative name (MTL libraries, external
/// glTF buffers); it may return `None`.
pub fn import(name: &str, bytes: &[u8], resolve: &dyn Fn(&str) -> Option<Vec<u8>>) -> Result<Imported, Error> {
    let ext = ext_of(name);
    let mut out = match ext.as_str() {
        "obj" => import_obj(bytes, resolve),
        "stl" => import_stl(bytes),
        "gltf" | "glb" => import_gltf(bytes, resolve),
        "dae" | "3ds" | "u3d" | "kmz" | "fbx" | "idtf" => Err(Error::Unsupported(format!(
            ".{ext} files aren't supported. Export the model as OBJ, STL, glTF or GLB from your 3D application and open that."
        ))),
        _ => Err(Error::Unsupported(format!("\"{name}\" isn't a 3D file PhotoCraft reads (OBJ, STL, glTF, GLB)."))),
    }?;
    for o in &mut out.objects {
        o.mesh.sanitize();
    }
    out.objects.retain(|o| o.mesh.triangle_count() > 0);
    if out.objects.is_empty() {
        return Err(invalid(fmt_name(&ext), "it contains no triangles"));
    }
    Ok(out)
}

fn fmt_name(ext: &str) -> &'static str {
    match ext {
        "obj" => "OBJ",
        "stl" => "STL",
        "glb" => "GLB",
        _ => "glTF",
    }
}

// ---------------------------------------------------------------- OBJ

fn parse_f(s: Option<&str>) -> f32 {
    s.and_then(|v| v.parse::<f32>().ok()).filter(|v| v.is_finite()).unwrap_or(0.0)
}

fn import_obj(bytes: &[u8], resolve: &dyn Fn(&str) -> Option<Vec<u8>>) -> Result<Imported, Error> {
    let text = String::from_utf8_lossy(bytes);
    let mut warnings = Vec::new();
    let (mut v, mut vt, mut vn): (Vec<Vec3>, Vec<[f32; 2]>, Vec<Vec3>) = (Vec::new(), Vec::new(), Vec::new());
    let mut mtl: HashMap<String, ImportedMaterial> = HashMap::new();
    struct Cur {
        name: String,
        mesh: Mesh,
        slots: Vec<String>,
        slot: u16,
        map: HashMap<(i64, i64, i64), u32>,
    }
    let new_cur = |name: &str| Cur { name: name.into(), mesh: Mesh::default(), slots: vec!["Default Material".into()], slot: 0, map: HashMap::new() };
    let mut cur = new_cur("Mesh");
    let mut done: Vec<Cur> = Vec::new();
    let mut tris = 0usize;
    for line in text.lines() {
        let line = line.trim();
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => v.push(v3(parse_f(it.next()), parse_f(it.next()), parse_f(it.next()))),
            Some("vt") => vt.push([parse_f(it.next()), 1.0 - parse_f(it.next())]),
            Some("vn") => vn.push(v3(parse_f(it.next()), parse_f(it.next()), parse_f(it.next())).normalize()),
            Some("o") | Some("g") => {
                let name = it.collect::<Vec<_>>().join(" ");
                if cur.mesh.triangle_count() > 0 {
                    let next = new_cur(if name.is_empty() { "Mesh" } else { &name });
                    let prev_slot = cur.slots.get(cur.slot as usize).cloned();
                    done.push(std::mem::replace(&mut cur, next));
                    if let Some(s) = prev_slot {
                        cur.slots[0] = s;
                    }
                } else if !name.is_empty() {
                    cur.name = name;
                }
            }
            Some("usemtl") => {
                let name = it.collect::<Vec<_>>().join(" ");
                cur.slot = match cur.slots.iter().position(|s| *s == name) {
                    Some(i) => i as u16,
                    None if cur.mesh.triangle_count() == 0 && cur.slots.len() == 1 => {
                        cur.slots[0] = name;
                        0
                    }
                    None => {
                        cur.slots.push(name);
                        (cur.slots.len() - 1).min(u16::MAX as usize) as u16
                    }
                };
            }
            Some("mtllib") => {
                let file = it.collect::<Vec<_>>().join(" ");
                match resolve(&file) {
                    Some(b) => mtl.extend(parse_mtl(&String::from_utf8_lossy(&b))),
                    None => warnings.push(format!("Material library \"{file}\" wasn't found next to the model; default materials are used.")),
                }
            }
            Some("f") => {
                let mut idx: Vec<u32> = Vec::new();
                for tok in it {
                    let mut parts = tok.split('/');
                    let fix = |s: Option<&str>, n: usize| -> Option<i64> {
                        let i: i64 = s.filter(|s| !s.is_empty())?.parse().ok()?;
                        let r = if i < 0 { n as i64 + i } else { i - 1 };
                        (r >= 0 && (r as usize) < n).then_some(r)
                    };
                    let Some(pi) = fix(parts.next(), v.len()) else { continue };
                    let ti = fix(parts.next(), vt.len()).unwrap_or(-1);
                    let ni = fix(parts.next(), vn.len()).unwrap_or(-1);
                    let key = (pi, ti, ni);
                    let id = match cur.map.get(&key) {
                        Some(&id) => id,
                        None => {
                            let p = v[pi as usize];
                            let n = if ni >= 0 { vn[ni as usize] } else { Vec3::ZERO };
                            let uv = if ti >= 0 { vt[ti as usize] } else { [0.0, 0.0] };
                            let id = cur.mesh.vertex(p, n, uv);
                            cur.map.insert(key, id);
                            id
                        }
                    };
                    idx.push(id);
                }
                for k in 1..idx.len().saturating_sub(1) {
                    cur.mesh.tri(idx[0], idx[k], idx[k + 1], cur.slot);
                    tris += 1;
                }
                if tris > MAX_TRIANGLES {
                    return Err(invalid("OBJ", format!("more than {MAX_TRIANGLES} triangles")));
                }
            }
            _ => {}
        }
    }
    done.push(cur);
    let objects = done
        .into_iter()
        .filter(|c| c.mesh.triangle_count() > 0)
        .map(|c| {
            let needs_normals = c.mesh.normals.iter().any(|n| n.length() < 0.5);
            let mut mesh = c.mesh;
            if needs_normals {
                mesh.compute_normals(40.0);
            }
            let materials = c.slots.iter().map(|s| mtl.get(s).cloned().unwrap_or_else(|| ImportedMaterial::plain(Material::named(s)))).collect();
            ImportedObject { name: c.name, mesh, materials }
        })
        .collect();
    Ok(Imported { objects, warnings })
}

fn parse_mtl(text: &str) -> HashMap<String, ImportedMaterial> {
    let mut out = HashMap::new();
    let mut cur: Option<(String, ImportedMaterial)> = None;
    let col = |it: &mut std::str::SplitWhitespace<'_>| -> [f32; 3] { [parse_f(it.next()).clamp(0.0, 1.0), parse_f(it.next()).clamp(0.0, 1.0), parse_f(it.next()).clamp(0.0, 1.0)] };
    // Texture statements end with the file name after any options.
    let file = |it: std::str::SplitWhitespace<'_>| -> Option<ImageRef> { it.last().map(|s| ImageRef::Path(s.replace('\\', "/"))) };
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let Some(k) = it.next() else { continue };
        if k == "newmtl" {
            if let Some((n, m)) = cur.take() {
                out.insert(n, m);
            }
            let name = it.collect::<Vec<_>>().join(" ");
            cur = Some((name.clone(), ImportedMaterial::plain(Material::named(&name))));
            continue;
        }
        let Some((_, m)) = cur.as_mut() else { continue };
        match k {
            "Kd" => m.material.diffuse = col(&mut it),
            "Ks" => m.material.specular = col(&mut it),
            "Ka" => m.material.ambient = col(&mut it),
            "Ke" => m.material.illumination = col(&mut it),
            "Ns" => m.material.shine = (parse_f(it.next()) / 1000.0).clamp(0.0, 1.0).sqrt(),
            "d" => m.material.opacity = parse_f(it.next()).clamp(0.0, 1.0),
            "Tr" => m.material.opacity = 1.0 - parse_f(it.next()).clamp(0.0, 1.0),
            "Ni" => m.material.refraction = parse_f(it.next()).clamp(1.0, 3.0),
            "map_Kd" => m.diffuse = file(it),
            "map_Bump" | "map_bump" | "bump" => m.bump = file(it),
            "norm" => m.normal = file(it),
            "map_d" => m.opacity = file(it),
            _ => {}
        }
    }
    if let Some((n, m)) = cur {
        out.insert(n, m);
    }
    out
}

// ---------------------------------------------------------------- STL

fn import_stl(bytes: &[u8]) -> Result<Imported, Error> {
    let mut mesh = Mesh::default();
    let binary_len = bytes.get(80..84).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
    let is_binary = binary_len.is_some_and(|n| n.checked_mul(50).and_then(|v| v.checked_add(84)) == Some(bytes.len()));
    if is_binary {
        let n = binary_len.unwrap_or(0);
        if n > MAX_TRIANGLES {
            return Err(invalid("STL", format!("more than {MAX_TRIANGLES} triangles")));
        }
        let f = |o: usize| bytes.get(o..o + 4).map_or(0.0, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        for t in 0..n {
            let o = 84 + t * 50;
            let p: [Vec3; 3] = std::array::from_fn(|k| v3(f(o + 12 + k * 12), f(o + 16 + k * 12), f(o + 20 + k * 12)));
            add_flat(&mut mesh, p);
        }
    } else {
        let text = String::from_utf8_lossy(bytes);
        if !text.trim_start().starts_with("solid") {
            return Err(invalid("STL", "neither binary nor ASCII STL"));
        }
        let mut pts: Vec<Vec3> = Vec::new();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if it.next() == Some("vertex") {
                pts.push(v3(parse_f(it.next()), parse_f(it.next()), parse_f(it.next())));
                if pts.len() == 3 {
                    add_flat(&mut mesh, [pts[0], pts[1], pts[2]]);
                    pts.clear();
                    if mesh.triangle_count() > MAX_TRIANGLES {
                        return Err(invalid("STL", format!("more than {MAX_TRIANGLES} triangles")));
                    }
                }
            }
        }
    }
    // STL is usually Z-up: stand it up in our Y-up world.
    mesh.transform(&Mat4::rot_x(-90.0));
    Ok(Imported { objects: vec![ImportedObject { name: "STL Mesh".into(), mesh, materials: vec![ImportedMaterial::plain(Material::default())] }], warnings: Vec::new() })
}

fn add_flat(m: &mut Mesh, p: [Vec3; 3]) {
    let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or(Vec3::Z);
    let a = m.vertex(p[0], n, [0.0, 0.0]);
    let b = m.vertex(p[1], n, [1.0, 0.0]);
    let c = m.vertex(p[2], n, [0.0, 1.0]);
    m.tri(a, b, c, 0);
}

// ---------------------------------------------------------------- glTF

fn import_gltf(bytes: &[u8], resolve: &dyn Fn(&str) -> Option<Vec<u8>>) -> Result<Imported, Error> {
    let (json_bytes, bin): (&[u8], Option<&[u8]>) = if bytes.starts_with(b"glTF") {
        let u = |o: usize| bytes.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
        if u(4) != Some(2) {
            return Err(invalid("GLB", "only glTF 2.0 is supported"));
        }
        let mut off = 12;
        let (mut js, mut bn) = (None, None);
        while let (Some(len), Some(kind)) = (u(off), u(off + 4)) {
            let data = bytes.get(off + 8..off + 8 + len).ok_or_else(|| invalid("GLB", "a chunk runs past the end of the file"))?;
            match kind {
                0x4E4F_534A => js = Some(data),
                0x004E_4942 => bn = Some(data),
                _ => {}
            }
            off += 8 + len;
        }
        (js.ok_or_else(|| invalid("GLB", "no JSON chunk"))?, bn)
    } else {
        (bytes, None)
    };
    let doc: Value = serde_json::from_slice(json_bytes).map_err(|e| invalid("glTF", e.to_string()))?;
    let mut warnings = Vec::new();
    let arr = |k: &str| doc.get(k).and_then(Value::as_array).cloned().unwrap_or_default();
    // Buffers.
    let mut buffers: Vec<Vec<u8>> = Vec::new();
    for (i, b) in arr("buffers").iter().enumerate() {
        let data = match b.get("uri").and_then(Value::as_str) {
            None => bin.map(<[u8]>::to_vec).unwrap_or_default(),
            Some(uri) if uri.starts_with("data:") => uri.split_once(',').and_then(|(_, d)| crate::b64::decode(d)).unwrap_or_default(),
            Some(uri) => resolve(uri).unwrap_or_else(|| {
                warnings.push(format!("Buffer {i} (\"{uri}\") wasn't found next to the model."));
                Vec::new()
            }),
        };
        buffers.push(data);
    }
    let views = arr("bufferViews");
    let accessors = arr("accessors");
    let view_bytes = |vi: usize| -> Option<(&[u8], usize)> {
        let v = views.get(vi)?;
        let buf = buffers.get(v.get("buffer")?.as_u64()? as usize)?;
        let off = v.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let len = v.get("byteLength")?.as_u64()? as usize;
        let stride = v.get("byteStride").and_then(Value::as_u64).unwrap_or(0) as usize;
        Some((buf.get(off..off.checked_add(len)?)?, stride))
    };
    // Accessor → floats (n components each).
    let read = |ai: usize| -> Option<(Vec<f32>, usize)> {
        let a = accessors.get(ai)?;
        let count = a.get("count")?.as_u64()? as usize;
        if count > MAX_TRIANGLES * 3 {
            return None;
        }
        let ncomp = match a.get("type")?.as_str()? {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            _ => return None,
        };
        let ct = a.get("componentType")?.as_u64()?;
        let norm = a.get("normalized").and_then(Value::as_bool).unwrap_or(false);
        let csize = match ct {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            _ => return None,
        };
        let (data, stride) = view_bytes(a.get("bufferView")?.as_u64()? as usize)?;
        let off = a.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let stride = if stride == 0 { csize * ncomp } else { stride };
        let mut out = Vec::with_capacity(count * ncomp);
        for i in 0..count {
            for c in 0..ncomp {
                let o = off + i * stride + c * csize;
                let b = data.get(o..o + csize)?;
                let v = match ct {
                    5120 => {
                        let x = b[0] as i8 as f32;
                        if norm { (x / 127.0).max(-1.0) } else { x }
                    }
                    5121 => {
                        let x = b[0] as f32;
                        if norm { x / 255.0 } else { x }
                    }
                    5122 => {
                        let x = i16::from_le_bytes([b[0], b[1]]) as f32;
                        if norm { (x / 32767.0).max(-1.0) } else { x }
                    }
                    5123 => {
                        let x = u16::from_le_bytes([b[0], b[1]]) as f32;
                        if norm { x / 65535.0 } else { x }
                    }
                    5125 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                    _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                };
                out.push(v);
            }
        }
        Some((out, ncomp))
    };
    let read_u32 = |ai: usize| -> Option<Vec<u32>> {
        let a = accessors.get(ai)?;
        let count = a.get("count")?.as_u64()? as usize;
        let ct = a.get("componentType")?.as_u64()?;
        let csize = match ct {
            5121 => 1,
            5123 => 2,
            5125 => 4,
            _ => return None,
        };
        let (data, stride) = view_bytes(a.get("bufferView")?.as_u64()? as usize)?;
        let off = a.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let stride = if stride == 0 { csize } else { stride };
        (0..count)
            .map(|i| {
                let o = off + i * stride;
                let b = data.get(o..o + csize)?;
                Some(match csize {
                    1 => u32::from(b[0]),
                    2 => u32::from(u16::from_le_bytes([b[0], b[1]])),
                    _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                })
            })
            .collect()
    };
    // Images and materials.
    let images = arr("images");
    let textures = arr("textures");
    let image_ref = |tex: Option<&Value>| -> Option<ImageRef> {
        let ti = tex?.get("index")?.as_u64()? as usize;
        let ii = textures.get(ti)?.get("source")?.as_u64()? as usize;
        let img = images.get(ii)?;
        if let Some(uri) = img.get("uri").and_then(Value::as_str) {
            if uri.starts_with("data:") {
                return uri.split_once(',').and_then(|(_, d)| crate::b64::decode(d)).map(ImageRef::Bytes);
            }
            return Some(ImageRef::Path(uri.to_string()));
        }
        let (data, _) = view_bytes(img.get("bufferView")?.as_u64()? as usize)?;
        Some(ImageRef::Bytes(data.to_vec()))
    };
    let mats: Vec<ImportedMaterial> = arr("materials")
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let pbr = m.get("pbrMetallicRoughness");
            let f = |k: &str, d: f64| pbr.and_then(|p| p.get(k)).and_then(Value::as_f64).unwrap_or(d) as f32;
            let base: Vec<f32> = pbr.and_then(|p| p.get("baseColorFactor")).and_then(Value::as_array).map(|a| a.iter().map(|v| v.as_f64().unwrap_or(1.0) as f32).collect()).unwrap_or_else(|| vec![1.0; 4]);
            let g = |k: usize| base.get(k).copied().unwrap_or(1.0).clamp(0.0, 1.0);
            let (metal, rough) = (f("metallicFactor", 1.0).clamp(0.0, 1.0), f("roughnessFactor", 1.0).clamp(0.0, 1.0));
            let em: Vec<f32> = m.get("emissiveFactor").and_then(Value::as_array).map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default();
            let diffuse = [linear_to_srgb(g(0)), linear_to_srgb(g(1)), linear_to_srgb(g(2))];
            let spec = 0.25 + (diffuse.iter().sum::<f32>() / 3.0 - 0.25) * metal;
            let blend = m.get("alphaMode").and_then(Value::as_str) == Some("BLEND");
            let material = Material {
                name: m.get("name").and_then(Value::as_str).map_or_else(|| format!("Material {}", i + 1), str::to_string),
                diffuse,
                specular: [spec; 3],
                illumination: [em.first().copied().unwrap_or(0.0), em.get(1).copied().unwrap_or(0.0), em.get(2).copied().unwrap_or(0.0)],
                shine: 1.0 - rough,
                reflection: metal * 0.8,
                roughness: rough,
                opacity: if blend { g(3) } else { 1.0 },
                double_sided: m.get("doubleSided").and_then(Value::as_bool).unwrap_or(false),
                ..Default::default()
            };
            ImportedMaterial { material, diffuse: image_ref(pbr.and_then(|p| p.get("baseColorTexture"))), bump: None, normal: image_ref(m.get("normalTexture")), opacity: None }
        })
        .collect();
    // Node hierarchy.
    let nodes = arr("nodes");
    let meshes = arr("meshes");
    let scene_idx = doc.get("scene").and_then(Value::as_u64).unwrap_or(0) as usize;
    let roots: Vec<usize> = doc
        .get("scenes")
        .and_then(Value::as_array)
        .and_then(|s| s.get(scene_idx))
        .and_then(|s| s.get("nodes"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
        .unwrap_or_else(|| (0..nodes.len()).collect());
    let mut objects = Vec::new();
    let mut stack: Vec<(usize, Mat4, u32)> = roots.into_iter().map(|r| (r, Mat4::IDENTITY, 0)).collect();
    let mut total = 0usize;
    let mut skipped_modes = false;
    while let Some((ni, parent, depth)) = stack.pop() {
        let Some(node) = nodes.get(ni) else { continue };
        if depth > 64 {
            warnings.push("The node hierarchy is deeper than 64 levels; deeper nodes were skipped.".into());
            continue;
        }
        let m = parent * node_matrix(node);
        if let Some(children) = node.get("children").and_then(Value::as_array) {
            for c in children.iter().filter_map(Value::as_u64) {
                stack.push((c as usize, m, depth + 1));
            }
        }
        let Some(mi) = node.get("mesh").and_then(Value::as_u64) else { continue };
        let Some(gm) = meshes.get(mi as usize) else { continue };
        let mut mesh = Mesh::default();
        let mut slots: Vec<Option<usize>> = Vec::new();
        for prim in gm.get("primitives").and_then(Value::as_array).into_iter().flatten() {
            if prim.get("mode").and_then(Value::as_u64).unwrap_or(4) != 4 {
                skipped_modes = true;
                continue;
            }
            let attrs = prim.get("attributes");
            let Some((pos, 3)) = attrs.and_then(|a| a.get("POSITION")).and_then(Value::as_u64).and_then(|i| read(i as usize)) else { continue };
            let nrm = attrs.and_then(|a| a.get("NORMAL")).and_then(Value::as_u64).and_then(|i| read(i as usize)).filter(|(_, n)| *n == 3);
            let uv = attrs.and_then(|a| a.get("TEXCOORD_0")).and_then(Value::as_u64).and_then(|i| read(i as usize)).filter(|(_, n)| *n == 2);
            let n = pos.len() / 3;
            let base = mesh.positions.len() as u32;
            for i in 0..n {
                let p = v3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
                let nn = nrm.as_ref().and_then(|(d, _)| d.get(i * 3..i * 3 + 3)).map_or(Vec3::ZERO, |d| v3(d[0], d[1], d[2]));
                let t = uv.as_ref().and_then(|(d, _)| d.get(i * 2..i * 2 + 2)).map_or([0.0, 0.0], |d| [d[0], d[1]]);
                mesh.vertex(p, nn, t);
            }
            let mat = prim.get("material").and_then(Value::as_u64).map(|v| v as usize);
            let slot = match slots.iter().position(|s| *s == mat) {
                Some(s) => s,
                None => {
                    slots.push(mat);
                    slots.len() - 1
                }
            } as u16;
            let idx: Vec<u32> = match prim.get("indices").and_then(Value::as_u64) {
                Some(ai) => read_u32(ai as usize).unwrap_or_default(),
                None => (0..n as u32).collect(),
            };
            for t in idx.chunks_exact(3) {
                mesh.tri(base + t[0], base + t[1], base + t[2], slot);
            }
            total += idx.len() / 3;
            if total > MAX_TRIANGLES {
                return Err(invalid("glTF", format!("more than {MAX_TRIANGLES} triangles")));
            }
        }
        if mesh.triangle_count() == 0 {
            continue;
        }
        mesh.transform(&m);
        let materials = slots.iter().map(|s| s.and_then(|i| mats.get(i).cloned()).unwrap_or_else(|| ImportedMaterial::plain(Material::default()))).collect();
        let name = node.get("name").or_else(|| gm.get("name")).and_then(Value::as_str).map_or_else(|| format!("Mesh {}", objects.len() + 1), str::to_string);
        objects.push(ImportedObject { name, mesh, materials });
    }
    if skipped_modes {
        warnings.push("Points and lines in the file were skipped; only triangles are imported.".into());
    }
    if doc.get("skins").is_some() || doc.get("animations").is_some() {
        warnings.push("Skinning and animation aren't supported; the model is imported in its rest pose.".into());
    }
    Ok(Imported { objects, warnings })
}

fn node_matrix(n: &Value) -> Mat4 {
    let nums = |k: &str| -> Option<Vec<f32>> { n.get(k)?.as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()) };
    if let Some(m) = nums("matrix").filter(|m| m.len() == 16) {
        // Column-major.
        let mut r = Mat4::IDENTITY;
        for c in 0..4 {
            for row in 0..4 {
                r.m[row][c] = m[c * 4 + row];
            }
        }
        return r;
    }
    let t = nums("translation").filter(|v| v.len() == 3).map_or(Vec3::ZERO, |v| v3(v[0], v[1], v[2]));
    let s = nums("scale").filter(|v| v.len() == 3).map_or(Vec3::ONE, |v| v3(v[0], v[1], v[2]));
    let r = nums("rotation").filter(|v| v.len() == 4).map_or(Mat4::IDENTITY, |q| quat(q[0], q[1], q[2], q[3]));
    Mat4::translate(t) * r * Mat4::scale(s)
}

fn quat(x: f32, y: f32, z: f32, w: f32) -> Mat4 {
    let l = (x * x + y * y + z * z + w * w).sqrt().max(1e-12);
    let (x, y, z, w) = (x / l, y / l, z / l, w / l);
    Mat4 {
        m: [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), 0.0],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), 0.0],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
}

// ---------------------------------------------------------------- export

/// Files an export produces: (file name, bytes). The first is the main file.
pub type Files = Vec<(String, Vec<u8>)>;

/// Export the visible objects of `scene` as `name` (extension picks the format). `png` encodes
/// a texture image to PNG bytes.
pub fn export(scene: &Scene, name: &str, png: &dyn Fn(&Image) -> Option<Vec<u8>>) -> Result<Files, Error> {
    let ext = ext_of(name);
    let stem = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let stem = stem.rsplit_once('.').map_or(stem, |(s, _)| s).to_string();
    if scene.objects.iter().all(|o| !o.visible) {
        return Err(Error::Unsupported("The 3D layer has no visible objects to export.".into()));
    }
    match ext.as_str() {
        "obj" => Ok(export_obj(scene, &stem, png)),
        "stl" => Ok(vec![(format!("{stem}.stl"), export_stl(scene))]),
        "glb" => Ok(vec![(format!("{stem}.glb"), export_glb(scene, png))]),
        "gltf" => Ok(vec![(format!("{stem}.gltf"), export_gltf(scene, png).into_bytes())]),
        "dae" | "kmz" | "u3d" | "3ds" | "idtf" => Err(Error::Unsupported(format!(".{ext} export isn't supported; use OBJ, STL, glTF or GLB."))),
        _ => Err(Error::Unsupported(format!("Pick a 3D format to export: .obj, .stl, .gltf or .glb (got \"{name}\")."))),
    }
}

/// World-space meshes of the visible objects, with their materials.
fn baked(scene: &Scene) -> Vec<(String, Mesh, Vec<Material>)> {
    scene
        .objects
        .iter()
        .filter(|o| o.visible)
        .map(|o| {
            let mut m = (*o.mesh()).clone();
            m.transform(&o.transform.matrix());
            (o.name.clone(), m, o.materials.clone())
        })
        .collect()
}

fn export_obj(scene: &Scene, stem: &str, png: &dyn Fn(&Image) -> Option<Vec<u8>>) -> Files {
    let mut obj = format!("# PhotoCraft 3D export\nmtllib {stem}.mtl\n");
    let mut mtl = String::from("# PhotoCraft 3D materials\n");
    let mut files: Files = Vec::new();
    let mut written_tex: HashMap<u32, String> = HashMap::new();
    let mut base = 1usize;
    for (oi, (name, m, mats)) in baked(scene).into_iter().enumerate() {
        obj.push_str(&format!("o {}\n", name.replace(['\n', '\r'], " ")));
        for p in &m.positions {
            obj.push_str(&format!("v {} {} {}\n", p.x, p.y, p.z));
        }
        for t in &m.uvs {
            obj.push_str(&format!("vt {} {}\n", t[0], 1.0 - t[1]));
        }
        for n in &m.normals {
            obj.push_str(&format!("vn {} {} {}\n", n.x, n.y, n.z));
        }
        for slot in 0..m.slot_count() {
            let mname = format!("m{oi}_{slot}");
            obj.push_str(&format!("usemtl {mname}\n"));
            for (t, s) in m.indices.iter().zip(&m.material) {
                if *s as usize == slot {
                    let f = |i: u32| {
                        let k = i as usize + base;
                        format!("{k}/{k}/{k}")
                    };
                    obj.push_str(&format!("f {} {} {}\n", f(t[0]), f(t[1]), f(t[2])));
                }
            }
            let mat = mats.get(slot).cloned().unwrap_or_default();
            mtl.push_str(&format!(
                "newmtl {mname}\n# {}\nKd {} {} {}\nKs {} {} {}\nKe {} {} {}\nNs {}\nd {}\nNi {}\n",
                mat.name.replace(['\n', '\r'], " "),
                mat.diffuse[0],
                mat.diffuse[1],
                mat.diffuse[2],
                mat.specular[0],
                mat.specular[1],
                mat.specular[2],
                mat.illumination[0],
                mat.illumination[1],
                mat.illumination[2],
                mat.shine * mat.shine * 1000.0,
                mat.opacity,
                mat.refraction
            ));
            if let Some(id) = mat.diffuse_texture
                && let Some(img) = scene.texture(Some(id))
            {
                let file = match written_tex.get(&id) {
                    Some(f) => Some(f.clone()),
                    None => png(img).map(|bytes| {
                        let f = format!("{stem}_texture{id}.png");
                        files.push((f.clone(), bytes));
                        written_tex.insert(id, f.clone());
                        f
                    }),
                };
                if let Some(f) = file {
                    mtl.push_str(&format!("map_Kd {f}\n"));
                }
            }
        }
        base += m.positions.len();
    }
    let mut out = vec![(format!("{stem}.obj"), obj.into_bytes()), (format!("{stem}.mtl"), mtl.into_bytes())];
    out.extend(files);
    out
}

fn export_stl(scene: &Scene) -> Vec<u8> {
    let meshes = baked(scene);
    let n: usize = meshes.iter().map(|(_, m, _)| m.triangle_count()).sum();
    let mut out = Vec::with_capacity(84 + n * 50);
    let mut header = [0u8; 80];
    let tag = b"PhotoCraft 3D export";
    header[..tag.len()].copy_from_slice(tag);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(n as u32).to_le_bytes());
    // Back to Z-up for 3D printing tools.
    let zup = Mat4::rot_x(90.0);
    for (_, m, _) in &meshes {
        for t in 0..m.triangle_count() {
            let [a, b, c] = m.indices[t].map(|i| zup.transform_point(m.positions[i as usize]));
            let nn = (b - a).cross(c - a).normalize_or(Vec3::ZERO);
            for v in [nn, a, b, c] {
                for f in [v.x, v.y, v.z] {
                    out.extend_from_slice(&f.to_le_bytes());
                }
            }
            out.extend_from_slice(&[0, 0]);
        }
    }
    out
}

/// glTF JSON plus the binary buffer.
fn gltf_parts(scene: &Scene, png: &dyn Fn(&Image) -> Option<Vec<u8>>) -> (Value, Vec<u8>) {
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut accessors = Vec::new();
    let mut meshes = Vec::new();
    let mut nodes = Vec::new();
    let mut materials = Vec::new();
    let mut images = Vec::new();
    let mut textures = Vec::new();
    let mut tex_index: HashMap<u32, usize> = HashMap::new();
    let pad = |bin: &mut Vec<u8>| {
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
    };
    let mut push_view = |bin: &mut Vec<u8>, data: &[u8], target: Option<u32>| -> usize {
        pad(bin);
        let off = bin.len();
        bin.extend_from_slice(data);
        let mut v = json!({"buffer": 0, "byteOffset": off, "byteLength": data.len()});
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        views.push(v);
        views.len() - 1
    };
    for o in scene.objects.iter().filter(|o| o.visible) {
        let m = o.mesh();
        let mut prims = Vec::new();
        for slot in 0..m.slot_count() {
            // Gather this slot's triangles with their own vertex set.
            let mut remap: HashMap<u32, u32> = HashMap::new();
            let (mut pos, mut nrm, mut uv, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (t, s) in m.indices.iter().zip(&m.material) {
                if *s as usize != slot {
                    continue;
                }
                for &i in t {
                    let ni = *remap.entry(i).or_insert_with(|| {
                        let p = m.positions[i as usize];
                        let n = m.normals.get(i as usize).copied().unwrap_or(Vec3::Z);
                        let u = m.uvs.get(i as usize).copied().unwrap_or([0.0, 0.0]);
                        pos.push(p);
                        nrm.push(n);
                        uv.push(u);
                        (pos.len() - 1) as u32
                    });
                    idx.push(ni);
                }
            }
            if idx.is_empty() {
                continue;
            }
            let fbytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
            let pv: Vec<f32> = pos.iter().flat_map(|p: &Vec3| p.to_array()).collect();
            let nv: Vec<f32> = nrm.iter().flat_map(|p: &Vec3| p.to_array()).collect();
            let uvv: Vec<f32> = uv.iter().flatten().copied().collect();
            let mut b = crate::mesh::Aabb::EMPTY;
            for &p in &pos {
                b.add(p);
            }
            let vp = push_view(&mut bin, &fbytes(&pv), Some(34962));
            accessors.push(json!({"bufferView": vp, "componentType": 5126, "count": pos.len(), "type": "VEC3", "min": b.min.to_array(), "max": b.max.to_array()}));
            let ap = accessors.len() - 1;
            let vn = push_view(&mut bin, &fbytes(&nv), Some(34962));
            accessors.push(json!({"bufferView": vn, "componentType": 5126, "count": nrm.len(), "type": "VEC3"}));
            let an = accessors.len() - 1;
            let vu = push_view(&mut bin, &fbytes(&uvv), Some(34962));
            accessors.push(json!({"bufferView": vu, "componentType": 5126, "count": uv.len(), "type": "VEC2"}));
            let au = accessors.len() - 1;
            let ib: Vec<u8> = idx.iter().flat_map(|i| i.to_le_bytes()).collect();
            let vi = push_view(&mut bin, &ib, Some(34963));
            accessors.push(json!({"bufferView": vi, "componentType": 5125, "count": idx.len(), "type": "SCALAR"}));
            let ai = accessors.len() - 1;
            // Material.
            let mat = o.materials.get(slot).cloned().unwrap_or_default();
            let lin = |c: f32| srgb_to_linear(c);
            let mut pbr = json!({
                "baseColorFactor": [lin(mat.diffuse[0]), lin(mat.diffuse[1]), lin(mat.diffuse[2]), mat.opacity],
                "metallicFactor": (mat.reflection / 0.8).clamp(0.0, 1.0),
                "roughnessFactor": (1.0 - mat.shine).clamp(0.0, 1.0),
            });
            if let Some(id) = mat.diffuse_texture
                && let Some(img) = scene.texture(Some(id))
            {
                let ti = match tex_index.get(&id) {
                    Some(&t) => Some(t),
                    None => png(img).map(|bytes| {
                        let v = push_view(&mut bin, &bytes, None);
                        images.push(json!({"bufferView": v, "mimeType": "image/png"}));
                        textures.push(json!({"source": images.len() - 1}));
                        tex_index.insert(id, textures.len() - 1);
                        textures.len() - 1
                    }),
                };
                if let Some(ti) = ti {
                    pbr["baseColorTexture"] = json!({"index": ti});
                    pbr["baseColorFactor"] = json!([1.0, 1.0, 1.0, mat.opacity]);
                }
            }
            let mut mj = json!({"name": mat.name, "pbrMetallicRoughness": pbr, "doubleSided": mat.double_sided, "emissiveFactor": mat.illumination.map(|v| v.clamp(0.0, 1.0))});
            if mat.opacity < 1.0 {
                mj["alphaMode"] = json!("BLEND");
            }
            materials.push(mj);
            prims.push(json!({"attributes": {"POSITION": ap, "NORMAL": an, "TEXCOORD_0": au}, "indices": ai, "material": materials.len() - 1, "mode": 4}));
        }
        if prims.is_empty() {
            continue;
        }
        meshes.push(json!({"name": o.name, "primitives": prims}));
        let t = &o.transform;
        let mtx = t.matrix();
        let mut col = Vec::with_capacity(16);
        for c in 0..4 {
            for r in 0..4 {
                col.push(mtx.m[r][c]);
            }
        }
        nodes.push(json!({"name": o.name, "mesh": meshes.len() - 1, "matrix": col}));
    }
    pad(&mut bin);
    let node_ids: Vec<usize> = (0..nodes.len()).collect();
    let mut doc = json!({
        "asset": {"version": "2.0", "generator": "PhotoCraft 3D"},
        "scene": 0,
        "scenes": [{"nodes": node_ids}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": bin.len()}],
    });
    if !images.is_empty() {
        doc["images"] = json!(images);
        doc["textures"] = json!(textures);
    }
    (doc, bin)
}

fn export_glb(scene: &Scene, png: &dyn Fn(&Image) -> Option<Vec<u8>>) -> Vec<u8> {
    let (doc, bin) = gltf_parts(scene, png);
    let mut js = serde_json::to_vec(&doc).unwrap_or_default();
    while js.len() % 4 != 0 {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(js.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
    out.extend_from_slice(&js);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E_4942u32.to_le_bytes());
    out.extend_from_slice(&bin);
    out
}

fn export_gltf(scene: &Scene, png: &dyn Fn(&Image) -> Option<Vec<u8>>) -> String {
    let (mut doc, bin) = gltf_parts(scene, png);
    doc["buffers"] = json!([{"byteLength": bin.len(), "uri": format!("data:application/octet-stream;base64,{}", crate::b64::encode(&bin))}]);
    serde_json::to_string_pretty(&doc).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::MeshPreset;
    use crate::scene::Geometry;

    fn scene() -> Scene {
        let mut s = Scene::default();
        s.add_object("Cube", Geometry::Preset { preset: MeshPreset::Cube, aspect: 1.0 }, vec![Material::default()]);
        let id = s.add_object("Ball", Geometry::Preset { preset: MeshPreset::Sphere, aspect: 1.0 }, vec![Material::default()]);
        if let Some(o) = s.object_mut(id) {
            o.transform.position = v3(2.0, 0.0, 0.0);
        }
        s
    }

    fn tris(i: &Imported) -> usize {
        i.objects.iter().map(|o| o.mesh.triangle_count()).sum()
    }

    #[test]
    fn every_format_round_trips() {
        let s = scene();
        let expect = s.triangle_count();
        let png = |_: &Image| Some(vec![1, 2, 3]);
        for ext in EXPORT_EXTENSIONS {
            let files = export(&s, &format!("model.{ext}"), &png).unwrap();
            let main = &files[0];
            let lookup = |n: &str| files.iter().find(|(f, _)| f == n).map(|(_, b)| b.clone());
            let back = import(&main.0, &main.1, &lookup).unwrap();
            assert_eq!(tris(&back), expect, "{ext}");
            // Geometry lands in the same place (the sphere sits at x = 2).
            let maxx = back.objects.iter().flat_map(|o| o.mesh.positions.iter()).map(|p| p.x).fold(f32::MIN, f32::max);
            assert!((maxx - 2.5).abs() < 0.02, "{ext}: {maxx}");
        }
    }

    #[test]
    fn obj_with_quads_negative_indices_and_mtl() {
        let obj = "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nusemtl red\nf -4 -3 -2 -1\n";
        let mtl = "newmtl red\nKd 1 0 0\nmap_Kd tex/red.png\n";
        let r = import("a.obj", obj.as_bytes(), &|n| (n == "m.mtl").then(|| mtl.as_bytes().to_vec())).unwrap();
        assert_eq!(tris(&r), 2);
        let m = &r.objects[0].materials[0];
        assert_eq!(m.material.diffuse, [1.0, 0.0, 0.0]);
        assert_eq!(m.diffuse, Some(ImageRef::Path("tex/red.png".into())));
    }

    #[test]
    fn bad_files_give_errors_not_panics() {
        assert!(import("x.dae", b"<COLLADA/>", &|_| None).is_err());
        assert!(import("x.stl", b"garbage", &|_| None).is_err());
        assert!(import("x.glb", b"glTF\x02\0\0\0\xff\xff\xff\xff\x10\0\0\0JSON", &|_| None).is_err());
        assert!(import("x.gltf", b"{\"accessors\":[{\"count\":99999999999}]}", &|_| None).is_err());
        assert!(import("x.obj", b"f 1 2 3\n", &|_| None).is_err());
        let mut s = Scene::default();
        s.add_object("Cube", Geometry::Preset { preset: MeshPreset::Cube, aspect: 1.0 }, vec![]);
        assert!(export(&s, "a.dae", &|_| None).is_err());
    }
}

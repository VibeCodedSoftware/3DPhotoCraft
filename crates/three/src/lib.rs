//! `photocraft-three`: the 3D engine behind PhotoCraft's 3D layers, a clean-room return of the
//! 3D workspace Photoshop removed in 2023.
//!
//! A 3D layer is a pixel layer whose pixels are rendered from a [`Scene`]: objects (mesh presets,
//! extrusions of layers, selections and paths, depth maps, imported models), Photoshop-style
//! materials and textures, infinite / spot / point lights, an image-based environment with a
//! ground-plane shadow catcher, and a camera. [`render::render_preview`] rasterizes it for live
//! editing; [`trace::render_traced`] ray traces the final render. [`io`] reads and writes OBJ,
//! STL and glTF.
//!
//! The crate is standalone (no workspace dependencies), pure Rust and wasm-safe.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod b64;
pub mod depth;
pub mod extrude;
pub mod io;
pub mod math;
pub mod mesh;
pub mod render;
pub mod scene;
pub mod trace;

pub use math::{Mat4, Vec3, v3};
pub use mesh::{Aabb, Mesh, MeshPreset};
pub use render::{Output, render_preview};
pub use scene::{Camera, Environment, Geometry, Image, Light, LightKind, Material, Object, RenderSettings, Scene, Style, Texture};
pub use trace::render_traced;

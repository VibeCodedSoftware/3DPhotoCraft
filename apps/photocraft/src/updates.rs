//! Release checks through `craft-update`, the library every Craft App shares. Only public release
//! metadata is requested.

use photocraft_engine::updates_cmds::{CheckFn, CheckResult, Release};

pub fn service() -> CheckFn {
    Box::new(|| {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("update check".into())
            .spawn(move || {
                let result = check();
                if let Err(e) = &result {
                    log::warn!("Update check: {e}");
                }
                let _ = tx.send(result);
            })
            .map_err(|e| format!("Could not start update check: {e}"))?;
        Ok(rx)
    })
}

/// 3DPhotoCraft has its own releases (not PhotoCraft's): its files are named `3dphotocraft-<version>-...`, and
/// its program is still `photocraft.exe`.
pub const APP: craft_update::App = craft_update::App {
    former_ids: &["photocraft"],
    ..craft_update::App::new("3dphotocraft", "3DPhotoCraft", "Photo editing with a 3D workspace", "VibeCodedSoftware/3DPhotoCraft")
};

fn check() -> CheckResult {
    let http = craft_update::Http::new("3DPhotoCraft/Update")?;
    let found = craft_update::check_for_update(&APP, &http, photocraft_engine::build_info::VERSION)?;
    Ok(found.map(|release| Release { version: release.tag.trim_start_matches(['v', 'V']).to_string(), url: release.html_url }))
}

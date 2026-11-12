<#
.SYNOPSIS
  Package the 3DPhotoCraft portable zip (PhotoCraft with the 3D workspace) for Windows x64.

.DESCRIPTION
  Expects a finished static-CRT build of photocraft and photocraft-cli for x86_64-pc-windows-msvc:
    $env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS = '-C target-feature=+crt-static'
    cargo build --release -p photocraft -p photocraft-cli --features heif --target x86_64-pc-windows-msvc
  Writes 3DPhotoCraft-<version>-windows-x64.zip to -OutDir (default: dist\release).
#>
param([string] $OutDir)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $OutDir) { $OutDir = Join-Path $Root 'dist\release' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Version = $null
$inPkg = $false
foreach ($line in Get-Content (Join-Path $Root 'Cargo.toml')) {
  if ($line -match '^\[workspace\.package\]') { $inPkg = $true; continue }
  if ($inPkg -and $line -match '^\[') { break }
  if ($inPkg -and $line -match '^version\s*=\s*"([^"]+)"') { $Version = $Matches[1]; break }
}
if (-not $Version) { throw 'no [workspace.package] version in Cargo.toml' }

$Bin = Join-Path $Root 'target\x86_64-pc-windows-msvc\release'
foreach ($exe in 'photocraft.exe', 'photocraft-cli.exe') {
  if (-not (Test-Path (Join-Path $Bin $exe))) { throw "$exe not built in $Bin" }
}
# The app must be a GUI-subsystem x64 binary, the CLI a console one.
function Get-PeHeader([string] $Path) {
  $bytes = [System.IO.File]::ReadAllBytes($Path)
  $pe = [BitConverter]::ToInt32($bytes, 0x3C)
  return @{ Machine = [BitConverter]::ToUInt16($bytes, $pe + 4); Subsystem = [BitConverter]::ToUInt16($bytes, $pe + 0x5C) }
}
foreach ($check in @(@('photocraft.exe', 2), @('photocraft-cli.exe', 3))) {
  $h = Get-PeHeader (Join-Path $Bin $check[0])
  if ($h.Machine -ne 0x8664 -or $h.Subsystem -ne $check[1]) { throw "$($check[0]): unexpected PE header" }
}

$Name = "3DPhotoCraft-$Version-windows-x64"
$Stage = Join-Path $Root "target\windows-package\$Name"
Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
Copy-Item (Join-Path $Bin 'photocraft.exe'), (Join-Path $Bin 'photocraft-cli.exe') $Stage
foreach ($f in 'README.md', 'LICENSE-MIT', 'LICENSE-APACHE', 'NOTICE', 'ATTRIBUTION.md') {
  Copy-Item (Join-Path $Root $f) $Stage
}
Copy-Item (Join-Path $Root 'docs\3d.md') (Join-Path $Stage '3D-GUIDE.md')
Copy-Item (Join-Path $PSScriptRoot 'portable.txt') $Stage
$Examples = Join-Path $Stage 'examples\3d'
New-Item -ItemType Directory -Force -Path $Examples | Out-Null
Copy-Item (Join-Path $Root 'examples\3d\*') $Examples

@"
3DPhotoCraft $Version for Windows (x64), portable
=================================================

PhotoCraft, the open-source image editor, with Photoshop's removed 3D workspace brought back.

Start:  double-click photocraft.exe (nothing to install; settings stay in PhotoCraftData\ here).

Try 3D in one minute:
  1. File > New (any size).
  2. 3D > New 3D Layer from File... and open examples\3d\crate.obj
     (its .mtl materials and checker.png texture load with it).
  3. Window > 3D: drag on the canvas to Orbit / Roll / Pan / Slide / Scale.
  4. 3D > Light Presets > Three-Point Lighting, then 3D > Render 3D Layer (Alt+Shift+Ctrl+R).
  Or type some text with the Type tool and choose 3D > New 3D Extrusion from Selected Layer.

Full guide: 3D-GUIDE.md.  Command line: photocraft-cli.exe --help.

Licence: MIT OR Apache-2.0 (LICENSE-MIT, LICENSE-APACHE, NOTICE, ATTRIBUTION.md).
PhotoCraft is an independent open-source project, not affiliated with Adobe Inc.
Unsigned build: Windows SmartScreen may ask you to confirm the first run.
"@ | Set-Content -Encoding utf8 (Join-Path $Stage 'READ-ME-FIRST.txt')

$Zip = Join-Path $OutDir "$Name.zip"
Remove-Item -Force $Zip -ErrorAction SilentlyContinue
Compress-Archive -Path $Stage -DestinationPath $Zip -CompressionLevel Optimal
$hash = (Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLowerInvariant()
"$hash  $Name.zip" | Set-Content -Encoding ascii "$Zip.sha256"
Get-Item $Zip | Select-Object FullName, Length
"sha256 $hash"

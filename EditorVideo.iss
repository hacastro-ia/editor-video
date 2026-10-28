; Script para Inno Setup — Editor de Video
; Requisitos: Inno Setup 6 (https://jrsoftware.org/isinfo.php) — descargar el instalador "innosetup-6-*.exe"

; La versión se puede pasar al compilar:  ISCC.exe /DAppVersion=1.1.0 EditorVideo.iss
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppName=Editor de Video
AppVersion={#AppVersion}
DefaultDirName={autopf}\EditorVideo
DefaultGroupName=Editor Video
OutputBaseFilename=EditorVideo-Setup
Compression=lzma
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\EditorVideo.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Editor Video"; Filename: "{app}\EditorVideo.exe"; WorkingDir: "{app}"
Name: "{commondesktop}\Editor Video"; Filename: "{app}\EditorVideo.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\EditorVideo.exe"; Description: "{cm:LaunchProgram,{app}}"; Flags: nowait postinstall skipifsilent

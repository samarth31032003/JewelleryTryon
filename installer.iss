[Setup]
; --- Application Details ---
AppName=AR Jewelry TryOn
; The version is injected dynamically by GitHub Actions
AppVersion={#MyAppVersion}
AppPublisher=Aspiron IT solution
DefaultDirName={autopf}\AR Jewelry TryOn
DefaultGroupName=AR Jewelry TryOn
UninstallDisplayIcon={app}\main.exe
Compression=lzma2
SolidCompression=yes
OutputDir=Output
; Filename will dynamically include the version number
OutputBaseFilename=AR_Jewelry_TryOn_Setup_v{#MyAppVersion}

; --- Visuals ---
; SetupIconFile=data\assets\icon.ico

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; --- 1. The Main Executable ---
Source: "dist\main\main.exe"; DestDir: "{app}"; Flags: ignoreversion

; --- 2. The Internal PyInstaller Folder (CRITICAL) ---
Source: "dist\main\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Creates the Start Menu shortcut
Name: "{group}\AR Jewelry TryOn"; Filename: "{app}\main.exe"
Name: "{group}\Uninstall AR Jewelry TryOn"; Filename: "{uninstallexe}"

; Creates the Desktop shortcut (if they checked the box)
Name: "{autodesktop}\AR Jewelry TryOn"; Filename: "{app}\main.exe"; Tasks: desktopicon

[Run]
; Offers to launch the app immediately after installation finishes
Filename: "{app}\main.exe"; Description: "{cm:LaunchProgram,AR Jewelry TryOn}"; Flags: nowait postinstall skipifsilent
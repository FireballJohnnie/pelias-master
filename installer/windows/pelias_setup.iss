#define MyAppName "Pelias AI Unified System"
#define MyAppVersion "0.2.0-v58c"
#define MyAppPublisher "John Lipka & Pelias Contributors"
#define MyAppExeName "pelias.exe"

[Setup]
AppId={{8E58C07A-58B0-4F26-9A11-7E11A558C001}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Pelias
DefaultGroupName=Pelias AI
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\build-installer
OutputBaseFilename=Pelias_v58c_Unified_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Files]
Source: "..\..\build\host\pelias.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\resources\ui\source\logo_full.png"; DestDir: "{app}\resources\ui\source"; Flags: ignoreversion
Source: "..\..\resources\ui\spear\*"; DestDir: "{app}\resources\ui\spear"; Flags: ignoreversion recursesubdirs
Source: "..\..\profiles\*"; DestDir: "{app}\profiles"; Flags: ignoreversion recursesubdirs
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Pelias Lithic GUI (All-in-One)"; Filename: "{app}\{#MyAppExeName}"; Parameters: "gui --port 8062"; WorkingDir: "{app}"
Name: "{group}\Pelias Hardware & Bus Probe"; Filename: "{app}\{#MyAppExeName}"; Parameters: "probe"; WorkingDir: "{app}"
Name: "{autodesktop}\Pelias Lithic GUI"; Filename: "{app}\{#MyAppExeName}"; Parameters: "gui --port 8062"; WorkingDir: "{app}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "gui --port 8062"; Description: "Launch Pelias Lithic GUI now"; Flags: nowait postinstall skipifsilent

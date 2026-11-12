; Inno Setup script for SimpleWhisper.
; The relative defaults resolve against the folder holding this script, so they
; work when the script is compiled by hand from the repository root:
;
;     iscc installer\simplewhisper.iss

#ifndef AppName
#define AppName "SimpleWhisper"
#endif

#ifndef AppVersion
#define AppVersion "0.1.0"
#endif

#ifndef Publisher
#define Publisher "SimpleWhisper"
#endif

; The PyInstaller folder build: dist\SimpleWhisper
#ifndef SourceDir
#define SourceDir "..\dist\SimpleWhisper"
#endif

; build.bat passes the absolute path; this default is for running by hand.
#ifndef LicenseFile
#define LicenseFile "..\LICENSE"
#endif

#ifndef OutputDir
#define OutputDir "..\build\installer"
#endif

#ifndef OutputBaseFilename
#define OutputBaseFilename "simplewhisper-setup"
#endif

[Setup]
AppId={{2E7C9B14-3F8A-4D62-9A5E-6C1B8D4F0A77}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
AppCopyright=MIT license
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\SimpleWhisper-GUI.exe
#ifdef SetupIconFile
SetupIconFile={#SetupIconFile}
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; Everything PyInstaller produced: both executables and the _internal folder.
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; The MIT notice travels with the program.
Source: "{#LicenseFile}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName} (command line)"; Filename: "{app}\SimpleWhisper.exe"
Name: "{autoprograms}\{#AppName} (desktop app)"; Filename: "{app}\SimpleWhisper-GUI.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\SimpleWhisper-GUI.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SimpleWhisper-GUI.exe"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent
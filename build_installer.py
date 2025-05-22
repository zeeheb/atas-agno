#!/usr/bin/env python
"""
Build script for creating a standalone executable of the iATAS application
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    """Main build function to create the executable and installer"""
    print("Building iATAS Application...")
    
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyInstaller==5.13.0"])
    
    # Create spec file for PyInstaller
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('app_data', 'app_data'),
        ('locale', 'locale'),
        ('docs', 'docs'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='iATAS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_data/icon.ico' if os.path.exists('app_data/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='iATAS',
)
"""
    
    # Write the spec file
    with open('iATAS.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    # Create an icon file if it doesn't exist
    icon_path = Path('app_data/icon.ico')
    if not icon_path.exists():
        # Create app_data directory if it doesn't exist
        icon_path.parent.mkdir(exist_ok=True)
        print("Note: No icon file found at app_data/icon.ico. Using default icon.")
    
    # Ensure docs directory exists
    docs_dir = Path('docs')
    if not docs_dir.exists():
        docs_dir.mkdir()
    
    # Run PyInstaller
    print("Running PyInstaller to build executable...")
    subprocess.check_call([
        sys.executable, 
        "-m", 
        "PyInstaller", 
        "iATAS.spec", 
        "--clean"
    ])
    
    # Create installer using NSIS (if available on Windows)
    if sys.platform == 'win32':
        try:
            # Check if NSIS is installed
            nsis_path = r"C:\Program Files (x86)\NSIS\makensis.exe"
            if os.path.exists(nsis_path):
                print("Creating Windows installer with NSIS...")
                # Create NSIS script
                nsis_script = """
!include "MUI2.nsh"

; Application information
Name "iATAS - Analisador de ATAS"
OutFile "iATAS_Setup.exe"
InstallDir "$PROGRAMFILES\\iATAS"
InstallDirRegKey HKCU "Software\\iATAS" ""

; Interface settings
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\\Contrib\\Graphics\\Icons\\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\\Contrib\\Graphics\\Icons\\modern-uninstall.ico"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Languages
!insertmacro MUI_LANGUAGE "Portuguese"

; Installation section
Section "Install"
    SetOutPath "$INSTDIR"
    
    ; Copy files
    File /r "dist\\iATAS\\*.*"
    
    ; Create shortcuts
    CreateDirectory "$SMPROGRAMS\\iATAS"
    CreateShortcut "$SMPROGRAMS\\iATAS\\iATAS.lnk" "$INSTDIR\\iATAS.exe"
    CreateShortcut "$DESKTOP\\iATAS.lnk" "$INSTDIR\\iATAS.exe"
    
    ; Create uninstaller
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
    
    ; Create registry entries
    WriteRegStr HKCU "Software\\iATAS" "" $INSTDIR
    WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\iATAS" "DisplayName" "iATAS - Analisador de ATAS"
    WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\iATAS" "UninstallString" "$INSTDIR\\Uninstall.exe"
SectionEnd

; Uninstallation section
Section "Uninstall"
    ; Remove files and directories
    RMDir /r "$INSTDIR"
    
    ; Remove shortcuts
    Delete "$SMPROGRAMS\\iATAS\\iATAS.lnk"
    RMDir "$SMPROGRAMS\\iATAS"
    Delete "$DESKTOP\\iATAS.lnk"
    
    ; Remove registry entries
    DeleteRegKey HKCU "Software\\iATAS"
    DeleteRegKey HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\iATAS"
SectionEnd
"""
                
                # Write NSIS script
                with open('installer.nsi', 'w', encoding='utf-8') as f:
                    f.write(nsis_script)
                
                # Run NSIS
                subprocess.check_call([nsis_path, "installer.nsi"])
                print("Windows installer created successfully: iATAS_Setup.exe")
            else:
                print("NSIS not found. Skipping installer creation.")
                print("To create an installer, install NSIS from https://nsis.sourceforge.io/Download")
                print("Then run: makensis installer.nsi")
        except Exception as e:
            print(f"Error creating installer: {e}")
            print("You can distribute the executable from the 'dist/iATAS' directory.")
    
    print("\nBuild completed!")
    print("You can find the executable in the 'dist/iATAS' directory.")
    if sys.platform == 'win32' and os.path.exists("iATAS_Setup.exe"):
        print("The installer is available as 'iATAS_Setup.exe'")
    
    print("\nTo distribute this application:")
    print("1. If an installer was created, distribute 'iATAS_Setup.exe'")
    print("2. Otherwise, zip the 'dist/iATAS' directory and distribute the zip file")
    
if __name__ == "__main__":
    main() 
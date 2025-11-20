; WhisperJAV v1.5.3 Custom Desktop Shortcut Creation
; This snippet is injected into constructor's NSIS template
; to create a desktop shortcut after installation completes

; Check if user wants shortcuts (respects constructor's shortcut checkbox)
${If} $Ana_CreateShortcuts_State = ${BST_CHECKED}

    ; Set working directory to installation folder (self-contained app)
    SetOutPath "$INSTDIR"

    ; Create desktop shortcut
    ; Target: WhisperJAV v1.5.3.lnk on user's desktop
    ; Executable: pythonw.exe (no console window)
    ; Parameters: -m whisperjav.webview_gui.main (launch GUI)
    ; Icon: whisperjav_icon.ico in installation folder
    ; Start in: $INSTDIR (working directory already set by SetOutPath)
    CreateShortCut "$DESKTOP\WhisperJAV v1.5.3.lnk" \
        "$INSTDIR\pythonw.exe" \
        "-m whisperjav.webview_gui.main" \
        "$INSTDIR\whisperjav_icon.ico" \
        0 \
        SW_SHOWNORMAL \
        "" \
        "WhisperJAV v1.5.3 - Japanese AV Subtitle Generator"

    ; Log success (visible in installer UI if debug enabled)
    DetailPrint "Desktop shortcut created: WhisperJAV v1.5.3"

${EndIf}

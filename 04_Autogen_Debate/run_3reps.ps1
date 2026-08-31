$ErrorActionPreference = "Stop"
$env:OLLAMA_CONTEXT_LENGTH = "8192"
$logFile = "run_3reps.log"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $msg" | Tee-Object -FilePath $logFile -Append
}

Log "=== Iniciando 3 repeticiones de AutoGen (8B heterogeneo, num_ctx=8192 corregido) ==="

foreach ($rep in 1,2,3) {
    $decisionCheck = "Resultados\rep$rep\q050\decision_050.json"
    if (Test-Path $decisionCheck) {
        Log "SKIP rep${rep}: ya tiene q050."
        continue
    }
    Log "LANZANDO rep$rep..."
    $psi = Start-Process -FilePath "..\venv\Scripts\python.exe" `
        -ArgumentList "autogen_debate.py --start 1 --rep $rep" `
        -WorkingDirectory "c:\Users\Laura\Desktop\TFG_Laura\04_Autogen_Debate" `
        -RedirectStandardOutput "rep${rep}_stdout.log" `
        -RedirectStandardError "rep${rep}_stderr.log" `
        -NoNewWindow -PassThru
    $psi.WaitForExit()
    if (Test-Path $decisionCheck) {
        Log "  OK: rep$rep completado."
    } else {
        Log "  AVISO: rep$rep termino (code=$($psi.ExitCode)) pero q050 no existe."
    }
}

Log "=== 3 repeticiones de AutoGen TERMINADAS ==="

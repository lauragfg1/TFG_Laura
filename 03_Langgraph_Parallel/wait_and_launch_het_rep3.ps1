$ErrorActionPreference = "Stop"
$logFile = "wait_and_launch_het_rep3.log"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $msg" | Tee-Object -FilePath $logFile -Append
}

Log "Esperando a que termine el proceso actual (8B/homogeneo/rep3)..."

while ($true) {
    $procs = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*launch_debate.py*--size 8B*--mode homogeneo*--rep 3*" }
    if (-not $procs) {
        break
    }
    Start-Sleep -Seconds 15
}

Log "Proceso homogeneo/rep3 finalizado. Esperando 5s de margen..."
Start-Sleep -Seconds 5

Log "Lanzando 8B/heterogeneo/rep3 (q001 a q050)..."

$psi = Start-Process -FilePath ".\..\venv\Scripts\python.exe" `
    -ArgumentList "launch_debate.py ../Dataset_Preguntas ./data --size 8B --mode heterogeneo --rep 3 --start 1 --limit 50" `
    -WorkingDirectory "c:\Users\Laura\Desktop\TFG_Laura\03_Langgraph_Parallel" `
    -RedirectStandardOutput "het_rep3_stdout.log" `
    -RedirectStandardError "het_rep3_stderr.log" `
    -NoNewWindow -PassThru

Log "Lanzado con PID $($psi.Id). Esperando a que termine (hasta 50 preguntas)..."

$psi.WaitForExit()

Log "8B/heterogeneo/rep3 (hasta 50) ha terminado. Codigo de salida: $($psi.ExitCode)"

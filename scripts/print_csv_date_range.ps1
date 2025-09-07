$csvPath = Join-Path $PSScriptRoot "..\data\smart_logistics_dataset.csv"
if (-not (Test-Path $csvPath)) {
  Write-Output "Dataset not found: $csvPath"
  exit 1
}

$csv = Import-Csv $csvPath
$dates = @()
foreach ($row in $csv) {
  try {
    $dt = [datetime]::ParseExact($row.Timestamp, 'yyyy-MM-dd HH:mm:ss', $null)
    $dates += $dt
  } catch {
    try {
      $dt = [datetime]::Parse($row.Timestamp)
      $dates += $dt
    } catch {
      # skip invalid
    }
  }
}

if ($dates.Count -eq 0) {
  Write-Output "No valid timestamps found."
  exit 0
}

$min = ($dates | Measure-Object -Minimum).Minimum
$max = ($dates | Measure-Object -Maximum).Maximum

Write-Output ("CSV date range: {0} to {1}" -f $min.ToString('yyyy-MM-dd'), $max.ToString('yyyy-MM-dd'))


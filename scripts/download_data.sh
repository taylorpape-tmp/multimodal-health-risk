#!/usr/bin/env bash

#download_data.sh acquires all raw data for the Multimodal Diabetes Risk Predictor.
#One function per source. Each downloads into data/raw/<source>/ and checksums the
#result so the pipeline is reproducible on any machine. All URLs verified against
#each dataset's published data-availability statement.

#Usage:
#  bash scripts/download_data.sh /download everything
#  bash scripts/download_data.sh omics /download a single source
#  bash scripts/download_data.sh /cgm imaging
#
#Sources: omics | cgm | imaging | wearable

set -euo pipefail

#resolve repo root so the script works from anywhere
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${REPO_ROOT}/data/raw"

#curl flags: -L follow redirects, -f fail on HTTP error, -C - resume if interrupted, --retry for flaky links
CURL="curl -fL -C - --retry 3 --retry-delay 5"

#writes a .sha256 next to the file 
checksum () {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" > "$f.sha256"
  else
    shasum -a 256 "$f" > "$f.sha256"
  fi
  echo "  checksum -> $f.sha256"
}

#omics + clinical labs, Zhou et al. 2019 Nature (doi 10.1038/s41586-019-1236-x)
#three Springer supplementary workbooks, ~48 MB total
#these workbooks hold both the omics matrices and the clinical labs, so they land
#in data/raw/clinical (the omics/ folder is reserved for any future raw omics reads)
#MOESM3 = main supplementary tables (S1_Subjects, S4/S8/S9 matrices, etc.)
#MOESM4 = within-subject associations, MOESM5 = between-subject associations
#license: data availability statement states CC0 for the dataset
download_omics () {
  local dir="${RAW}/clinical"; mkdir -p "$dir"
  echo "[omics] Zhou 2019 Nature supplementary tables -> $dir"
  local base="https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1236-x/MediaObjects"
  $CURL -o "$dir/zhou2019_supp_tables.xlsx"            "$base/41586_2019_1236_MOESM3_ESM.xlsx"
  $CURL -o "$dir/zhou2019_within_subject_assoc.xlsx"   "$base/41586_2019_1236_MOESM4_ESM.xlsx"
  $CURL -o "$dir/zhou2019_between_subject_assoc.xlsx"  "$base/41586_2019_1236_MOESM5_ESM.xlsx"
  checksum "$dir/zhou2019_supp_tables.xlsx"
}

#CGM, Hall et al. 2018 PLoS Biology (doi 10.1371/journal.pbio.2005143)
#"S1 Data" is supplementary file s010, gzip-compressed tab-delimited CGM traces
#57 subjects, glucose readings every 5 min in mg/dL (verified from the extracted file)
#PLOS Biology article is open access under CC-BY 4.0
#PLOS redirects the file to a Google Cloud bucket, so download from a normal network
download_cgm () {
  local dir="${RAW}/cgm"; mkdir -p "$dir"
  echo "[cgm] Hall 2018 PLoS Biology S1 Data -> $dir"
  local url="https://journals.plos.org/plosbiology/article/file?type=supplementary&id=10.1371/journal.pbio.2005143.s010"
  $CURL -o "$dir/hall2018_cgm_S1_Data.gz" "$url"
  gunzip -f "$dir/hall2018_cgm_S1_Data.gz"          #-> hall2018_cgm_S1_Data (ASCII, tab-delimited)
  checksum "$dir/hall2018_cgm_S1_Data"
}

#imaging, RetinaMNIST (MedMNIST v2, Zenodo record 10519652)
#1,600 fundus images, 5-class DR grade, two resolutions
#MedMNIST v2 is released under CC-BY 4.0
download_imaging () {
  local dir="${RAW}/imaging"; mkdir -p "$dir"
  echo "[imaging] RetinaMNIST 28px + 224px -> $dir"
  local base="https://zenodo.org/records/10519652/files"
  $CURL -o "$dir/retinamnist.npz"     "$base/retinamnist.npz?download=1"       #28x28,  ~3.3 MB
  $CURL -o "$dir/retinamnist_224.npz" "$base/retinamnist_224.npz?download=1"   #224x224, ~128 MB
  checksum "$dir/retinamnist.npz"
  checksum "$dir/retinamnist_224.npz"
}

#wearable, Li et al. 2017 PLoS Biology (doi 10.1371/journal.pbio.2001402)
#Basis-watch biosensor data, 43 subjects, 20.4 GB tar of per-subject zips
#the paper's hmpdacc.org/data/wearable/stanford.tar URL no longer serves the tar
#(the HMP DACC site was rebuilt; that path is not retrievable), so use the live
#Stanford iPOP host below. Large file so -C - resume matters
#PLOS Biology article is open access under CC-BY 4.0
download_wearable () {
  local dir="${RAW}/wearable"; mkdir -p "$dir"
  echo "[wearable] Li 2017 Stanford wearable tar (~20 GB) -> $dir"
  local url="http://ipop-data.stanford.edu/wearable_data/Stanford_Wearables_data.tar"
  $CURL -o "$dir/Stanford_Wearables_data.tar" "$url"
  checksum "$dir/Stanford_Wearables_data.tar"
  echo "  tip: 'tar -xf Stanford_Wearables_data.tar wearables/Subject1_rawdata.zip' to extract one subject"
}

#dispatcher
main () {
  local sources=("$@")
  if [ ${#sources[@]} -eq 0 ]; then
    sources=(omics cgm imaging wearable)
  fi
  for s in "${sources[@]}"; do
    case "$s" in
      omics)    download_omics ;;
      cgm)      download_cgm ;;
      imaging)  download_imaging ;;
      wearable) download_wearable ;;
      *) echo "unknown source: $s (expected: omics|cgm|imaging|wearable)" >&2; exit 1 ;;
    esac
  done
  echo "done."
}

main "$@"

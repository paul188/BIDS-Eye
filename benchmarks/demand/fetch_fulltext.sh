#!/bin/sh
# Parallel, resumable download of Europe PMC full-text XML (4 at a time).
# usage: fetch_fulltext.sh work/pmcids.txt work/epmc_xml
# Prints "<PMCID> <http code>" for every paper without full text.
ids="$1"; dir="$2"
xargs -P 4 -I{} sh -c '
  id="$1"; dir="$2"
  [ -s "$dir/$id.xml" ] && exit 0
  code=$(curl -s --retry 3 --max-time 90 -A "dataset-catalogue-enrichment/0.1" -o "$dir/$id.part" -w "%{http_code}" "https://www.ebi.ac.uk/europepmc/webservices/rest/$id/fullTextXML")
  if [ "$code" = "200" ]; then mv "$dir/$id.part" "$dir/$id.xml"; else rm -f "$dir/$id.part"; echo "$id $code"; fi
' _ {} "$dir" < "$ids"

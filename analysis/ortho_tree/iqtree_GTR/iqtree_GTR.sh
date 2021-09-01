# Run IQ-TREE on meta alignments using GTR model

if [ ! -d out/${1}/ ]; then
  mkdir -p out/${1}/
fi

../../../bin/iqtree -s ../make_metaNT/out/${1}/${2}.fasta -m GTR+FO${3}+G -pre out/${1}/${2} -quiet

# DEPENDENCIES
# ../make_metaNT/make_metaNT.py
#   ../make_metaNT/out/*/*.phy
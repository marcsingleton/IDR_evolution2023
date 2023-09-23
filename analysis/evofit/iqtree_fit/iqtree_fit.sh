# Run IQ-TREE on meta alignments and fit the rate matrix

if [ ! -d out/ ]; then
  mkdir out/
fi

../../../bin/iqtree -s ../iqtree_meta/out/${1}.afa -m GTR20+FO+I+R -pre out/${1} -nt ${2} -quiet

# Generate cell GEX matrix

## cellranger count
First, to pull data from tape, see `filetransfer.sh`. Then double check that all the files are there, e.g.

```
count=1
fastq_path=~/scratch60/200709/gpfs/ycga/sequencers/pacbio/gw92/10x/Single_Cell/20200707_llt26/H5YNJDSXY/outs/fastq_path/H5YNJDSXY/
for file in ${fastq_path}*; do printf '%d: %s\n\n' "$count" "$file"; (( count++)); done
```

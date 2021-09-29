# Generate cell GEX matrix

## cellranger count

First, to pull data from tape, run `filetransfer.sh` after adding filenames to pull from the `Single_Cell` backup. Then transfer the fastq files per sample to a more convenient location. 

To check that all samples are there, run:
```
count=1
fastq_path=~/scratch60/200709/gpfs/ycga/sequencers/pacbio/gw92/10x/Single_Cell/20200707_llt26/H5YNJDSXY/outs/fastq_path/H5YNJDSXY/
for file in ${fastq_path}*; do printf '%d: %s\n\n' "$count" "$file"; (( count++)); done
```

Usually the files generated by the sequencer have been de-multiplexed and sorted into fastq files per sample before storing the backup. I.e., `cellranger mkfastq` has already been run, generating 3x fastq files per sample. See [10x tutorial](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/using/mkfastq) naming scheme for an example.

For a folder with the structure:
```
./fastq_folder
|___ sample_id
     |___ sample_id_L1.fastqz
.
.
.
```

Run a job for `cellranger count` per sample by specifying `fastq2cellranger.sh` variables. 


#!/bin/bash

# #### REFERENCES
# gw92 s
MMT="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/refdata-cellranger-mm10-3.0.0/"
MNT="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/mm10-3.0.0_GFP_premrna/"
HHT="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/refdata-cellranger-GRCh38-3.0.0/"
HNT="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/GRCh38-3.0.0.premrna/"
HMT="/ycga-gpfs/apps/bioinfo/genomes/10xgenomics/refdata-cellranger-hg19_and_mm10-1.2.0"
RMT="/gpfs/ycga/apps/bioinfo/genomes/10xgenomics/refdata-cellranger-macaca-mulatta/macaca_mulatta"
GMT="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/mm10-3.0.0_GFP/"
CHK="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/Gallus_gallus_NCBI_build3.1/gallus_ncbi_3.1"
TBB="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/TBB927"
TCO="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/TcongolenseIL3000/"
DRT="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/Danio_rerio.GRCz11"
CEG="/ycga/sequencers/pacbio/gw92/10x/reference/c_elegans_ws268/"
HRV="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/hg19_HRV/"
HRP="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/GRCh38_RPVIRUS/"
HSV="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/GRCh38_HSV/"
RHT="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/Rattus_norvegicus.Rnor_6.0.95/"
VHT="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/refdata-cellranger-vdj-GRCh38-alts-ensembl-3.1.0/"
VMT="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/refdata-cellranger-vdj-GRCm38-alts-ensembl-3.1.0"
ZSG="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/mm10-3.0.0_ZSG/"
H2B="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/mm10-3.0.0_H2B/"
ZSG1="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/mm10-3.0.0_ZSG1/"
HCO="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/hum_scv2_exons/"
HCU="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/hum_scv2_exons_3utr/"
HSU="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/hum_scv2_S-3utr/"
HCR="/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/hum_scv2_orf1-10/"

# ngr 
human_fasta=/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/refdata-cellranger-GRCh38-3.0.0/fasta/genome.fa
human_gtf=/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/refdata-cellranger-GRCh38-3.0.0/genes/genes.gtf
mouse_fasta=/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/refdata-cellranger-mm10-3.0.0/fasta/genome.fa
mouse_gtf=/gpfs/ycga/sequencers/pacbio/gw92/10x/reference/refdata-cellranger-mm10-3.0.0/genes/genes.gtf
scv2_fasta=/home/ngr4/apps/references/MT020880.1.fasta
scv2_gtf=/home/ngr4/apps/references/scv2_orf1-10.gtf
hACE2_fasta=/home/ngr4/apps/references/hACE2.fa
hACE2_gtf=/home/ngr4/apps/references/hACE2.gtf
mNG_fasta=/home/ngr4/apps/references/mNG.fa
mNG_gtf=/home/ngr4/apps/references/mNG.gtf
# ####

# #### FLOW CONTROL 
# input files
fastq_path=~/scratch60/scnd/data/raw/human/fastqs/

# output files
output_dir=~/scratch60/scnd/processed/human_cellranger/
mkdir -p $output_dir

# calls
concat_refs=False
ref=$HNT
mk_cellcountmat=True
# ####

# concatenate fastas
# can declare concat_refs True if first iteration, then wait for it to finish
if [[ $concat_refs = True ]]
then 
  cat ${human_fasta} ${scv2_fasta} ${mNG_fasta} > ${output_dir}hs_scv2_mNG.fa
  cat ${human_gtf} ${scv2_gtf} ${mNG_gtf} > ${output_dir}hs_scv2_mNG.gtf
  sbatch --job-name=new_ref --time=5-00:00:00 --mem=96G -c 12 -o new_ref.sh && module load cellranger/6.0.1 && cellranger mkref --genome=hs_scv2_mNG --fasta=${output_dir}hs_scv2_mNG.fa --genes=${output_dir}hs_scv2_mNG.gtf
  printf 'done combining %s' "hs_scv2_mNG"
fi

# cellranger count
# arg1=sample_id
# arg2=transcriptome
# arg3=output_dir
count=1
if [[ $mk_cellcountmat = True ]]
then
  for sample in ${fastq_path}*
  do
    sample_id=$(basename $sample)
    sbatch --job-name="$sample_id"_cnt --output="$sample_id"_count.log ~/project/scnd/scripts/cellranger_count.sh $sample_id $ref $output_dir $fastq_path
    printf '\n... job submitted for  %s\n' $sample_id
    ((count++))
  done
fi

printf '\nExiting. Submitted %d jobs\n' $count
 
exit

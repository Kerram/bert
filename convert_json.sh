python3 convert_raw.py --data_path=bert/data/${1} --output_path=gs://zpp-bucket-1920/tpu-fine-tune/data/${2} --vocab_file=bert/model/vocab.txt --set_type=${3}

# Example usage:
# ./convert_tsv.sh train.tsv train.tf_record train

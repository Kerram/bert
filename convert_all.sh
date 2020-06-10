python3 tokenize_all_thms.py --data_path=bert/data/${1} --output_path=gs://zpp-bucket-1920/tpu-fine-tune/data/${2} --vocab_file=bert/model/vocab.txt

# Example usage:
# ./convert_all.sh thms_ls.train thms_ls.train

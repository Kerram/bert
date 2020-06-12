sudo python3 run_deephol.py --data_dir=bert/data --bert_config_file=bert/model/bert_config.json \
--vocab_file=bert/model/vocab.txt --output_dir=gs://zpp-bucket-1920/tpu-fine-tune/exported/wavenet_separate_embeddings \
--init_checkpoint=gs://zpp-bucket-1920/tpu-fine-tune/models/wavenet_separate_embeddings/model.ckpt-333000 --max_seq_length=512 \
--do_train=False --do_eval=False --do_export=True --use_tpu=False \
--test_file=gs://zpp-bucket-1920/tpu-fine-tune/data/test_without_brackets_wavenet.tf_record

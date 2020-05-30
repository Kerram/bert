sudo python3 run_deephol.py --data_dir=bert/data --bert_config_file=bert/model/bert_config.json \
--vocab_file=bert/model/vocab.txt --output_dir=gs://zpp-bucket-1920/tpu-fine-tune/exported/smaller_set \
--init_checkpoint=gs://zpp-bucket-1920/tpu-fine-tune/models/smaller_set/model.ckpt-360000 --max_seq_length=512 \
--do_train=False --do_eval=False --do_export=True --use_tpu=False \
--test_file=gs://zpp-bucket-1920/tpu-fine-tune/data/test.tf_record

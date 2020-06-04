sudo python3 run_deephol.py --data_dir=bert/data --bert_config_file=bert/model/bert_config.json \
--vocab_file=bert/model/vocab.txt --output_dir=gs://zpp-bucket-1920/tpu-fine-tune/exported/more_data_best \
--init_checkpoint=gs://zpp-bucket-1920/tpu-fine-tune/models/more_data_best/model.ckpt-360000 --max_seq_length=512 \
--do_train=False --do_eval=False --do_export=True --use_tpu=False \
--test_file=gs://zpp-bucket-1920/tpu-fine-tune/data/test.tf_record

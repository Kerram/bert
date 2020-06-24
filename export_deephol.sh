sudo python3 run_deephol.py --data_dir=bert/data --bert_config_file=bert/model/bert_config.json \
--vocab_file=bert/model/vocab.txt --output_dir=gs://zpp-bucket-1920/tpu-fine-tune/exported/params_cond_on_tac_wavenet \
--init_checkpoint=gs://zpp-bucket-1920/tpu-fine-tune/models/params_cond_on_tac_wavenet/model.ckpt-903935 --max_seq_length=1024 \
--do_train=False --do_eval=False --do_export=True --use_tpu=False \
--test_file=gs://zpp-bucket-1920/tpu-fine-tune/data/preprocessed/test_1024.tf_record

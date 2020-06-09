sudo python3 run_deephol.py --data_dir=bert/data --bert_config_file=bert/model/bert_config.json \
--vocab_file=bert/model/vocab.txt --output_dir=gs://zpp-bucket-1920/tpu-fine-tune/models/new_5_lr \
--init_checkpoint=gs://zpp-bucket-1920/bert-bucket-golkarolka/bert_model11/model.ckpt-500000 --num_train_epochs=3.0 \
--max_seq_length=512 --do_train=True --do_eval=True --do_export=False --use_tpu=True --tpu_name=bert9 \
--gcp_project=zpp-mim-1920 --tpu_zone=us-central1-f --num_tpu_cores 8 --save_checkpoints_steps 10000 \
--train_file=gs://zpp-bucket-1920/tpu-fine-tune/data/new_train.tf_record \
--eval_file=gs://zpp-bucket-1920/tpu-fine-tune/data/new_eval.tf_record

sudo python3 run_deephol.py --data_dir=bert/data \
--vocab_file=bert/model/vocab.txt --output_dir=gs://zpp-bucket-1920/tpu-fine-tune/models/wavenet_full_brackets_shorter \
--num_train_epochs=100.0 \
--max_seq_length=512 --do_train=True --do_eval=True --do_predict=False --do_export=False --use_tpu=True --tpu_name=bert3 \
--gcp_project=zpp-mim-1920 --tpu_zone=us-central1-f --num_tpu_cores 8 --save_checkpoints_steps 10000 \
--train_file=gs://zpp-bucket-1920/tpu-fine-tune/data/train_without_brackets_wavenet.tf_record \
--eval_file=gs://zpp-bucket-1920/tpu-fine-tune/data/valid_without_brackets_wavenet.tf_record
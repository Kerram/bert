sudo python3 run_deephol.py --data_dir=bert/data \
--vocab_file=bert/model/vocab.txt --output_dir=gs://zpp-bucket-1920/tpu-fine-tune/models/params_cond_on_tac_wavenet \
--num_train_epochs=100.0 \
--max_seq_length=1024 --do_train=False --do_eval=True --do_export=False --use_tpu=True --tpu_name=bert14 \
--gcp_project=zpp-mim-1920 --tpu_zone=us-central1-f --num_tpu_cores 8 \
--eval_file=gs://zpp-bucket-1920/tpu-fine-tune/data/preprocessed/valid_mini_1024.tf_record \
--infinite_eval=True

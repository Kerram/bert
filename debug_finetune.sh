sudo python3 run_deephol.py --data_dir=bert/data \
--vocab_file=bert/model/vocab.txt --output_dir=gs://zpp-bucket-1920/tpu-fine-tune/models/wavenet_separate_embeddings \
--num_train_epochs=100.0 \
--max_seq_length=512 --do_train=False --do_eval=True --do_export=False --use_tpu=True --tpu_name=bert14 \
--gcp_project=zpp-mim-1920 --tpu_zone=us-central1-f --num_tpu_cores 8 \
--eval_file=gs://zpp-bucket-1920/tpu-fine-tune/data/valid-random_without_brackets_wavenet.tf_record \
--infinite_eval=True

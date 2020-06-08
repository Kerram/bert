sudo python3 run_deephol.py --data_dir=bert/data \
--vocab_file=bert/model/vocab.txt --output_dir=gs://zpp-bucket-1920/tpu-fine-tune/models/legend/eval \
--num_train_epochs=100.0 \
--init_checkpoint=gs://zpp-bucket-1920/tpu-fine-tune/models/legend/holparam_3330000/model.ckpt-3330000
--max_seq_length=512 --do_train=False --do_eval=True --do_predict=False --do_export=False --use_tpu=True --tpu_name=bert3 \
--gcp_project=zpp-mim-1920 --tpu_zone=us-central1-f --num_tpu_cores 8 --save_checkpoints_steps 10000 \
--eval_file=gs://zpp-bucket-1920/tpu-fine-tune/data/valid_without_brackets_wavenet.tf_record
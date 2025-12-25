python3 -m main det \
 --data_root ./VOC_dataset/ \
 --out_dir runs/det_full_train_c4_setup_head_only \
 --max_iters 24000 \
 --eval_every 2000 \
 --log_every 10 \
 --batch_size 8 \
 --lr 0.01 \
 --bt_ckpt hub \
 --trainable_backbone_layers 0 \
 --train_fpn 0 \
 --det_backbone c4 \
 --device cuda:3


source venv/bin/activate
cd bt 
 python3 -m main det \
 --data_root ./VOC_dataset/ \
 --out_dir runs/det_full_train_c4_setup_fullfinetune_b16_lr2_miters48_test0 \
 --max_iters 24000 \
 --eval_every 2000 \
 --log_every 10 \
 --batch_size 8 \
 --lr 0.02 \
 --bt_ckpt hub \
 --trainable_backbone_layers 5 \
 --train_fpn 0 \
 --det_backbone c4 \
 --device cuda:2 \
 --accum_steps 2 

 tensorboard --logdir runs --port 6007




curl -L -o ~/bt/voc/pascal-voc-2012.zip\
  https://www.kaggle.com/api/v1/datasets/download/huanghanchina/pascal-voc-2012

curl -L -o ~/bt/voc/pascal-voc-data-2007.zip\
  https://www.kaggle.com/api/v1/datasets/download/ahmedsharaf09/pascal-voc-data-2007

curl -L -o ~/bt/voc0712.zip\
  https://www.kaggle.com/api/v1/datasets/download/bardiaardakanian/voc0712
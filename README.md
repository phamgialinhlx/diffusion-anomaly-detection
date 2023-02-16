# Diffusion Models for Medical Anomaly Detection

Đây là mã nguồn cho NCKH: ỨNG DỤNG MÔ HÌNH KHUẾCH TÁN TRONG PHÂN VÙNG ẢNH Y TẾ

## Sinh viên
| Họ và tên     | Mã sinh viên |
| ------------- | ------------ |
| [Phạm Tiến Du](https://github.com/dupham2206)  | 20020039     |
| [Phạm Gia Linh](https://github.com/phamgialinhlx) | 20020203     |
| [Trịnh Ngọc Huỳnh](https://github.com/huynhspm) | 20020054     |


## Dữ liệu

Bộ dữ liệu được sử dụng là LIDC-IDRI. LIDC-IDRI là một bộ dữ liệu y tế được sử dụng để nghiên cứu về tìm kiếm bệnh lý phổi. Nó bao gồm hàng trăm nghìn hình ảnh CT ghi lại từ các bệnh nhân mạng bệnh tật về phổi. Bộ dữ liệu này đã được sử dụng trong nhiều nghiên cứu về phát hiện và phân loại các bệnh lý phổi dựa trên hình ảnh CT, bao gồm cả các nghiên cứu sử dụng các mô hình deep learning. Bộ dữ liệu này là một tài nguyên quan trọng cho các nhà nghiên cứu trong lĩnh vực y tế và AI, vì nó cung cấp một số lượng lớn dữ liệu để hỗ trợ nghiên cứu và phát triển các mô hình AI.

## Huấn luyện 

Ta đặt các flags như sau:
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True"
```
Huấn luyện mô hình phân lớp
```
python scripts/classifier_train.py --data_dir path_to_traindata --dataset brats_or_chexpert $TRAIN_FLAGS $CLASSIFIER_FLAGS
```
<!-- V100
```
python scripts/classifier_train.py --data_dir /home/pill/lung/LIDC_IDRI_preprocessing/config_data/train.csv --dataset LIDC --lr 1e-4 --batch_size 10 --image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True
``` -->
Huấn luyện mô hình khuếch tán
```
python scripts/image_train.py --data_dir --data_dir path_to_traindata --datasaet brats_or_chexpert  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

Mô hình sẽ được lưu ở thư mục *results*.

Quá trình chuyển ảnh sang ảnh khỏe mạnh
```
python scripts/classifier_sample_known.py  --data_dir path_to_testdata  --model_path ./results/model.pt --classifier_path ./results/classifier.pt --dataset brats_or_chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS  $SAMPLE_FLAGS 
```

Trực quan hóa quá trình lấy mẫu sử dụng [Visdom](https://github.com/fossasia/visdom).




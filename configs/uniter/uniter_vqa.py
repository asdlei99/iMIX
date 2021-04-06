_base_ = [
    '../_base_/models/uniter_config.py',
    '../_base_/datasets/vqa_dataset.py',
    '../_base_/schedules/schedule_vqa.py',
    '../_base_/default_runtime.py'
]  # yapf:disable

dataset_type = 'VQADATASET'
data_root = '/home/datasets/mix_data/mmf/'
feature_path = 'data/datasets/vqa2/grid_features/features/'
annotation_path = 'data/datasets/vqa2/grid_features/annotations/'
feature_default_path = 'data/datasets/vqa2/defaults/features/'
feature_global_path = 'data/datasets/vqa2/defaults/resnet152/'
annotation_default_path = 'data/datasets/vqa2/defaults/annotations/'
vocab_path = 'data/datasets/vqa2/defaults/extras/vocabs/'

vqa_reader_train_cfg = dict(
    mix_annotations=dict(
        train=data_root + annotation_default_path + 'uniter_imdb_train2014.npy',
        val=data_root + annotation_default_path + 'uniter_imdb_val2014.npy',
        test=data_root + annotation_default_path + 'imdb_test2015.npy',
        minival=data_root + annotation_default_path + 'uniter_imdb_minival2014.npy',
        train_coco10pc=data_root + annotation_default_path + 'imdb_train2014_len_coco_10_pc.npy',
        train_coco50pc=data_root + annotation_default_path + 'imdb_train2014_len_coco_50_pc.npy',
        valminusminival=data_root + annotation_default_path + 'imdb_valminusminival2014.npy',
    ))

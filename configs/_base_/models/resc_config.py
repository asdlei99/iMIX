# model settings
model = dict(
    type='ReSC',
    encoder=dict(
        type='DarknetEncoder',
        config_path='/home/zrz/code/ReSC/model/yolov3.cfg',
        img_size=416,
        obj_out=False),
    backbone=dict(
        type='ReSC_BACKBONE',
        emb_size=512,
        jemb_drop_out=0.1,
        NFilm=3,
        fusion='prod',
        intmd=False,
        mstage=False,
        convlstm=False,
        leaky=False),
    weights_file='/home/zrz/code/ReSC/saved_models/yolov3.weights')

loss = [dict(type='YOLOLossV2'), dict(type='DiverseLoss')]

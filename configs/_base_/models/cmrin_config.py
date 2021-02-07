# model settings
model = dict(
    type='CMRIN',
    encoder=dict(
        type='DarknetEncoder',
        config_path='/home/zrz/code/ReSC/model/yolov3.cfg',
        img_size=416,
        obj_out=False),
    backbone=dict(
        type='CMRIN_BACKBONE',
        emb_size=512,
        jemb_drop_out=0.1,
        NFilm=2,
        light=False,
        coordmap=True,
        intmd=False,
        mstage=False,
        convlstm=False,
        tunebert=True,
        leaky=False),
    weights_file='/home/zrz/code/ReSC/saved_models/yolov3.weights')

loss = dict(type='YOLOLoss')

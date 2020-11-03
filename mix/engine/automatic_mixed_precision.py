# from torch.cuda.amp.autocast_mode import autocast
# from mix.utils.comm import get_world_size
# #from mix.engine.organizer import is_mixed_precision
# import mix.engine.organizer as organizer
#
#
# def mixed_precision(func):
#     @autocast(enabled=True)
#     def warpper(*args, **kwargs):
#         return func(*args, **kwargs)
#
#     return warpper
#
#
# def is_multi_gpus_mixed_precision():
#     if get_world_size() > 1 and organizer.is_mixed_precision():
#         return True
#     else:
#         return False

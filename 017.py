#tf基础: GPU/CPU硬件信息
import tensorflow as tf

#获得当前GPU硬件情况
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#获得当前CPU硬件情况
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')

if len(gpus)==0:
    print("No GPU found!")
else:
    print(gpus)
    
print(cpus)

#通过tf.config.experimental.set_visible_devices设置可见的设备范围，表示需要使用的设备
#这里表示使用0和1两块显卡, 这等价于
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
tf.config.experimental.set_visible_devices(devices=gpus[0:2], device_type='GPU')

#以下代码表示仅仅在需要显存的时候才分配显存
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

#以下代码表示在GPU0上，使用1G显存
if len(gpus)>0:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])    

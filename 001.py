#ts矩阵入门
import tensorflow as tf
A = tf.constant([[1,2],[3,4]], dtype=tf.float32)
B = tf.constant([[5,6],[7,8]], dtype=tf.float32)
C = tf.add(A,B)
print(C)
D = tf.matmul(A,B)
print(D)

print()
print("shape=",A.shape, "\n")
print("type=",A.dtype, "\n")
print("value=",A.numpy())  	#numpy返回A的值
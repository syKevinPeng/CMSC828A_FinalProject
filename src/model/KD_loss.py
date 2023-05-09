import tensorflow as tf
from inspect import currentframe, getframeinfo
from utils.utils import get_logger
import numpy as np

def my_kd_loss( y_true, y_pred, inputs, teacher_model, temperature=1.0, alpha=0.1):
    task_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    teacher_pred = teacher_model(inputs, training=False)
    
    student_prob = tf.nn.softmax(y_pred / temperature, axis=-1)    
    teacher_prob = tf.nn.softmax(teacher_pred / temperature, axis=-1)
    
    kd_loss = tf.keras.losses.categorical_crossentropy(teacher_prob, student_prob, from_logits=False)
    kd_loss *= alpha * temperature**2    

    total_loss = task_loss + kd_loss    
    return total_loss

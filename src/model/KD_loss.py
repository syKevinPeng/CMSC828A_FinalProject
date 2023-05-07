import tensorflow as tf


class KDLoss(tf.keras.losses.Loss):
    def __init__(self, teacher_model, temperature=1.0, alpha=0.1, **kwargs):
        super(KDLoss, self).__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        # Ensure the teacher model is not trainable
        self.teacher_model.trainable = False

    def call(self, y_true, student_pred,inputs):
        task_loss = tf.keras.losses.categorical_crossentropy(y_true, student_pred)

        teacher_pred = self.teacher_model(inputs, training=False)
        student_prob = tf.nn.softmax(student_pred / self.temperature, axis=-1)
        teacher_prob = tf.nn.softmax(teacher_pred / self.temperature, axis=-1)
        kd_loss = tf.keras.losses.categorical_crossentropy(teacher_prob, student_prob, from_logits=False)
        kd_loss *= self.alpha * self.temperature**2

        total_loss = task_loss + kd_loss
        return total_loss

    def __call__(self, y_true, y_pred,inputs, **kwargs):
        return super(KDLoss, self).__call__(y_true, y_pred, inputs=inputs, **kwargs)

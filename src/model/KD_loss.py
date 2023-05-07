import tensorflow as tf

def kd_loss(student_logits, teacher_logits, temperature=1.0, alpha=0.1):
    """
    Compute the knowledge distillation loss.
    
    Args:
        student_logits (tf.Tensor): Logits from the student model.
        teacher_logits (tf.Tensor): Logits from the teacher model.
        temperature (float, optional): Temperature for softmax. Defaults to 1.0.
        alpha (float, optional): Weight for the KD loss. Defaults to 0.1.
        
    Returns:
        The KD loss.
    """
    # Apply temperature to logits and compute softmax
    student_probs = tf.nn.softmax(student_logits / temperature)
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)

    # Compute KL divergence between teacher and student distributions
    kl_div = tf.keras.losses.KLDivergence()(teacher_probs, student_probs)
    
    # Scale the KL divergence by the temperature squared and alpha
    scaled_kl_div = alpha * kl_div * (temperature ** 2)
    
    return scaled_kl_div

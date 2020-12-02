import tensorflow as tf

def parse_aug_fn(dataset):
    """
    Image Augmentation
    """
    x = tf.cast(dataset['image'], tf.float32) / 255. 
    x = flip(x)  

    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: color(x), lambda: x)
 
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: rotate(x), lambda: x)

    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(x), lambda: x)
 
    y = tf.one_hot(dataset['label'], 10)
    return x, y


def parse_fn(dataset):
    x = tf.cast(dataset['image'], tf.float32) / 255. 
 
    y = tf.one_hot(dataset['label'], 10)
    return x, y


def flip(x):
    """
    flip image
    """
    x = tf.image.random_flip_left_right(x)  
    return x


def color(x):
    """
     Color change
    """
    x = tf.image.random_hue(x, 0.08)  # 
    x = tf.image.random_saturation(x, 0.6, 1.6)  
    x = tf.image.random_brightness(x, 0.05)  
    x = tf.image.random_contrast(x, 0.7, 1.3)  
    return x


def rotate(x):
    """
    Rotation image
    """

    x = tf.image.rot90(x,tf.random.uniform(shape=[],minval=1,maxval=4,dtype=tf.int32))
    return x


def zoom(x, scale_min=0.6, scale_max=1.4):
    """
    Zoom Image
    """
    h, w, c = x.shape
    scale = tf.random.uniform([], scale_min, scale_max) 
    sh = h * scale  
    sw = w * scale  
    x = tf.image.resize(x, (sh, sw)) 
    x = tf.image.resize_with_crop_or_pad(x, h, w) 
    return x

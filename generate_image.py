import torch
import tensorflow as tf
import tensorflow_hub as hub

from diffusers import StableDiffusionPipeline

class ImageGenerator():
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16, safety_checker=None)
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
    
    # PyTorch implementation of Stable Diffusion v1.5
    def get_diffusion_output(self, prompt, guidance=7, steps=40, count=4):
        return self.pipe(
            prompt=prompt,
            guidance_scale=guidance,
            num_images_per_prompt=count,
            num_inference_steps=steps
        ).images

    # TensorFlow pretrained Neural Style Transfer model
    def get_merged_output(self, image, style, weight=1000000, steps=300):
        content_img = self.load_image(image)
        style_img = self.load_image(style)
        merged_img = self.model(
            tf.constant(content_img), tf.constant(style_img))[0]
        
        return self.unload_image(merged_img)

    def load_image(self, image):
        image = tf.keras.utils.img_to_array(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        
        shape = tf.cast(tf.shape(image)[:-1], tf.float32)
        scale = 512 / max(shape)

        new_shape = tf.cast(shape * scale, tf.int32)

        image = tf.image.resize(image, new_shape)
        image = image[tf.newaxis, :]
        return image / 255
    
    def unload_image(self, image):
        image = tf.squeeze(image, axis=0)
        return tf.keras.utils.array_to_img(image)
## Rendering Megascans

You need to download an environment map (e.g., [this one](https://polyhaven.com/a/studio_small_09)), and save it.
Then put`/path/to/my_environment_map.exr` in the config.

To render the megascans, obtain the necessary models from the [website](https://quixel.com/megascans/home), convert them into GLTF, create a `enb.blend` Blender environment, and then run:
```
blender --python render_dataset.py rendering.blend --background
```
The rendering config is located in `render_dataset.py`.

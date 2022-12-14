"""
This script is intended to render Megascans, exported in the gltf format.
It can be used to render other gltf models as well.

It was written with the use of the following resources:
- original rendering script from https://github.com/bmild/nerf

- https://blenderartists.org/t/render-settings-for-depth-normal-albedo-in-2-80/1199454
- https://blender.stackexchange.com/questions/42579/render-depth-map-to-image-with-python-script

We used Blender 2.93.10 (hash 0a65e1a8e7a9 built 2022-08-02 23:34:42)
"""

#----------------------------------------------------------------------------

# TODO: The command below might fail to work
# You might need to install those manually in a *very* annoying way:
# /Applications/Blender\ 2.app/Contents/Resources/2.93/python/bin/pip install -t /Applications/Blender\ 2.app/Contents/Resources/2.93/python/lib/python3.9 hydra-core
# import pip; pip.main(['install', 'tqdm', 'hydra-core', '--user'])

#----------------------------------------------------------------------------

import sys
import os
import json
import math
import argparse
from typing import List, Dict, Any, Optional

import bpy
from mathutils import Vector

import numpy as np
try:
    from tqdm import tqdm
except:
    tqdm = lambda x, *args, **kwargs: x

render_layers = None
depth_file_output = None
normal_file_output = None

#----------------------------------------------------------------------------

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

#----------------------------------------------------------------------------

cfg = EasyDict({
    'num_views': 128,
    'random': True,
    'resolution': (1024, 1024), # Assuming square images
    # 'collection_path': '/path/to/plants/plants/vfelafzia_Coriander', # Render a single collection
    'collections_dir': '/path/to/megascans/glb-models/plants_collections', # Render a directory of collections
    # 'collections_skip_up_to': 'vfelafzia_Coriander', # Skip collections until we encounter this one
    'scale_wrt_collection': False, # Should we scale objects with respect to their collection
    'output_path': '/path/where/to/save/rendered/data',
    'color_depth': 8, # TODO: what's that?
    'camera': EasyDict({
        # 'fov': EasyDict({'dist': 'uniform', 'min': np.deg2rad(20.0), 'max': np.deg2rad(40.0)}),
        'fov': EasyDict({'dist': 'normal', 'mean': np.pi / 4, 'std': 0.0}),
        'radius': 3.5,
    }),
    'device': 'gpu',
    'environment_texture_path': '/path/to/lighting.exr',
    'model_ext': '.glb',
    'small_objects_filter_thresh': 2.0, # Remove objects which are smaller than `small_objects_filter_thresh` in any of the dimensions
})

#----------------------------------------------------------------------------

def initialize(cfg):
    global render_layers, depth_file_output, normal_file_output

    np.random.seed(42)
    os.makedirs(cfg.output_path, exist_ok=True)

    # Render Optimizations
    scene = bpy.context.scene
    scene.render.use_persistent_data = True

    # Set up rendering of depth map.
    scene.use_nodes = True

    # Depth map output
    render_layers = scene.node_tree.nodes.new('CompositorNodeRLayers')
    depth_file_output = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = ''
    depth_file_output.format.file_format = "OPEN_EXR"
    scene.node_tree.links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

    # Normal map output
    normal_file_output = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.base_path = ''
    normal_file_output.format.file_format = "PNG"
    scene.node_tree.links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

    # Add passes for additionally dumping albedo and normals.
    scene.view_layers["View Layer"].use_pass_normal = True
    scene.render.image_settings.color_depth = str(cfg.color_depth)

    # Background
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = True

    scene.render.resolution_x = cfg.resolution[0]
    scene.render.resolution_y = cfg.resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    # Setting the GPU device
    if cfg.device == 'gpu':
        bpy.data.scenes[0].render.engine = "CYCLES"

        # Set the device_type
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA" # or "OPENCL"

        # Set the device and feature set
        bpy.context.scene.cycles.device = "GPU"

        # get_devices() to let Blender detects GPU device
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d["use"] = 1 # Using all devices, include GPU and CPU
            print(d["name"], d["use"])

    if not cfg.environment_texture_path is None:
        print('Adding the environment texture')
        C = bpy.context
        world = C.scene.world
        world.use_nodes = True
        enode = C.scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        enode.image = bpy.data.images.load(cfg.environment_texture_path)
        node_tree = C.scene.world.node_tree
        node_tree.links.new(enode.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])

#----------------------------------------------------------------------------

def run(cfg):
    if 'collection_path' in cfg:
        assert not 'collections_dir' in cfg, f"Cant handle both single collection and multi-collection: {cfg.collection_path} and {cfg.collections_dir}"
        collection_dirs = [cfg.collection_path]
    else:
        collection_dirs = [d for d in listdir_full_paths(cfg.collections_dir) if os.path.isdir(d)]
        print(f'Found {len(collection_dirs)} collections to render')

        if 'collections_skip_up_to' in cfg:
            pivot_col_idx = [i for i, d in enumerate(collection_dirs) if os.path.basename(d) == cfg.collections_skip_up_to][0]
            collection_dirs = collection_dirs[pivot_col_idx:]
            print(f'Will render {len(collection_dirs)} collections (skipped up to {cfg.collections_skip_up_to})')

    # camera_angles = generate_camera_angles(cfg.num_views)
    # Uncomment this section for debugging with some simple object
    # camera_angles = generate_camera_angles(cfg.num_views)
    # generate_object_renderings(
    #     obj=get_object('Cube'),
    #     save_dir='/Users/universome/Downloads/tmp/cube',
    #     camera_angles=camera_angles,
    #     radius=cfg.camera.radius,
    # )

    for collection_dir in collection_dirs:
        print('<===============================>')
        print(f'Processing collection: {collection_dir}')
        print('<===============================>')
        render_collection(cfg, collection_dir)

#----------------------------------------------------------------------------

def render_collection(cfg: EasyDict, collection_dir: os.PathLike):
    # Make sure that the state is correct
    current_objects = bpy.context.scene.objects
    # if len(current_objects) != 2:
    #     print('Found the following objects:', [f'{o.name} ({o.type})' for o in bpy.context.scene.objects])
    #     raise Exception('There should be only 2 objects (camera and light) in the scene.')
    # else:
    #     assert {o.type for o in current_objects} == {'CAMERA', 'LIGHT'}, \
    #         f"The object should be a camera + light, but found {[o.type for o in current_objects]}."
    if len(current_objects) != 1:
        for o in current_objects:
            if not o.type in ['CAMERA']:
                print(f'Deleting object {o.name} of type {o.type}')
                delete_object(o)
    current_objects = bpy.context.scene.objects
    assert len(current_objects) == 1, f"There should only be the camera object, but found: {[o.type for o in current_objects]}"

    collection_name: str = os.path.basename(collection_dir).replace(' ', '_')
    objects: List[obj.types.Object] = load_collection(collection_dir, ext=cfg.model_ext)
    assert len(objects) > 0, f"The collection is empty: {collection_dir}"
    print(f'Loaded collection:', [f'{o.name} ({o.type})' for o in objects])

    # Set the camera into the "canonic" position
    init_camera(np.pi / 4, cfg.camera.radius)

    # Scale the collection
    # scales = [get_max_object_size(o) for o in objects]
    # print('Object scales before:', [compute_object_scale(o) for o in objects])
    scales = [max(o.dimensions) for o in objects]
    max_scale = max(scales)
    for o, s in zip(objects, scales):
        target_scale = 2.0 / (max_scale if cfg.scale_wrt_collection else s)
        normalize_object(o, scale=target_scale)

    # Call the update function: otherwise obj.dimensions are not updated (sic!).
    bpy.context.view_layer.update()

    print('Num objects before filtering:', len(objects))
    # Remove the objects which are too tiny
    objects = filter_small_objects(objects, cfg.small_objects_filter_thresh)
    print('Num objects after filtering:', len(objects))

    if len(objects) == 0:
        print('Nothing to render: all objects are too small!')

    # Render each object separately
    metadata = {}
    for o in objects:
        metadata[o.name] = generate_object_renderings(
            obj=o,
            save_dir=os.path.join(cfg.output_path, collection_name, o.name.replace(' ', '_')),
            root_dir=cfg.output_path,
            camera_angles=generate_camera_angles(cfg.num_views, cfg.random),
            fovs=sample_bounded_scalar(cfg.camera.fov, cfg.num_views),
            radius=cfg.camera.radius,
        )

    # Save the metadata
    os.makedirs(os.path.join(cfg.output_path, collection_name), exist_ok=True)
    metadata_save_path = os.path.join(cfg.output_path, collection_name, 'metadata.json')
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Clean the scene
    for o in objects:
        delete_object(o)

#----------------------------------------------------------------------------

def generate_object_renderings(obj: bpy.types.Object, save_dir: os.PathLike, camera_angles: np.ndarray, fovs: np.ndarray, radius: float, root_dir: os.PathLike) -> Dict[str, List[float]]:
    """
    Generates a set of renderings for a given object.
    """
    metadata = []
    show_object_hide_others(obj)
    os.makedirs(save_dir, exist_ok=True)

    for i, curr_angles in tqdm(enumerate(camera_angles), desc=f'Rendering {obj.name}'):
        curr_save_path = os.path.join(save_dir, f'{i:06d}')
        frame_data = render_current_view(curr_save_path, curr_angles, fovs[i], radius, root_dir)
        metadata.append(frame_data)

    return metadata

#----------------------------------------------------------------------------

def render_current_view(save_path: os.PathLike, angles: np.ndarray, fov: float, radius: float, root_dir: os.PathLike) -> Dict:
    """
    This function renders a single frame and returns the frame data.
    """
    rotate_camera(*angles, fov=fov, radius=radius)
    bpy.context.scene.render.filepath = save_path
    depth_file_output.file_slots[0].path = save_path + "_depth_"
    normal_file_output.file_slots[0].path = save_path + "_normal_"
    bpy.ops.render.render(write_still=True)  # render still

    frame_data = {
        'file_path': os.path.relpath(save_path, start=root_dir),
        'transform_matrix': listify_matrix(bpy.context.scene.objects['Camera'].matrix_world),
        'camera_angles': angles.tolist(),
        'camera_radius': radius,
        'fov': fov,
    }

    return frame_data

#----------------------------------------------------------------------------

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

#----------------------------------------------------------------------------

def filter_small_objects(objects: List[bpy.types.Object], threshold: float=0.0) -> List[bpy.types.Object]:
    if threshold == 0.0:
        return objects

    new_objects = []

    for o in objects:
        if np.prod(o.dimensions) < threshold:
            print(f' - Removing {o.name} [{o.type}] because it is too small: max dimension {np.prod(o.dimensions):.04f} < {threshold}')
            delete_object(o)
        else:
            new_objects.append(o)

    return new_objects

#----------------------------------------------------------------------------

def compute_object_scale(obj: bpy.types.Object) -> float:
    min_c = np.array([1000000000000.0, 1000000000000.0, 1000000000000.0])
    max_c = -min_c

    for v in obj.data.vertices:
        c = obj.matrix_world @ v.co
        min_c = np.stack([min_c, c], axis=1).min(axis=1)
        max_c = np.stack([max_c, c], axis=1).max(axis=1)

    return (max_c - min_c).max()

#----------------------------------------------------------------------------

def normalize_object(obj: bpy.types.Object, scale: float=None):
    """
    Scales the object and returns the original maximum dimension of the mesh.
    """
    # Scaling the object
    if scale is None:
        # Scaling to a unit cube
        scale = 1.0 / max(obj.dimensions)
        assert scale > 0, "Object has zero dimensions"

    obj.scale *= scale

    # Shifting the center to (0, 0, 0)
    v_coords = np.array([vertex.co for vertex in obj.data.vertices]) # [num_coords, 3]
    center = (np.max(v_coords, axis=0) + np.min(v_coords, axis=0)) / 2.0 # [3]
    center = Vector(center) # [3]

    for v in obj.data.vertices:
        v.co -= center # Set all coordinates start from (0, 0, 0)

#----------------------------------------------------------------------------

def hide_object_by_name(object_name: str):
    get_object(object_name).hide_render = True

#----------------------------------------------------------------------------

def show_object_by_name(object_name: str):
    get_object(object_name).hide_render = False

#----------------------------------------------------------------------------

def get_object(obj_name: str) -> Optional[bpy.types.Object]:
    objects_with_name = [o for o in get_all_objects() if o.name == obj_name]
    if len(objects_with_name) > 1:
        raise ValueError(f'There are {len(objects_with_name)} > 1 instances of {obj_name}')
    return None if len(objects_with_name) == 0 else objects_with_name[0]

#----------------------------------------------------------------------------

def get_all_objects(filter_type: str=None) -> List[bpy.types.Object]:
    return [o for o in bpy.data.objects if (filter_type is None or o.type == filter_type)]

#----------------------------------------------------------------------------

def show_object_hide_others(obj: bpy.types.Object):
    """
    Displays only the object with the given name and hides all other objects.
    """
    for o in get_all_objects("MESH"):
        o.hide_render = True
    obj.hide_render = False

#----------------------------------------------------------------------------

def import_gltf(path: os.PathLike):
    assert os.path.isfile(path), f"{path} is not a file"
    bpy.ops.import_scene.gltf(filepath=path, loglevel=1000)

#----------------------------------------------------------------------------

def load_collection(collection_dir: os.PathLike, ext='.gltf') -> List[bpy.types.Object]:
    """
    Imports all the meshes of LOD0 in the given collection.
    Returns the names of the loaded meshes.
    """
    assert os.path.isdir(collection_dir), f"{collection_dir} is not a directory"
    file_paths = [os.path.join(collection_dir, f) for f in sorted(os.listdir(collection_dir)) if file_ext(f) == ext]
    lod0_file_paths = [f for f in file_paths if 'lod0' in f.lower()]
    loaded_objects = []

    for f in tqdm(lod0_file_paths, desc=f'Loading the collection {collection_dir}'):
        curr_objects = [o for o in get_all_objects()]
        import_gltf(f)
        new_objects = [o for o in get_all_objects() if not o in curr_objects]
        # assert len(new_objects) == 2, f"Only a new object + dummy node should be imported, but got {[f'{o.name} ({o.type})' for o in new_objects]}."
        dummy_objects = [o for o in new_objects if o.name.startswith('Node') or o.type == 'EMPTY']
        for dummy_obj in dummy_objects:
            delete_object(dummy_obj)
        new_objects = [o for o in new_objects if not o in dummy_objects]
        loaded_objects.extend(new_objects)

    # Just in case, clean empty objects
    clean_empty_objects()

    return loaded_objects

#----------------------------------------------------------------------------

def delete_object(obj: bpy.types.Object):
    bpy.data.objects.remove(obj, do_unlink=True)

#----------------------------------------------------------------------------

def delete_object_by_name(object_name: str) -> int:
    """
    Deletes the active object.
    Returns the number of deleted objects.
    """
    curr_objects = get_all_objects()
    bpy.data.objects.remove(get_object(object_name), do_unlink=True)
    return len(get_all_objects()) - len(curr_objects)

#----------------------------------------------------------------------------

def clean_empty_objects():
    for o in get_all_objects("EMPTY"):
        delete_object(o)

#----------------------------------------------------------------------------

def generate_camera_angles(num_views: int, random: bool=True) -> np.ndarray:
    """
    Generates view directions (in terms of euler angles) for the given number of shots.
    - yaw in radians (-pi, pi)
    - pitch in radians (0, pi)
    - roll is not used (all zeros).
    """
    if random:
        yaw = np.random.rand(num_views) * 2 * np.pi - np.pi  # [num_views]
        pitch = np.arccos(1 - 2 * np.random.rand(num_views)) # [num_views]
        roll = np.zeros(pitch.shape) # [num_views]
    else:
        assert num_views == 129, f"Can only render 129 views deterministically... Adjust the number of steps manually"
        num_steps = 15
        pitch = np.linspace(1e-6, np.pi - 1e-6, num=num_steps, dtype=np.float32) # [num_steps]
        yaw = [np.linspace(0, 2 * np.pi, num=max(int(np.sin(p) * num_steps), 1), dtype=np.float32) for p in pitch] # (num_steps, [num_views_per_step])
        pitch = np.array([p for i, p in enumerate(pitch) for _ in yaw[i]]) # [num_views]
        yaw = np.array([y for ys in yaw for y in ys]) # [num_views]
        pitch = np.clip(pitch, 1e-8, np.pi - 1e-8) # [num_views]
        roll = np.zeros(yaw.shape, dtype=np.float32) # [num_views]

    angles = np.stack([yaw, pitch, roll], axis=1) # [num_views, 3]
    assert angles.shape == (num_views, 3), f"Wrong shape: {angles.shape}"

    return angles

#----------------------------------------------------------------------------

def file_ext(path: os.PathLike) -> str:
    return os.path.splitext(path)[1].lower()

#----------------------------------------------------------------------------

def init_camera(fov: float, radius: float):
    """
    Sets FOV and radius for the camera
    """
    cam = bpy.data.objects['Camera']
    cam.rotation_mode = 'XYZ'
    cam.data.lens_unit = 'FOV'
    cam.data.angle = fov

    rotate_camera(yaw=0.0, pitch=np.pi / 2, roll=0.0, fov=np.pi / 4, radius=radius)

#----------------------------------------------------------------------------

def rotate_camera(yaw: float, pitch: float, roll: float, fov: float, radius: float):
    """
    Puts the camera in the given orientation. Assumes that the radius is constant.
    Params:
    - rotation_angle: [yaw, pitch, roll] in radians
    """
    assert roll == 0.0, f"We do not use roll: {roll}"

    # bpy.ops.object.camera_add()
    scene = bpy.data.scenes["Scene"]
    cam = bpy.data.objects['Camera']

    cam.location = Vector([
        radius * np.sin(pitch) * np.cos(yaw),
        radius * np.sin(pitch) * np.sin(yaw),
        radius * np.cos(pitch),
    ])
    direction = -cam.location
    # Point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # Assume we're using euler rotation
    cam.rotation_euler = rot_quat.to_euler()

    # Set field-of-view
    cam.data.angle = fov

#----------------------------------------------------------------------------

def listdir_full_paths(dir_path: os.PathLike) -> List[os.PathLike]:
    """
    Returns a list of full paths to all objects in the given directory.
    """
    return [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path))]

#----------------------------------------------------------------------------

def sample_bounded_scalar(cfg: EasyDict, batch_size: int):
    """
    cfg --- sampling config
    """
    if cfg.dist == 'normal':
        assert cfg.std == 0.0, f"Scalar must be bounded"
        x = cfg.mean + np.zeros((batch_size,)) # [batch_size]
    elif cfg.dist == 'truncnorm':
        from scipy.stats import truncnorm
        x_min_norm = (cfg.min - cfg.mean) / cfg.std # [1]
        x_max_norm = (cfg.max - cfg.mean) / cfg.std # [1]
        x = truncnorm.rvs(a=x_min_norm, b=x_max_norm, loc=cfg.mean, scale=cfg.std, size=(batch_size,)) # [batch_size]
    elif cfg.dist == 'uniform':
        x = np.random.rand(batch_size) * (cfg.max - cfg.min) + cfg.min # [batch_size]
    else:
        raise NotImplementedError(f'Uknown distribution: {cfg.dist}')

    return x

#----------------------------------------------------------------------------

if __name__ == '__main__':
    initialize(cfg)
    print('Initialized! Running...')
    run(cfg)

#----------------------------------------------------------------------------

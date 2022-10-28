"""
This script exports all the models from Blender into the glb format
It saves them "as is" without preserving the directory structure of the collections.
So you will need to arrange them into folders then yourself if you like.
Hopefully, there are not too many of them...
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
    'output_path': 'path/where/to/save/glb-models',
    'exclude_types': ['CAMERA', 'LIGHT', 'EMPTY'],
    'export_format': 'GLB',
    'delete_all_objects_after_save': True,
})

#----------------------------------------------------------------------------

def run(cfg):
    objects_to_save = [o for o in get_all_objects() if not o.type in cfg.exclude_types]
    save_paths = [os.path.join(cfg.output_path, get_object_col_path(o)) for o in objects_to_save]
    save_ext = '.glb' if cfg.export_format == 'GLB' else '.gltf'
    print('Going to export the following objects:')
    for o, p in zip(objects_to_save, save_paths):
        print(f' - [{o.type}] {o.name} => {p}')
        assert not os.path.exists(p + save_ext), f"Cannot run: {p}{save_ext} already exists."
    for o, p in zip(objects_to_save, save_paths):
        save_object(o, p, cfg.export_format)
    print('Saving is complete!')

    if cfg.delete_all_objects_after_save:
        print('Deleting the objects...')
        delete_all_objects()
        print('Done!')

#----------------------------------------------------------------------------

def get_object_col_path(obj: bpy.types.Object):
    """
    Returns the object name path in the object/collection hierarchy
    """
    if obj.parent is None:
        return obj.name
    else:
        return os.path.join(get_object_col_path(obj.parent), obj.name)

#----------------------------------------------------------------------------

def save_object(obj: bpy.types.Object, save_path: os.PathLike, export_format: str):
    assert not os.path.isfile(save_path), f"Cannot overwrite the file: {save_path}"
    assert export_format in ['GLB', 'GLTF_EMBEDDED'], f"Bad or non-existing export format: {export_format}"

    deselect_all()
    select_object(obj)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=save_path,
        export_format=export_format,

        # By default, GLTF exports the whole scene
        # For some reason, Blender has 2 arguments to specify this...
        # Let's check both then (just in case)
        export_selected=True,
        use_selection=True,

        # These kwargs are set to false by default, but just in case let's be explicit.
        export_cameras=False,
        export_lights=False,

        # No idea whether we really need this, but let's export those just in case.
        # export_tangents=True,

        # A huge list of default kwargs:
        # ui_tab='GENERAL',
        # export_copyright='',
        # export_image_format='AUTO',
        # export_texture_dir='',
        # export_texcoords=True,
        # export_normals=True,
        # export_draco_mesh_compression_enable=False,
        # export_draco_mesh_compression_level=6,
        # export_draco_position_quantization=14,
        # export_draco_normal_quantization=10,
        # export_draco_texcoord_quantization=12,
        # export_draco_generic_quantization=12,
        # export_tangents=False,
        # export_materials='EXPORT',
        # export_colors=True,
        # export_cameras=False,
        # export_selected=False,
        # use_selection=False,
        # export_extras=False,
        # export_yup=True,
        # export_apply=False,
        # export_animations=True,
        # export_frame_range=True,
        # export_frame_step=1,
        # export_force_sampling=True,
        # export_nla_strips=True,
        # export_def_bones=False,
        # export_current_frame=False,
        # export_skins=True,
        # export_all_influences=False,
        # export_morph=True,
        # export_morph_normal=True,
        # export_morph_tangent=False,
        # export_lights=False,
        # export_displacement=False,
        # will_save_settings=False,
        # filepath='',
        # check_existing=True,
        # filter_glob='*.glb;*.gltf',
    )

    # deselect_object(obj)

def deselect_all():
    bpy.ops.object.select_all(action='DESELECT')
    for o in get_all_objects():
        o.select_set(False)
#----------------------------------------------------------------------------

def select_object(obj: bpy.types.Object):
    """Selects a given object in the context"""
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

#----------------------------------------------------------------------------

def get_all_objects(filter_type: str=None) -> List[bpy.types.Object]:
    return [o for o in bpy.data.objects if (filter_type is None or o.type == filter_type)]

#----------------------------------------------------------------------------

def delete_object(obj: bpy.types.Object):
    bpy.data.objects.remove(obj, do_unlink=True)

#----------------------------------------------------------------------------

def delete_all_objects():
    for o in get_all_objects():
        delete_object(o)

#----------------------------------------------------------------------------

def file_ext(path: os.PathLike) -> str:
    return os.path.splitext(path)[1].lower()

#----------------------------------------------------------------------------

if __name__ == '__main__':
    run(cfg)

#----------------------------------------------------------------------------

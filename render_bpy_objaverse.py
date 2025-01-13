import argparse, sys, os, math
import bpy
from mathutils import Vector, Matrix
import sys
import time
import numpy as np
import PIL.Image as Image
import json
import mathutils

argv = sys.argv
if "--" not in argv:
    argv = []  
else:
    argv = argv[argv.index("--") + 1:]  

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--view', type=int, default=132, help='the index of view to be rendered')
parser.add_argument('--subject', type=str, default='00003', required=True, help="Path to the object file")
parser.add_argument('--resolution', type=int, default=1024, help='Resolution of the images.')
parser.add_argument('--reset_object_euler', action='store_true', help='set object rotation euler to 0')   
parser.add_argument('--radius', type=float, default=1.5, help='radius of rendering sphere')
parser.add_argument("--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"])

args = parser.parse_args(argv)  

print("args.subject: ", args.subject)

#################### BBY CONFIGURATION ####################
context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True

scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA"


def create_camera_to_world_matrix(elevation, azimuth, radius=1.0):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    x = np.cos(elevation) * np.sin(azimuth) * radius
    y = np.sin(elevation) * radius
    z = np.cos(elevation) * np.cos(azimuth) * radius

    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    # Construct view matrix
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    cam2world = np.eye(4)
    cam2world[:3, :3] = np.array([right, new_up, -forward]).T
    cam2world[:3, 3] = camera_pos
    return cam2world

def convert_opengl_to_blender(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        camera_matrix_blender = np.dot(flip_yz, camera_matrix)
    return camera_matrix_blender

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene_human():
    bbox_min, bbox_max = scene_bbox()
    scale = 0.8 / max(bbox_max - bbox_min) 
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    norm_human_loc = bpy.data.objects["objaverse"].location
    bpy.ops.object.select_all(action="DESELECT")

    return norm_human_loc, scale

def get_a_camera_location(loc):
    location = Vector([loc[0],loc[1],loc[2]])
    direction = - location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    rotation_euler = rot_quat.to_euler()
    return location, rotation_euler

def normalize_camera(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (
            np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8
        )
        camera_matrix[:, :3, 3] = translation
    return camera_matrix.reshape(-1, 16)

def get_3x4_RT_matrix_from_blender(cam):
    
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    T_world2bcam = -1*R_world2bcam @ location

    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

def get_calibration_matrix_K_from_blender(mode='simple'):

    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale
    height = scene.render.resolution_y * scale 

    camdata = scene.camera.data

    if mode == 'simple':

        aspect_ratio = width / height
        K = np.zeros((3,3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()
    
    if mode == 'complete':

        focal = camdata.lens 
        sensor_width = camdata.sensor_width 
        sensor_height = camdata.sensor_height 
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            s_u = width / sensor_width / pixel_aspect_ratio 
            s_v = height / sensor_height
        else: 
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0

        K = np.array([
            [alpha_u,    skew, u_0],
            [      0, alpha_v, v_0],
            [      0,       0,   1]
        ], dtype=np.float32)
    
    return K

def load_objaverse_obj(objaverse_obj_file):

    bpy.ops.object.select_all(action='DESELECT')
    print("importing objaverse obj file")
    print(objaverse_obj_file)

    if objaverse_obj_file.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=objaverse_obj_file)
    elif objaverse_obj_file.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=objaverse_obj_file, forward_axis='NEGATIVE_Z', up_axis='Y')
    elif objaverse_obj_file.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=objaverse_obj_file, forward_axis='NEGATIVE_Z', up_axis='Y', validate_meshes=False)
    elif objaverse_obj_file.endswith(".ply"):
        bpy.ops.import_mesh.ply(filepath=objaverse_obj_file)
    
    imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

    for obj in imported_objects:
        bpy.ops.object.select_all(action='DESELECT')
        
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    obj.name = 'objaverse' 
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.data.objects['objaverse']
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    bpy.ops.object.shade_smooth()


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

cam = bpy.context.scene.objects['Camera']
cam.data.sensor_width = 32
desired_fov_deg = 49.1
desired_fov_rad = math.radians(desired_fov_deg)
sensor_width = cam.data.sensor_width

focal_length = sensor_width / (2 * math.tan(desired_fov_rad / 2))
cam.data.lens = focal_length

def get_camera_objects():
    cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
    return cameras


VIEWS = ["_0", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10", "_11", "_12", "_13", "_14", "_15", "_16", "_17", "_18", "_19", "_20", "_21", "_22", "_23", "_24", "_25", "_26", "_27", "_28", "_29", "_30", "_31", "_32", "_33", "_34", "_35", "_36", "_37", "_38", "_39", "_40", "_41", "_42", "_43", "_44", "_45", "_46", "_47", "_48", "_49", "_50", "_51", "_52", "_53", "_54", "_55", "_56", "_57", "_58", "_59", "_60", "_61", "_62", "_63", "_64", "_65", "_66", "_67", "_68", "_69", "_70", "_71", "_72", "_73", "_74", "_75", "_76", "_77", "_78", "_79", "_80", "_81", "_82", "_83", "_84", "_85", "_86", "_87", "_88", "_89", "_90", "_91", "_92", "_93", "_94", "_95", "_96", "_97", "_98", "_99", "_100", "_101", "_102", "_103", "_104", "_105", "_106", "_107", "_108", "_109", "_110", "_111", "_112", "_113", "_114", "_115", "_116", "_117", "_118", "_119", "_120", "_121", "_122", "_123", "_124", "_125", "_126", "_127", "_128", "_129", "_130", "_131"]
ELEVATION = np.arange(-60, 60, 120/100)
AZIMUTH = np.arange(0*10, 360*10, 360*10/args.view)

def save_images(save_folder, scan_subj_path) -> None:

    reset_scene() 

    subj_name = scan_subj_path.split('/')[-2] + '_' + scan_subj_path.split('/')[-1]
    scan_file = scan_subj_path + '.glb'

    os.makedirs(os.path.join(save_folder, subj_name), exist_ok=True)

    load_objaverse_obj(scan_file)

    if args.reset_object_euler:
        for obj in scene_root_objects():
            obj.rotation_euler[0] = 0  
        bpy.ops.object.select_all(action="DESELECT")

    norm_human_loc, norm_scale = normalize_scene_human() 
    
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    
    radius = args.radius
    
    camera_locations = [np.array([0,-radius,0]),] * args.view 
    for location in camera_locations:
        _location,_rotation = get_a_camera_location(location)
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=_location, rotation=_rotation,scale=(1, 1, 1))
        _camera = bpy.context.selected_objects[0]
        _camera.data.sensor_width = 32
        _camera.data.lens = focal_length

        _constraint = _camera.constraints.new(type='TRACK_TO')
        _constraint.track_axis = 'TRACK_NEGATIVE_Z'
        _constraint.up_axis = 'UP_Y'
        _camera.parent = cam_empty
        _constraint.target = cam_empty
        _constraint.owner_space = 'LOCAL'
    
    area_light_dist = 7
    light_locations = [np.array([area_light_dist,area_light_dist,area_light_dist]), np.array([area_light_dist,area_light_dist,-area_light_dist]), np.array([area_light_dist,-area_light_dist,area_light_dist]), np.array([area_light_dist,-area_light_dist,-area_light_dist]), np.array([-area_light_dist,area_light_dist,area_light_dist]), np.array([-area_light_dist,area_light_dist,-area_light_dist]), np.array([-area_light_dist,-area_light_dist,area_light_dist]), np.array([-area_light_dist,-area_light_dist,-area_light_dist])]
    for location in light_locations:
        bpy.ops.object.light_add(type='AREA', radius=1, align='WORLD', location=location, scale=(1, 1, 1))
        light = bpy.context.selected_objects[0]
        light.data.energy = 1000
        light.data.size = 5.0
        light.data.color = (1, 1, 1)
        light.data.use_nodes = True
        light.data.node_tree.nodes["Emission"].inputs[1].default_value = 100
        light.data.node_tree.nodes["Emission"].inputs[0].default_value = (1, 1, 1, 1)

        track_constraint = light.constraints.new(type='TRACK_TO')
        track_constraint.target = bpy.data.objects['objaverse']
        track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        track_constraint.up_axis = 'UP_Y'
    bpy.context.view_layer.update()

    viewidx = args.view
    
    for j in range(viewidx):
        view = f"{viewidx:03d}"+ VIEWS[j]
        
        if j < 100:
            elevation = ELEVATION[j]
            azimuth = AZIMUTH[j]
        else:
            elevation = 0            
            azimuth = 360/(viewidx-100)*(j-100)

        camera_matrix = create_camera_to_world_matrix(elevation, azimuth, args.radius) 
        camera_matrix = convert_opengl_to_blender(camera_matrix)
        
        cam = bpy.data.objects[f'Camera.{j+1:03d}']
        cam.matrix_world = mathutils.Matrix(camera_matrix)
        
        scene.camera = cam

        file_name = f"rgb_132_{str(j)}.png"
        scene.render.filepath = os.path.join(save_folder, subj_name, file_name)
        bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    # try:
    start_i = time.time()

    # blender-4.1.1-linux-x64/blender -b -P render_bpy_objaverse.py -- --subject 000-001/112c059282cf4511a01fd27211edcae8

    save_folder = './rendering_data/ImagedreamLGM_Objaverse_132view'
    scan_base_path = '/mnt/lustre/datasets/objaverse/glbs'
    
    subj = args.subject # "000-052/cfd1a14262c9452a98265ff51df738da" 
    scan_subj_path = os.path.join(scan_base_path, subj)

    save_images(save_folder, scan_subj_path)

    end_i = time.time()
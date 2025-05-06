#import blenderproc as bproc
#from blenderproc.python.types.MeshObjectUtility import create_from_point_cloud
#
#bproc.init()
#
#def render(meshes, render_params):
#
#    camera_position = render_params["camera_position"]
#    camera_euler_rotation = render_params["camera_euler_rotation"]
#
#    light = bproc.types.Light()
#    light.set_type("POINT")
#    light.set_location(camera_position)
#    light.set_energy(1000)
#    
#    for mesh in meshes:
#        points = mesh.vertices[mesh.faces]
#        create_from_point_cloud(
#                points=points,
#                object_name=mesh.name,
#                ) 
#
#    bproc.camera.set_resolution(128, 128)
#
#    matrix_world = bproc.math.build_transformation_mat(camera_position, camera_euler_rotation)
#    bproc.camera.add_camera_pose(matrix_world)
#
#    data = bproc.renderer.render()
#    return data

import bpy
import numpy as np
import mathutils
import math

def render(meshes, render_params):
    # Clear existing objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()
    
    # Get camera parameters
    camera_position = render_params["position"]
    look_at = render_params["look_at"]
    fov = render_params.get("fov", 60.0)
    aspect_ratio = render_params.get("aspect_ratio", 16/9)
    near_clip = render_params.get("near", 0.01)
    far_clip = render_params.get("far", 100.0)
    
    # Create a light
    light_data = bpy.data.lights.new(name="Light", type='POINT')
    light_data.energy = 1000
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = camera_position
    
    # Create meshes
    for mesh in meshes:
        create_mesh_from_point_cloud(mesh.vertices, mesh.faces, mesh.name)
    
    # Set up the camera
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new(name="Camera", object_data=camera_data)
    bpy.context.collection.objects.link(camera_object)
    
    # Set camera position
    camera_object.location = camera_position
    
    # Calculate rotation to look at target
    direction = mathutils.Vector(look_at) - mathutils.Vector(camera_position)
    
    # Point the camera's -Z axis at the target (Blender cameras point -Z by default)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera_object.rotation_euler = rot_quat.to_euler()
    
    # Set field of view (convert from horizontal to vertical FOV)
    vertical_fov = 2 * math.atan(math.tan(math.radians(fov) / 2) / aspect_ratio)
    camera_data.angle = vertical_fov  # Blender uses radians
    
    # Set clipping planes
    camera_data.clip_start = near_clip
    camera_data.clip_end = far_clip
    
    # Set up the scene
    scene = bpy.context.scene
    scene.camera = camera_object
    
    # Set resolution based on aspect ratio
    base_height = 128
    scene.render.resolution_y = base_height
    scene.render.resolution_x = int(base_height * aspect_ratio)
    scene.render.resolution_percentage = 100
    
    # Set up the renderer
    scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Or 'CYCLES'
    scene.render.filepath = "/tmp/render.png"
    scene.render.image_settings.file_format = 'PNG'
    
    # Render
    bpy.ops.render.render(write_still=True)
    
    # Get the rendered image data
    rendered_image = bpy.data.images['Render Result']
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    pixels = np.array(rendered_image.pixels[:]).reshape((height, width, 4))
    import matplotlib.pyplot as plt
    plt.imshow(pixels)
    plt.show()
    
    # Return data in a format similar to blenderproc
    data = {
        "colors": pixels[:, :, :3],  # RGB channels
        "depth": None  # You would need to set up a depth pass to get this
    }
    
    return data

def create_mesh_from_point_cloud(vertices, faces, object_name):
    """Create a mesh from a point cloud of triangle vertices."""
    # Create a new mesh
    mesh = bpy.data.meshes.new(object_name)
    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Set mesh data
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    
    # Set smooth shading
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    
    return obj

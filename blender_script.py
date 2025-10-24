import bpy
import sys
import os
import argparse
from mathutils import Vector
from mathutils import Matrix

# Locate the importer code file
addon_file = "/opt/blender/4.3/scripts/addons/ImporterScript/__init__.py"

# Read and execute the code in __init__.py
with open(addon_file, 'r', encoding='utf-8') as f:
    code = f.read()

exec(code, globals(), globals())  # This will define the operator and register function

# Blender adds its own arguments before ours, so we find '--' and slice from there.
if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = sys.argv[1:]  # If no '--', take all but the script name

# Set up argparse
parser = argparse.ArgumentParser(description="Import and process a PES file in Blender.")
parser.add_argument("-i", "--input_pes", required=True, help="Input PES file path")
parser.add_argument("-o", "--output_image", required=True, help="Output image file path")
parser.add_argument("-v", "--output_video", required=True, help="Output video file path")

args = parser.parse_args(argv)
script_dir = os.path.dirname(__file__)
##################################removed these two lines of code for new batch script
pes_dir = os.path.join(script_dir, "input_PES")
input_pes = os.path.join(pes_dir, args.input_pes)

#modified the above line to this one instead
#input_pes = os.path.join(script_dir, args.input_pes)
input_pes = os.path.abspath(input_pes)

output_dir = os.path.join(script_dir, "output")
output_image = os.path.join(output_dir, args.output_image)
output_image = os.path.abspath(output_image)
output_video = os.path.join(output_dir, args.output_video)
output_video = os.path.abspath(output_video)

print("Attempting to import:", input_pes)
print("Absolute path:", os.path.abspath(input_pes))
print("File exists?", os.path.exists(input_pes))

# Now proceed as before
bpy.ops.import_scene.embroidery(
    filepath=input_pes,
    show_jump_wires=True,
    do_create_material=True,
    create_collection=True,  # Ensure collection is created so we can center objects
    line_depth='GEOMETRY_NODES',
    thread_thickness=0.2,
)

#########################################################################################################

# After import, assume the collection is named after the PES file:
collection_name = os.path.basename(input_pes)
collection = bpy.data.collections.get(collection_name)

if collection and len(collection.objects) > 0:
    # Convert all objects in the collection to meshes
    bpy.ops.object.select_all(action='DESELECT')
    for obj in collection.objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.convert(target='MESH')
        obj.select_set(False)

    bpy.context.view_layer.update()

    # Calculate the combined bounding box center of all objects
    min_coord = Vector((float('inf'), float('inf'), float('inf')))
    max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in collection.objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_coord.x = min(min_coord.x, world_corner.x)
            min_coord.y = min(min_coord.y, world_corner.y)
            min_coord.z = min(min_coord.z, world_corner.z)
            max_coord.x = max(max_coord.x, world_corner.x)
            max_coord.y = max(max_coord.y, world_corner.y)
            max_coord.z = max(max_coord.z, world_corner.z)

    center = (min_coord + max_coord) / 2

    # #Adjust the 3D cursor in the position of table top z axis
    # bpy.context.scene.cursor.location = (0, 0, table_max_vert)

    # Center object in the world 0,0,0
    print("Calculated center:", center)
    for obj in collection.objects:
        old_loc = obj.location.copy()
        obj.location = obj.location - center
        print(f"Moved {obj.name} from {old_loc} to {obj.location}")

    #Center Origin for each object
    for obj in collection.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='BOUNDS')

    # Define and store the biggest dimensions of the imported object
    obj_width = (max_coord.x - min_coord.x)
    obj_length = (max_coord.y - min_coord.y)
    obj_height = (max_coord.z - min_coord.z)
    biggest_dimension = max(obj_width, obj_length, obj_height)
    print("the biggest dimension is:", biggest_dimension)

    # Define the reference dimension you want all models to conform to
    desired_dimension = 0.127

    # Compute the scale factor
    if biggest_dimension > 0:
        scale_factor = desired_dimension / biggest_dimension
        print(f"Scaling objects by a factor of {scale_factor} to fit into {desired_dimension}m")
    else:
        scale_factor = 1.0
        print("No scaling needed (biggest_dimension is zero or invalid)")

    # Apply the scale factor to all objects in the collection
    for obj in collection.objects:
        # Since objects are centered at the origin, scaling is straightforward
        obj.scale *= scale_factor

    bpy.context.view_layer.update()

    # Re-initialize min/max coords before the second calculation
    min_coord = Vector((float('inf'), float('inf'), float('inf')))
    max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in collection.objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_coord.x = min(min_coord.x, world_corner.x)
            min_coord.y = min(min_coord.y, world_corner.y)
            min_coord.z = min(min_coord.z, world_corner.z)
            max_coord.x = max(max_coord.x, world_corner.x)
            max_coord.y = max(max_coord.y, world_corner.y)
            max_coord.z = max(max_coord.z, world_corner.z)

    # # Align the table's top to the collection's bottom
    z_offset = -min_coord.z  # If min_coord.z is negative, we shift up; if positive, we shift down
    if abs(z_offset) > 1e-7:  # Just a tiny tolerance check
        for obj in collection.objects:
            obj.location.z += z_offset

    bpy.context.view_layer.update()

    # bpy.context.view_layer.update()

else:
    print(f"No collection found or empty collection for {collection_name}")

# Reference area and hair density
reference_width = 0.0670
reference_length = 0.0670
reference_area = reference_width * reference_length
reference_hair_count = 300

# Add hair particle system to each object in the collection
for obj in collection.objects:
    # Check if the object has geometry (vertices)
    if not obj.data or len(obj.data.vertices) == 0:
        print(f"Skipping object {obj.name}: No geometry found.")
        continue

    # Select and make the object active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Add a particle system
    bpy.ops.object.particle_system_add()

    # Access the particle system and settings
    psys = obj.particle_systems[-1]
    pset = psys.settings
    pset.type = 'HAIR'
    pset.hair_length = 0.001
    pset.count = 500  # Temporary, will calculate precisely below
    pset.use_advanced_hair = True
    pset.root_radius = 0.001
    pset.tip_radius = 0.001
    pset.rendered_child_count = 0  # Just to ensure we see actual hairs
    pset.child_type = 'NONE'  # No children for simplicity
    # Jittering is typically for children, if needed:
    # pset.child_jitter = 2.0 if using children. For simple hairs, no children needed.

    # Calculate object’s bounding box dimensions in local space
    # We assume object is roughly flat, so just width and length
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    x_coords = [v.x for v in bbox]
    y_coords = [v.y for v in bbox]
    obj_width = (max(x_coords) - min(x_coords))
    obj_length = (max(y_coords) - min(y_coords))

    # Calculate hair count proportionally
    obj_area = obj_width * obj_length
    scale_factor = obj_area / reference_area
    hair_count = int(reference_hair_count * scale_factor)
    pset.count = max(hair_count, 1)  # Ensure at least 1 hair

thread_mat = bpy.data.materials.get("ThreadMaterial")
if thread_mat:
    print("Found ThreadMaterial, adjusting parameters.")
    # Ensure material is using nodes
    if thread_mat.use_nodes and thread_mat.node_tree:
        bsdf_node = None
        # Find the Principled BSDF node
        for node in thread_mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf_node = node
                break

        if bsdf_node:
            # Metallic
            bsdf_node.inputs["Metallic"].default_value = 0.5
            # Roughness
            bsdf_node.inputs["Roughness"].default_value = 0.35
            # Coat (Clearcoat)
            bsdf_node.inputs[18].default_value = 1.0
            bsdf_node.inputs[19].default_value = 0.3
            # IOR
            bsdf_node.inputs[20].default_value = 1.5

#########################################################################################################

# Now the PES embroidery is imported and centered into the BlenderSetup scene.
# Camera and lights are pre-set in BlenderSetup.blend.

# ------------------------------------------------------------
# Render Settings (example)
# ------------------------------------------------------------
# A helper function to set render resolution
def set_resolution(scene, width, height):
    scene.render.resolution_x = width
    scene.render.resolution_y = height

# A helper function to set the active camera
def set_active_camera(scene, camera_name):
    cam = bpy.data.objects.get(camera_name)
    if cam and cam.type == 'CAMERA':
        scene.camera = cam
    else:
        print(f"Camera {camera_name} not found or not a camera")

# A helper function to render and save an image (composite output)
def render_image(scene, filepath):
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

# ------------------------------------------------------------
# Setup compositing for background
# ------------------------------------------------------------
#scene = bpy.context.scene
#scene.render.engine = 'CYCLES'
#scene.cycles.samples = 100
scene = bpy.context.scene

# Enable GPU rendering
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA' #OPTIX or CUDA
bpy.context.preferences.addons['cycles'].preferences.get_devices()

# Enable all CUDA devices
gpu_found = False  # Initialize a flag to check if GPU is found

for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if device.type == 'CUDA':
        device.use = True
        gpu_found = True
        print(f"Enabled GPU: {device.name}")

# If no GPU was found, print a message
if not gpu_found:
    print("GPU not found")


# Optimize render tiles for GPU
bpy.context.scene.cycles.tile_x = 256
bpy.context.scene.cycles.tile_y = 256

scene.render.engine = 'CYCLES'
scene.cycles.samples = 512
scene.cycles.device = 'GPU'
scene.cycles.tile_size = 4096  # Increased for modern GPUs
scene.cycles.use_persistent_data = True  # Cache BVH
scene.cycles.use_denoising = True
scene.render.use_persistent_data = True
scene.cycles.max_subdivisions = 4  # Limit subdivisions for performance

# Set output resolution and scaling
scene.render.resolution_percentage = 100  # Ensure full resolution is used
scene.render.image_settings.compression = 0  # Save output at full quality

# Ensure compositing is used
scene.use_nodes = True
scene.render.use_compositing = True



######################################
tree = scene.node_tree

# Find the Alpha Over node
alpha_over_node = None
for node in tree.nodes:
    if node.type == 'ALPHAOVER':
        alpha_over_node = node
        break

if not alpha_over_node:
    raise RuntimeError("Alpha Over node not found in the compositing setup!")

# Find or create an RGB node for setting the background color
rgb_node = None
for node in tree.nodes:
    if node.type == 'RGB':
        rgb_node = node
        break

if not rgb_node:
    rgb_node = tree.nodes.new(type='CompositorNodeRGB')
    rgb_node.location = alpha_over_node.location.x - 200, alpha_over_node.location.y
    # Connect the RGB node to the Alpha Over node's background input (second input)
    tree.links.new(rgb_node.outputs[0], alpha_over_node.inputs[1])

# ------------------------------------------------------------
# Assume 'output_image' and 'output_video' come from outside (e.g., PowerShell arguments)
# ------------------------------------------------------------
# Example:
# output_image = "D:/renders/output.png"
# output_video = "D:/renders/output.mp4"

output_image_base = output_image.rsplit('.', 1)[0]  # Remove file extension if any

# Define postfixes dynamically
# bg options: "white", "black", "side_white"
# resolution: "1024", "2048"
postfixes = [
    ("white", "1024"),
    ("white", "2048"),
    ("black", "1024"),
    ("black", "2048"),
    ("side_white", "1024"),
    ("side_white", "2048")
]

for bg, res in postfixes:
    # Determine the camera based on bg
    if bg == "white":
        # White background from TopView
        set_active_camera(scene, "TopView")
        rgb_node.outputs[0].default_value = (1,1,1,1)  # Pure white background
    if bg == "black":
        # Black background from TopView
        set_active_camera(scene, "TopView")
        rgb_node.outputs[0].default_value = (0,0,0,1)  # Pure black background
    elif bg == "side_white":
        # White background from SideView
        set_active_camera(scene, "SideView")
        rgb_node.outputs[0].default_value = (1,1,1,1)  # White background again
        bpy.data.collections["Environment_Table"].hide_render = False

    width = int(res)
    height = int(res)
    set_resolution(scene, width, height)

    unique_output_image = f"{output_image_base}_{bg}_{res}.png"
    print(f"Rendering: {unique_output_image}")
    render_image(scene, unique_output_image)


# Render animation to video
# Set up video render settings
scene.render.engine = 'CYCLES'
scene.cycles.samples = 10
scene.render.filepath = output_video
scene.frame_start = 0
scene.frame_end = 160
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'
scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

bpy.data.collections["Environment_Table"].hide_render = True
bpy.data.collections["Lights"].hide_render = True
bpy.data.collections["Environment_Anim"].hide_render = False
bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[2] = 1.09956
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.1
bpy.data.scenes["Scene"].node_tree.nodes["Mix"].inputs[0].default_value = 0
set_active_camera(scene, "CameraAnim")
bpy.ops.render.render(animation=True, write_still=True)
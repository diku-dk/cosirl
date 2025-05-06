from abc import abstractmethod
import os

import numpy as np
import torch
from gymnasium import spaces

from src.simulator import setup_sofa
from src.nn.colon_dataloader import create_mesh_dataloader
from src.nn.data_augmentation import IdentityTransform
from src.render.blender import render as blender_render

class Mesh:
    def __init__(self, vertices, faces, name):
        self.vertices = vertices
        self.faces = faces
        self.name = name

    @classmethod
    def from_sofa_topology(cls, topo, name):
        vertices = topo.position[:]
        faces = topo.triangles[:]
        return cls(vertices, faces, name)

class Robot:
    @abstractmethod
    def __init__(self, scene_root, robot_config):
        self.state = {}

    def step(self, scene_root, action):
        self._apply_action(action)
        self.state = self._get_state(scene_root)
        return self.state

    @abstractmethod
    def _apply_action(self, action):
        """
        Modify the objects of the scene root which are affected by the action to reflect that the action was taken.
        """
        # Potentially we don't need the scene_root here if we can save object references in self.__init__(scene_root).

    @abstractmethod
    def _get_state(self, scene_root):
        """
        Find the object, which is a part of the state the reinforcement agent needs to know about. E.g. positions,
        camera position and orientation, sensor output, etc.
        """
        new_state = {}
        # Potentially we don't need the scene_root here if we can save object references in self.__init__(scene_root).
        return new_state

class CapsuleRobot(Robot):
    def __init__(self, scene_root, robot_config):

        # Setup SOFA
        self.capsule_node = scene_root.addChild("Robot")
        
        load_mesh(sofa_target=self.capsule_node, mesh_path=robot_config["mesh_path"])
        self.topology = self.capsule_node.addObject('MeshTopology', src='@loader')

        self.capsule_node.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=0.1, rayleighMass=0.1)
        self.capsule_node.addObject('CGLinearSolver', name='linearSolver', iterations=25, tolerance=1e-9, threshold=1e-9)

        self.capsule_node.addObject("MechanicalObject", name="dofs", template="Vec3d")
        self.capsule_node.addObject("UniformMass", totalMass=1.0)  # Add mass
        self.capsule_node.addObject("ConstantForceField", name="force", forces=[[0,0,10]])  # Increased force
        
        self.capsule_node.addObject("OglModel", src="@../Robot/loader", color=[0.8, 0.2, 0.2, 1.0])
        self.force = self.capsule_node.addObject("ConstantForceField", name="force", forces=[[0,0,1]])

        # Setup action spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Camera output
        camera_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        
        # Robot arm location
        capsule_position = spaces.Box(
            low=-robot_config["world_min"],
            high=robot_config["world_max"],
            shape=(1, 3),
            dtype=np.float32,
        )
        
        self.observation_space = spaces.Dict({
            "camera_output": camera_space,
            "capsule_position": capsule_position,
        })
        
        verts = self.topology.position[:] # [:] implicitly converts the SOFA data container to a numpy array
        self.camera_top_idx = np.argmax(verts[:,1]) # This it the up and down axis
        self.camera_bottom_idx = np.argmin(verts[:,1])
        self.camera_parameters = self.get_camera_parameters(
                self.topology,
                self.camera_top_idx,
                self.camera_bottom_idx,
                )

        self.state = {
            "camera_output": np.zeros((128, 128, 3), dtype=np.uint8),
            "capsule_position": self.camera_parameters["position"],
        }

    def get_camera_parameters(self, topology, top_idx, bottom_idx):
        verts = topology.position[:] # [:] implicitly converts the SOFA data container to a numpy array
        p_top = verts[top_idx]
        p_bottom = verts[bottom_idx]
        camera_parameters = {
                "position" : (p_top + p_bottom) / 2,
                "look_at" : p_top,
                "fov": 60.0,
                "aspect_ratio": 16/9,
                "near": 0.01,
                "far": 100.0,
                }
        return camera_parameters

    def _apply_action(self, action):
        pass

    def _get_state(self, _scene_root):
        self.camera_parameters = self.get_camera_parameters(
                self.topology,
                self.camera_top_idx,
                self.camera_bottom_idx,
                )
        return self

def load_mesh(sofa_target, mesh_path):
    if mesh_path.endswith(".obj"):
        mesh_loader = sofa_target.addObject(
                'MeshOBJLoader',
                name='loader', 
                filename=mesh_path, 
                triangulate=True,
                )
        _ = mesh_loader # why do we save a reference to the mesh_loader object?
    else:
        raise NotImplemented(f"Add more readers for this filetype: '{mesh_path}'")

def setup_sofa_custom(config):
    # This is here for now, probably move to setup_sofa
    setup_sofa.setup_sofa_environment(config)
    
    import Sofa
    import SofaRuntime
    import Sofa.Simulation
    from SofaRuntime import PluginRepository, importPlugin

    importPlugin("Sofa.Component.Visual") # For VisualStyle
    importPlugin("Sofa.GL.Component.Shader") # For visual rendering
    importPlugin("Sofa.GL.Component.Rendering3D") # For OglModel
    importPlugin("Sofa.Component.StateContainer") # For MechanicalObject
    importPlugin("Sofa.Component.IO.Mesh") # For MeshOBJLoader
    importPlugin("Sofa.Component.Topology.Container.Constant") # For MeshTopology
    importPlugin("Sofa.Component.Setting")  # For BackgroundSetting
    importPlugin("Sofa.Component.Mass")
    
    importPlugin("Sofa.Component.Constraint.Projective")  # For constraints, if needed
    importPlugin("Sofa.Component.MechanicalLoad")  # For ConstantForceField
    #importPlugin("Sofa.GL.Component.Rendering2D")  # For OffscreenCamera

    importPlugin("Sofa.Component.ODESolver")        # For EulerImplicit
    importPlugin("Sofa.Component.LinearSolver")     # For CGLinearSolver
    importPlugin("Sofa.Component.Mapping")          # For IdentityMapping
    importPlugin("Sofa.Component.Mass")             # For UniformMass


class SofaColonEndoscopeEnv:
    def __init__(self, config):
        setup_sofa_custom(config)
        from Sofa.Simulation import animate
        self.sofa_animate = animate
        self.config = config
        self.observation_space = np.array([0.0])  # Example: 1D continuous
        self.action_space = np.array([0, 1])      # Example: 2 discrete actions
        self.state = None
        self.dt = 0.1
        
        self.colon_dataloader = create_mesh_dataloader(
            mesh_dir=config["user_settings"]["colon_dir"],
            batch_size=1, # TODO: While this should be here, potentially we should share a dataloader between vectorized envs later
            transform=IdentityTransform()
        )
        self.data_generator = iter(self.colon_dataloader)
        self.scene_root = None 
        self.world_min = 0
        self.world_max = 1
    
    def get_next_batch(self):
        try:
            batch = next(self.data_generator)
        except StopIteration:
            # Re-create the iterator, which will re-shuffle if shuffle=True
            self.data_generator = iter(self.colon_dataloader)
            batch = next(self.data_generator)
        return batch[0]

    def init_sofa_scene(self):
        import Sofa
        root = Sofa.Core.Node("root")
        root.dt = 0.1
        # This should no longer be needed if we recreate the root every time
        #if self.root.isInitialized():
        #    # Clear existing children if needed
        #    for child in list(self.root.children):
        #        self.root.removeChild(child)        # Get first mesh from dataloader
         
        # Add both animation loop and visual loop
        root.addObject('DefaultAnimationLoop')
        root.addObject('DefaultVisualManagerLoop')  # Required for visualization
        root.addObject('VisualStyle', displayFlags="showVisual showBehavior")
        
        # Add camera for better view
        #root.addObject('Camera', position=[0, 0, 10], lookAt=[0, 0, 0])
        
        return root

    def add_colon_to_scene(self, scene_root, colon_path: str):
        colon_node = scene_root.addChild("Colon")
        
        load_mesh(sofa_target=colon_node, mesh_path=colon_path)
        self.colon_topology = colon_node.addObject('MeshTopology', src='@loader')
        self.world_min = min(self.world_min, np.min(self.colon_topology.position[:]))
        self.world_max = max(self.world_max, np.max(self.colon_topology.position[:]))

        colon_node.addObject('MechanicalObject', name='dofs', template='Vec3d') 
        #visual = colon_node.addChild('Visual')
        colon_node.addObject('OglModel', name='visual', 
                            color=[0.8, 0.4, 0.3, 1.0],
                            src='@../Colon/loader') 

    def add_robot_to_scene(self, scene_root, robot_config):
        self.robot = CapsuleRobot(scene_root, robot_config)

    def reset(self):
        self.world_min = 0
        self.world_max = 1
        colon_path = self.get_next_batch()
        self.scene_root = self.init_sofa_scene()
        self.add_colon_to_scene(self.scene_root, colon_path=colon_path)
        self.config["robot"].update(
                {
                    "world_min" : self.world_min,
                    "world_max" : self.world_max,
                    })
        self.add_robot_to_scene(self.scene_root, robot_config=self.config["robot"])
 
        self.scene_root.init()

        self.sensor_output = self.update_sensor_output(self.scene_root) 
        camera_sensors = self.sensor_output["camera_sensors"]
        self.last_render = self.render()
        obs = [self.last_render, self.sensor_output]
        return obs

    def step(self, action):
        if self.robot is None or self.scene_root is None:
            raise RuntimeError("Neither self.robot nor self.scene_root can be none. Dif you forget to call reset first?")

        self.robot.step(self.scene_root, action)
        self.update_sofa_state(self.scene_root, action)
        self.sensor_output = self.update_sensor_output(self.scene_root)

        camera_sensors = self.sensor_output["camera_sensors"]
        self.last_render = self.render(self.scene_root, camera_sensors)
        obs = [self.last_render, self.sensor_output]

        reward = self.calculate_reward(self.scene_root, self.sensor_output)
        terminated = False # change to final terminated determiner
        truncated = False # change to final truncated determiner
        info = {} # Optional extra infsofa_stateo
        return obs, reward, terminated, truncated, info

    def close(self):
        pass

    def update_sofa_state(self, scene_root, actions):
        self.sofa_animate(scene_root, self.dt)

    def update_sensor_output(self, sofa_state):
        sensor_output = {"camera_sensors": None} # TODO: Implement
        return sensor_output

    def calculate_reward(self, scene_root, sensor_output):
        return 1.0 # TODO: Implement

    def render(self):
        colon_mesh = Mesh.from_sofa_topology(self.colon_topology, "Colon")
        render_params = {
                "world_min" : self.world_min,
                "world_max" : self.world_max,
                }
        render_params.update(self.robot.camera_parameters)
        render_output = blender_render([colon_mesh], render_params)
        return render_output

    def visualize_environment(self):
        # This function runs a simple Sofa GUI
        import Sofa.Gui
 
        # Create GUI
        Sofa.Gui.GUIManager.Init("main", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(self.scene_root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
 
        # Start GUI loop
        Sofa.Gui.GUIManager.MainLoop(self.scene_root)
        Sofa.Gui.GUIManager.closeGUI()

def main():
    config = EnvConfig()
    env = SofaColonEndoscopeEnv(config)
    
    obs, info = env.reset()
    for _ in range(1000):
        action = np.random.uniform(-1, 1, size=env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            
    env.close()

if __name__ == "__main__":
    main()

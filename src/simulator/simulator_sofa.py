import os

import numpy as np
import torch
from gymnasium import spaces

from src.simulator import setup_sofa
from src.nn.colon_dataloader import create_mesh_dataloader
from src.nn.data_augmentation import IdentityTransform

class SofaColonEndoscopeEnv:
    def __init__(self, config):
        setup_sofa.setup_sofa_environment(config)
        
        import Sofa
        import SofaRuntime
        from SofaRuntime import PluginRepository, importPlugin

        importPlugin("Sofa.Component.Visual") # For VisualStyle
        importPlugin("Sofa.GL.Component.Shader") # For visual rendering
        importPlugin("Sofa.GL.Component.Rendering3D") # For OglModel
        importPlugin("Sofa.Component.StateContainer") # For MechanicalObject
        importPlugin("Sofa.Component.IO.Mesh") # For MeshOBJLoader
        importPlugin("Sofa.Component.Topology.Container.Constant") # For MeshTopology
        importPlugin("Sofa.Component.Setting")  # For BackgroundSetting
 
        self.config = config
        self.observation_space = np.array([0.0])  # Example: 1D continuous
        self.action_space = np.array([0, 1])      # Example: 2 discrete actions
        self.state = None

        self.root = Sofa.Core.Node("root")
        self.root.dt = 0.01

        # Add both animation loop and visual loop
        self.root.addObject('DefaultAnimationLoop')
        self.root.addObject('DefaultVisualManagerLoop')  # Required for visualization
        self.root.addObject('VisualStyle', displayFlags="showVisual showBehavior")
        
        # Add camera for better view
        self.root.addObject('Camera', position=[0, 0, 10], lookAt=[0, 0, 0])
        
        # Set background color
        self.root.addObject('BackgroundSetting', color=[1, 1, 1, 1])  # White background
    
        self.colon_dataloader = create_mesh_dataloader(
            mesh_dir=config["user_settings"]["colon_dir"],
            batch_size=1,
            transform=IdentityTransform()
        )
        self.data_generator = iter(self.colon_dataloader)

        self.sofa_state = self.root
        
    def reset(self):
        new_colon_paths = next(self.data_generator)

        if self.root.isInitialized():
            self.root.reset()
            # Clear existing children if needed
            for child in list(self.root.children):
                self.root.removeChild(child)        # Get first mesh from dataloader
            
        # Create colon node
        colon_node = self.root.addChild("Colon")
        
        if new_colon_paths[0].endswith(".obj"):
            mesh_loader = colon_node.addObject('MeshOBJLoader', name='loader', 
                                              filename=new_colon_paths[0], 
                                              triangulate=True)
        else:
            raise NotImplemented(f"Add more readers for this filetype: '{new_colon_paths[0]}'")

        colon_node.addObject('MeshTopology', src='@loader')
        colon_node.addObject('MechanicalObject', name='dofs', template='Vec3d') 
        visual = colon_node.addChild('Visual')
        colon_node.addObject('OglModel', name='visual', 
                            color=[0.8, 0.4, 0.3, 1.0],
                            src='@../Colon/loader')
        
        self.root.init()

        self.sensor_output = self.update_sensor_output(self.sofa_state) 
        camera_sensors = self.sensor_output["camera_sensors"]
        self.last_render = self.render(self.sofa_state, camera_sensors)
        obs = [self.last_render, self.sensor_output]
        return obs

    def step(self, action):
        
        self.sofa_state = self.update_sofa_state(self.sofa_state, action)
        self.sensor_output = self.update_sensor_output(self.sofa_state)
        camera_sensors = self.sensor_output["camera_sensors"]
        self.last_render = self.render(self.sofa_state, camera_sensors)
        obs = [self.last_render, self.sensor_output]

        reward = self.calculate_reward(self.sofa_state, self.sensor_output)
        terminated = False # change to final terminated determiner
        truncated = False # change to final truncated determiner
        info = {} # Optional extra info
        return obs, reward, terminated, truncated, info

    def update_sofa_state(self, sofa_state, actions):
        return sofa_state # TODO: Implement

    def update_sensor_output(self, sofa_state):
        sensor_output = {"camera_sensors": None} # TODO: Implement
        return sensor_output

    def calculate_reward(self, sofa_state, sensor_output):
        return 1.0 # TODO: Implement

    def render(self, sofa_state, camera_sensors):
        return torch.zeros((100,100,1))

    def visualize_environment(self):
        # This function runs a simple Sofa GUI
        import Sofa.Gui
        
        # Create GUI
        Sofa.Gui.GUIManager.Init("main", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(self.root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        
        # Start GUI loop
        Sofa.Gui.GUIManager.MainLoop(self.root)
        Sofa.Gui.GUIManager.closeGUI()


class SofaColonEndoscopeEnv_old:
    def __init__(self, config):
        setup_sofa.setup_sofa_environment(config)
        
        import Sofa
        import SofaRuntime

        self.config = config
        self.maze_paths = self._load_colon_meshes(config.maze_dir)
        self.current_maze_idx = 0
        self.root = None
        self.robot_arm = None
        self.maze_node = None
        self.simulation_time = 0
        self.max_episode_steps = config.max_episode_steps
        self.current_step = 0
        
        # Set up action and observation spaces
        self._setup_spaces()
        
        # Initialize SOFA
        self._initialize_sofa()
        
    def _load_colon_meshes(self, maze_dir):
        """Load all maze .obj files from the specified directory."""
        maze_paths = []
        for file in os.listdir(maze_dir):
            if file.endswith('.obj'):
                maze_paths.append(os.path.join(maze_dir, file))
        assert len(maze_paths) > 0, f"No .obj files found in {maze_dir}"
        return maze_paths
        
    def _setup_spaces(self):
        """Define action and observation spaces."""
        # Action space: robot joint velocities/positions
        # Adjust dimensions based on your robot arm's DOF
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.config.robot_dof,),
            dtype=np.float32
        )
        
        # Observation space: robot state, target position, maze interaction forces
        obs_dim = (
            self.config.robot_dof * 2 +  # Joint positions and velocities
            3 +                          # End effector position
            3 +                          # Target position
            self.config.force_sensors    # Force sensor readings
        )
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def _initialize_sofa(self):
        """Initialize SOFA simulation."""
        # Initialize SOFA runtime and plugins
        setup_sofa.setup_sofa_environment(self.config)
        
        # Create root node
        self.root = Sofa.Core.Node("root")
        
        # Configure SOFA simulation parameters
        self.root.addObject('DefaultAnimationLoop')
        self.root.addObject('DefaultVisualManagerLoop')
        
        # Add solver
        self.root.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
        self.root.addObject('CGLinearSolver', iterations=25, tolerance=1e-5, threshold=1e-5)
        
        # Add gravity
        self.root.gravity.value = [0, -9.81, 0]
        
        # Create robot arm node
        self._create_robot_endoscope()
        
        # Load initial maze
        self._load_maze(self.maze_paths[self.current_maze_idx])
        
        # Set up goal position
        self._setup_goal()
        
        # Initialize simulation
        Sofa.Simulation.init(self.root)
    
    def _create_robot_arm(self):
        """Create robot arm in SOFA simulation."""
        # Robot arm node
        self.robot_arm = self.root.addChild("RobotArm")
        
        # TODO: Add your robot arm model with proper articulations
        # This is a simplified placeholder - you'll need to adapt it to your specific robot model
        
        # Example:
        # Base
        base = self.robot_arm.addChild("Base")
        base.addObject('MechanicalObject', name="mstate", template="Rigid3d", 
                      position=[0, 0, 0, 0, 0, 0, 1], showObject=True, showObjectScale=0.1)
        base.addObject('FixedConstraint', indices=0)
        
        # Arm segments (simplified)
        for i in range(self.config.robot_dof):
            joint = self.robot_arm.addChild(f"Joint_{i}")
            # Add mechanical object, articulation, and visual model for each joint
            # ...
        
        # End effector with collision detection
        end_effector = self.robot_arm.addChild("EndEffector")
        end_effector.addObject('MechanicalObject', name="mstate", template="Rigid3d")
        end_effector.addObject('SphereCollisionModel', radius=0.05)
        
        # Force sensors
        self.force_sensors = []
        for i in range(self.config.force_sensors):
            sensor = end_effector.addObject('ForceField', name=f"ForceSensor_{i}")
            self.force_sensors.append(sensor)
    
    def _rig_colon_for_simulation(self, mesh_data):
        """
        Rig colon and robot for simulation.
        
        Proposed actions:
        - Add colon mesh to SOFA
        - Setup colon boundary conditions
        - Setup colon material model
        - Add robot initial state to SOFA
            - Add robot bodies
            - Add joins (?)
            - Add light source (possibly done in the rendering pipeline
            - Add camera (possibly done in the rendering pipeline)
        - Setup robot material model
        - Setup contact model
        - Specify sofa parameters (These are from the config)
        """
        # Chat bot suggestions: 
        #if self.maze_node is not None:
        #    self.root.removeChild(self.maze_node)
        #    
        #self.maze_node = self.root.addChild("Maze")
        #
        ## Load mesh
        #self.maze_node.addObject('MeshSTLLoader', name="loader", filename=maze_path)
        #self.maze_node.addObject('MeshTopology', src="@loader")
        #
        ## Setup mechanical properties for deformable maze
        #self.maze_node.addObject('MechanicalObject', name="mstate")
        #self.maze_node.addObject('TetrahedronFEMForceField', youngModulus=self.config.maze_young_modulus, 
        #                        poissonRatio=self.config.maze_poisson_ratio)
        #self.maze_node.addObject('UniformMass', totalMass=self.config.maze_mass)
        #
        ## Add collision model
        #self.maze_node.addObject('TriangleCollisionModel')
        #
        ## Add visual model
        #visual = self.maze_node.addChild("Visual")
        #visual.addObject('OglModel', color=[0.8, 0.8, 0.8, 1.0])
        #visual.addObject('IdentityMapping')
    
    def _setup_goal(self):
        """Set up goal position for the current maze."""
        # In a real implementation, you might analyze the maze structure to find a valid goal
        # This is a simplified example
        self.goal_position = np.array([
            self.config.maze_size * 0.8,
            self.config.maze_size * 0.5,
            self.config.maze_size * 0.8
        ])
        
        # Add visual representation of goal
        goal_node = self.root.addChild("Goal")
        goal_node.addObject('MechanicalObject', position=self.goal_position.tolist())
        goal_node.addObject('SphereModel', radius=0.1, color=[0, 1, 0, 1])
    
    def reset(self, seed=None):
        """Reset the environment for a new episode."""
        if seed is not None:
            np.random.seed(seed)


            
        # Choose a random maze
        self.current_maze_idx = np.random.randint(len(self.maze_paths))
        self._load_maze(self.maze_paths[self.current_maze_idx])
        
        # Reset robot to initial position
        # TODO: Implement proper robot reset
        
        # Reset simulation
        Sofa.Simulation.reset(self.root)
        
        # Setup new goal
        self._setup_goal()
        
        # Reset step counter
        self.current_step = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Execute one environment step."""
        self.current_step += 1
        
        # Apply action to robot joints
        self._apply_action(action)
        
        # Step the simulation
        Sofa.Simulation.animate(self.root, self.config.time_step)
        self.simulation_time += self.config.time_step
        
        # Get observation, reward, done status
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        """Apply action to the robot arm."""
        # Scale action from normalized space to actual joint control
        scaled_action = action * self.config.action_scale
        
        # TODO: Apply actions to robot joints
        # Example:
        # for i, joint_action in enumerate(scaled_action):
        #     joint = self.robot_arm.getChild(f"Joint_{i}").getObject("articulation")
        #     current_pos = joint.position.value[0]
        #     joint.position.value = [current_pos + joint_action]
    
    def _get_observation(self):
        """Get current observation."""
        # TODO: Implement proper observation extraction from SOFA state
        # This is a placeholder


        # Get joint positions and velocities
        joint_pos = np.zeros(self.config.robot_dof)
        joint_vel = np.zeros(self.config.robot_dof)

        # Get end effector position
        end_effector_pos = self._get_end_effector_position()
        
        # Get force sensor readings
        force_readings = np.zeros(self.config.force_sensors)
        for i, sensor in enumerate(self.force_sensors):
            # Extract force magnitude - implementation depends on your sensor setup
            force_readings[i] = 0.0  # Placeholder
        
        # Combine all observations
        obs = np.concatenate([
            joint_pos,
            joint_vel,
            end_effector_pos,
            self.goal_position,
            force_readings
        ])
        
        return obs.astype(np.float32)
    
    def _get_end_effector_position(self):
        """Get current position of the end effector."""
        # TODO: Extract actual position from SOFA state
        return np.zeros(3)  # Placeholder
    
    def _compute_reward(self):
        """Compute reward based on task progress."""
        ee_pos = self._get_end_effector_position()
        
        # Distance to goal
        dist_to_goal = np.linalg.norm(ee_pos - self.goal_position)
        distance_reward = -dist_to_goal * self.config.distance_reward_scale
        
        # Reward for making progress toward goal
        progress_reward = 0.0
        if hasattr(self, 'prev_dist_to_goal'):
            progress = self.prev_dist_to_goal - dist_to_goal
            progress_reward = progress * self.config.progress_reward_scale
        self.prev_dist_to_goal = dist_to_goal
        
        # Penalty for excessive force on maze
        force_penalty = 0.0
        for i, sensor in enumerate(self.force_sensors):
            # Calculate force penalty - implementation depends on your sensor setup
            pass
            
        # Completion reward
        completion_reward = 0.0
        if dist_to_goal < self.config.goal_threshold:
            completion_reward = self.config.completion_reward
            
        total_reward = distance_reward + progress_reward - force_penalty + completion_reward
        return float(total_reward)
    
    def _check_termination(self):
        """Check if episode should terminate."""
        ee_pos = self._get_end_effector_position()
        
        # Check if goal reached
        if np.linalg.norm(ee_pos - self.goal_position) < self.config.goal_threshold:
            return True
            
        # Check for collision that should end episode
        # Implementation depends on your specific collision detection setup
        
        return False
    
    def _get_info(self):
        """Get additional information about the current state."""
        ee_pos = self._get_end_effector_position()
        distance_to_goal = np.linalg.norm(ee_pos - self.goal_position)
        
        info = {
            "distance_to_goal": distance_to_goal,
            "current_step": self.current_step,
            "maze_idx": self.current_maze_idx
        }
        return info
    
    def close(self):
        """Clean up resources."""
        # Free SOFA resources
        if self.root is not None:
            Sofa.Simulation.cleanup(self.root)
            self.root = None

# Example configuration class
class EnvConfig:
    def __init__(self):
        self.maze_dir = "path/to/mazes"
        self.robot_dof = 6
        self.force_sensors = 8
        self.max_episode_steps = 1000
        self.time_step = 0.01
        self.maze_young_modulus = 1000
        self.maze_poisson_ratio = 0.3
        self.maze_mass = 1.0
        self.maze_size = 1.0
        self.action_scale = 0.1
        self.distance_reward_scale = 1.0
        self.progress_reward_scale = 10.0
        self.completion_reward = 100.0
        self.goal_threshold = 0.1

# Example usage:
def main():
    config = EnvConfig()
    env = SofaColonEndoscopeEnv(config)
    
    # Example RL loop
    obs, info = env.reset()
    for _ in range(1000):
        action = np.random.uniform(-1, 1, size=env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            
    env.close()

if __name__ == "__main__":
    main()

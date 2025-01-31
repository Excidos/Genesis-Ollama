import streamlit as st
import multiprocessing as mp
import genesis as gs
import numpy as np
import tempfile
import os
import torch
import json
from typing import Dict, Any, Optional, List
import cv2
import asyncio
from ollama_handler import OllamaHandler
import math

class PhysicsSimulator:
    def __init__(self, model_name: Optional[str] = None, available_meshes: Optional[Dict[str, str]] = None):
        self.ollama = OllamaHandler(model_name=model_name)
        self.model_name = model_name
        self.available_meshes = available_meshes or {}
        self.scene = None  # Initialize scene to None

    async def setup(self):
        """Initial setup including model selection"""
        with st.spinner("Initializing Ollama..."):
            if not self.model_name:
                self.model_name = self.ollama.select_model()
            self.ollama.current_model = self.model_name
            st.success(f"Using model: {self.model_name}")

    def _init_genesis(self):
        """Initialize Genesis with proper GPU handling"""
        try:
            if torch.cuda.is_available():
                gs.init(backend=gs.gpu, seed=0, precision="32", logging_level="debug")
            else:
                gs.init(backend=gs.cpu, seed=0, precision="32", logging_level="debug")
        except Exception as e:
            st.warning(f"GPU initialization failed, falling back to CPU: {e}")
            gs.init(backend=gs.cpu, seed=0, precision="32", logging_level="debug")

    async def generate_simulation(self, description: str, frames: int, output_path: str, selected_mesh: Optional[str] = None) -> bool:
        """Generate and run physics simulation"""
        try:
            # Reinitialize Genesis for each simulation
            if gs._initialized:
                gs.clear()
            self._init_genesis()
            
            # Update the prompt with selected mesh if available
            if selected_mesh and selected_mesh in self.available_meshes:
                mesh_info = f" Use the mesh file '{self.available_meshes[selected_mesh]}' for the simulation."
                description = description + mesh_info

            config = await self.ollama.generate_physics_config(description)
            if not config:
                st.error("Failed to generate simulation configuration")
                return False

            # --- Potential Enhancement: Configuration Refinement ---
            config = self.refine_config(config, description)

            if st.session_state.get('show_config', True):
                st.subheader("Generated Configuration")
                st.json(config)

            # Create a new scene for the simulation
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(**config["scene_config"]["sim_options"]),
                viewer_options=gs.options.ViewerOptions(**config["scene_config"]["viewer_options"]),
                show_viewer=False,
                renderer=gs.renderers.Rasterizer()
            )

            success = self.run_simulation(config, frames, output_path)
            return success

        except Exception as e:
            st.error(f"Error in simulation generation: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False

    def refine_config(self, config: Dict[str, Any], description: str) -> Dict[str, Any]:
        """
        Refine the generated configuration based on further analysis or user feedback.

        Args:
            config: The initial configuration generated by the LLM.
            description: The original user description.

        Returns:
            The refined configuration.
        """
        # 1. Schema Validation (using a library like jsonschema)
        # ... (validate against the Genesis schema) ...

        # 2. Parameter Adjustments (based on reasoning or heuristics)
        if "entities" in config:
            for entity in config["entities"]:
                if entity["type"] == "PBD.Elastic" and "cube" in description.lower():
                    # Example: If it's a soft cube, make sure it has some initial velocity
                    # to show deformation upon impact.
                    if "initial_velocity" not in entity:
                        entity["initial_velocity"] = [0, 0, -5]  # Add an initial downward velocity

        # 3. Camera Logic
        if "follow" in description.lower() and "camera_lookat" in config["scene_config"]["viewer_options"]:
            # If the user wants the camera to follow an object, we need to figure out which one.
            # For simplicity, let's assume they mean the first entity.
            if config["entities"]:
                target_entity_config = config["entities"][0]  # Get the first entity

                # Set the initial camera lookat to the target entity's position.
                if "parameters" in target_entity_config.get("morph",{}):
                    config["scene_config"]["viewer_options"]["camera_lookat"] = target_entity_config["morph"]["parameters"].get("pos", [0, 0, 0])

        return config

    def run_simulation(self, config: Dict[str, Any], frames: int, output_path: str) -> bool:
        """Run the actual Genesis simulation"""
        try:
            # Validate configuration structure
            if "scene_config" not in config or "entities" not in config:
                raise ValueError("Invalid configuration structure")

            # Add camera
            camera = self.scene.add_camera(
                res=(640, 480),
                pos=config["scene_config"]["viewer_options"]["camera_pos"],
                lookat=config["scene_config"]["viewer_options"]["camera_lookat"],
                fov=config["scene_config"]["viewer_options"]["camera_fov"],
                GUI=False
            )

            # Add entities and track success
            entities_added = False
            # Find the target entity for camera tracking (if "follow" is in the description)
            target_entity = None
            if any("follow" in d.lower() for d in [config.get("description", ""), st.session_state.get("description", "")]):
                for entity_config in config["entities"]:
                    try:
                        entity = self._add_entity(self.scene, entity_config)
                        if entity is not None:
                            entities_added = True
                            target_entity = entity  # Set the first added entity as the target
                            break # assume we only follow one
                    except Exception as entity_error:
                        st.warning(f"Error adding entity {entity_config.get('type', 'unknown')}: {entity_error}")
                        continue
            else:
                for entity_config in config["entities"]:
                    try:
                        entity = self._add_entity(self.scene, entity_config)
                        if entity is not None:
                            entities_added = True
                    except Exception as entity_error:
                        st.warning(f"Error adding entity {entity_config.get('type', 'unknown')}: {entity_error}")
                        continue

            if not entities_added:
                raise RuntimeError("No entities could be added to the scene")

            # Build scene
            self.scene.build()

            # Create folder for frames
            frames_dir = tempfile.mkdtemp()
            frame_files = []

            # Camera Following Parameters (if needed)
            if target_entity:
                target_position = np.array(target_entity.get_mesh().get_vertex(0))

            # Simulation loop with progress bar
            progress_bar = st.progress(0)
            for i in range(frames):
                self.scene.step()

                # Update Camera Position for Following
                if target_entity:
                    # Update target position based on the target entity's current position
                    target_position = target_entity.get_pose()[:3, 3]

                    # Smoothly interpolate camera's lookat towards the target position
                    current_lookat = np.array(camera.lookat)
                    new_lookat = current_lookat + (target_position - current_lookat) * 0.1
                    camera.lookat = new_lookat.tolist()

                rgb, _, _, _ = camera.render(rgb=True)
                frame = (rgb * 255).astype(np.uint8)
                
                # Check for invalid frames
                if np.isnan(frame).any() or np.isinf(frame).any():
                    print(f"Invalid frame detected at frame {i}")
                else:
                    frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frame_files.append(frame_path)

                # Update progress
                progress = (i + 1) / frames
                progress_bar.progress(progress)

            # Create video using ffmpeg if available
            try:
                import ffmpeg

                # Use ffmpeg-python to create video
                (
                    ffmpeg
                    .input(os.path.join(frames_dir, 'frame_%04d.png'), pattern_type='sequence', framerate=30)
                    .output(output_path, vcodec='libx264', pix_fmt='yuv420p')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ImportError:
                # Fallback to OpenCV
                print("Using OpenCV for video creation...")
                frame = cv2.imread(frame_files[0])
                height, width = frame.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Try H.264 codec
                out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

                for frame_path in frame_files:
                    frame = cv2.imread(frame_path)
                    out.write(frame)

                out.release()
            except ffmpeg._run.Error as e:
                print(f"FFmpeg error: {e.stderr.decode()}")
                return False

            # Clean up frames
            for frame_path in frame_files:
                os.remove(frame_path)
            os.rmdir(frames_dir)

            if len(frame_files) == 0:
                raise RuntimeError("No frames were generated")

            return True

        except Exception as e:
            st.error(f"Error in simulation: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False

    def _add_entity(self, scene: gs.Scene, config: Dict[str, Any]) -> None:
        """Add an entity to the scene with better error handling"""
        try:
            # Get material type
            material_parts = config["type"].split('.')
            material_type = gs.materials
            for part in material_parts:
                material_type = getattr(material_type, part)

            # Get morph type
            morph_type = getattr(gs.morphs, config["morph"]["type"])

            # Handle mesh file paths
            morph_params = config["morph"].get("parameters", {}).copy()
            if "file" in morph_params:
                mesh_file = morph_params["file"]

                # Find the correct path to the mesh file
                if mesh_file:
                    mesh_file_path = self.ollama.find_mesh_file(mesh_file)
                    if mesh_file_path:
                        morph_params["file"] = mesh_file_path
                    else:
                        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

            # Handle material parameters
            material_params = config.get("material_params", {})
            if isinstance(material_type, gs.materials.PBD.Liquid):
                material_params.update({
                    "rho": 1.0,
                    "density_relaxation": 1.0,
                    "viscosity_relaxation": 0.0,
                    "sampler": "regular"
                })

            # Set vis_mode to "particle" for liquid entities
            surface_params = config.get("surface", {
                "color": [0.8, 0.8, 0.8, 1.0]
            })
            if "Liquid" in config["type"]:
                surface_params["vis_mode"] = "particle"
            else:
                surface_params["vis_mode"] = "visual"

            # Create entity
            try:
                entity = scene.add_entity(
                    material=material_type(**material_params),
                    morph=morph_type(**morph_params),
                    surface=gs.surfaces.Default(**surface_params)
                )
                return entity
            except Exception as e:
                raise RuntimeError(f"Failed to create entity: {str(e)}")

        except Exception as e:
            raise ValueError(f"Error creating entity: {str(e)}")

    def _get_material_type(self, type_str: str):
        """Get material type from string representation"""
        parts = type_str.split('.')
        material = gs.materials
        for part in parts:
            material = getattr(material, part)
        return material

    def _create_video(self, frame_files: List[str], output_path: str) -> None:
        """Create video from frames"""
        try:
            import ffmpeg
            (
                ffmpeg
                .input(os.path.join(os.path.dirname(frame_files[0]), 'frame_%04d}.png'),
                       pattern_type='sequence')
                .output(output_path, vcodec='libx264', pix_fmt='yuv420p', r=30)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ImportError:
            # Fallback to OpenCV
            frame = cv2.imread(frame_files[0])
            height, width = frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

            for frame_path in frame_files:
                frame = cv2.imread(frame_path)
                out.write(frame)

            out.release()

async def run_simulation_async(simulator, description, frames, output_path, selected_mesh):
    """Run the simulation asynchronously"""
    success = await simulator.generate_simulation(description, frames, output_path, selected_mesh)
    return success

def main():
    st.title("AI-Powered Physics Simulator")

    # Initialize handler
    handler = OllamaHandler()

    # Get available models and meshes
    available_models = handler.get_available_models()
    available_meshes = handler.get_available_meshes()

    # Sidebar for model selection and advanced options
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox(
            "Select Model",
            options=available_models,
            help="Choose the Ollama model to use for generating simulations"
        )

        st.header("Advanced Options")
        st.session_state.show_config = st.checkbox("Show Generated Configuration", value=True)

        # Mesh selection if meshes are available
        selected_mesh = None
        if available_meshes:
            selected_mesh = st.selectbox(
                "Select Mesh",
                options=list(available_meshes.keys()),
                help="Choose a mesh for the simulation"
            )
            if selected_mesh:
                st.info(f"Selected mesh: {available_meshes[selected_mesh]}")
        else:
            st.warning("No meshes found in the expected locations")

    # Main interface
    st.markdown("""
    Describe the physics simulation you want to see. You can specify:
    - Objects and their properties (size, material, position)
    - Forces and interactions
    - Camera angle and perspective
    - Duration and speed
    """)

    description = st.text_area(
        "Simulation Description",
        help="Example: 'Show me a red bouncing ball dropping onto a soft blue surface from 1 meter height'",
        height=100
    )

    col1, col2 = st.columns(2)

    with col1:
        frames = st.slider(
            "Number of Frames",
            min_value=10,
            max_value=200,
            value=50,
            help="More frames = longer simulation but smoother motion"
        )

    with col2:
        fps = st.slider(
            "Frames per Second",
            min_value=15,
            max_value=60,
            value=30,
            help="Higher FPS = smoother playback"
        )

    if st.button("Generate Simulation", type="primary"):
        if not description:
            st.warning("Please enter a simulation description")
            return

        try:
            # Create temporary file for video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                output_path = tmp_file.name

            # Initialize simulator with selected model
            simulator = PhysicsSimulator(model, available_meshes)

            # Run simulation asynchronously
            with st.spinner("Running simulation..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(
                    run_simulation_async(simulator, description, frames, output_path, selected_mesh)
                )
                loop.close()

            # Display video if successful
            if success and os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
                    if len(video_bytes) > 0:
                        st.success("Simulation completed!")
                        st.video(video_bytes)
                    else:
                        st.error("Generated video file is empty")
                os.unlink(output_path)

        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    mp.freeze_support()
    main()
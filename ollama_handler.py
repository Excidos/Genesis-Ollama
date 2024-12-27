import asyncio
import aiohttp
import subprocess
import json
from typing import Dict, Any, List, Optional
from pick import pick
import logging
import os
import requests
import streamlit as st

# Comprehensive prompt templates for physics understanding
PHYSICS_SYSTEM_PROMPT = """You are an expert physics simulation architect using the Genesis physics engine. Your role is to:

1. **Carefully analyze** natural language descriptions of physics simulations.
2. **Extract key information** about objects, materials, forces, interactions, and camera settings.
3. **Infer the appropriate Genesis parameters** to accurately represent the described scenario.
4. **Generate valid Genesis configuration JSON** that produces the desired simulation.

**Important Instructions:**

*   **Camera:**
    *   **ALWAYS include `camera_pos` and `camera_lookat` inside `viewer_options`.** These are essential for camera setup.
    *   If the user specifies a camera angle, perspective, or movement (e.g., "following the object"), determine the appropriate `camera_pos` and `camera_lookat` values.
    *   If the user does not specify camera details, use a default `camera_pos` of `[5, 5, 5]` and `camera_lookat` of `[0, 0, 0]`.
*   **JSON Format:**
    *   The output MUST be valid JSON.
    *   **DO NOT include any extra spaces, newlines, or characters that would make the JSON invalid.**
    *   Double-check the JSON structure, especially when adding new parameters.

**Available Materials and Their Uses:**

*   **MPM.Elastic:** For soft, deformable objects that maintain their shape (e.g., rubber ball, soft robot).
*   **PBD.Elastic:** For cloth-like materials or soft bodies that can stretch and deform (e.g., fabric, jelly).
*   **PBD.Liquid:** For simulating liquids using Position-Based Dynamics (e.g., water, honey).
*   **SPH.Liquid:** For simulating liquids using Smoothed Particle Hydrodynamics (more accurate but potentially slower).
*   **Rigid:** For solid, non-deformable objects (e.g., a metal cube, a wooden block).

**Available Morphs (Shapes):**

*   **Box:** `parameters: { pos: [x, y, z], size: [width, height, depth] }`
*   **Sphere:** `parameters: { pos: [x, y, z], radius: float }`
*   **Plane:** `parameters: {}` (for ground planes)
*   **Mesh:** `parameters: { file: "meshes/bunny.obj", scale: float, pos: [x, y, z] }`

**Important Considerations:**

*   **Object Properties:** Pay close attention to object properties like size, material, initial position, and velocity.
*   **Forces:** By default, gravity is [0, 0, -9.81]. Consider if other forces (e.g., wind, applied force) are mentioned or implied.
*   **Interactions:** If objects are meant to collide, ensure their positions and parameters allow for interaction.
*   **Units:** Genesis uses meters for distance, kilograms for mass, and seconds for time.

**Example Scenarios and Configurations:**

**(Add diverse examples here, including examples with camera following, different materials, and interactions.)**

**Example 1: Bouncing Ball**
content_copy
download
Use code with caution.
Python

Description: "A red rubber ball bouncing on a hard surface. The camera should be 5 meters away, looking at the origin."

Configuration:
{
"scene_config": {
"sim_options": {
"dt": 0.01,
"substeps": 10,
"gravity": [0, 0, -9.81]
},
"viewer_options": {
"camera_pos": [5, 0, 2],
"camera_lookat": [0, 0, 0],
"camera_fov": 40,
"res": [640, 480]
}
},
"entities": [
{
"type": "MPM.Elastic",
"morph": {
"type": "Sphere",
"parameters": {
"radius": 0.2,
"pos": [0, 0, 2.0]
}
},
"surface": {
"color": [1.0, 0.2, 0.2, 1.0],
"vis_mode": "visual"
}
},
{
"type": "Rigid",
"morph": {
"type": "Plane",
"parameters": {}
},
"surface": {
"color": [0.8, 0.8, 0.8, 1.0],
"vis_mode": "visual"
}
}
]
}

**Example 2: Falling Cube with Camera Following**
content_copy
download
Use code with caution.

Description: "A 1 meter wide blue cube made of PBD.Elastic material falling from a height of 10 meters. The camera should follow the cube's descent."

Configuration:
{
"scene_config": {
"sim_options": {
"dt": 0.01,
"substeps": 5,
"gravity": [0, -9.81, 0]
},
"viewer_options": {
"camera_pos": [10, 10, 10],
"camera_lookat": [0, 5, 0],
"camera_fov": 45,
"res": [640, 480]
}
},
"entities": [
{
"type": "PBD.Elastic",
"morph": {
"type": "Box",
"parameters": {
"pos": [0, 10, 0],
"size": [1, 1, 1]
}
},
"surface": {
"color": [0.2, 0.2, 1.0, 1.0],
"vis_mode": "visual"
}
},
{
"type": "Rigid",
"morph": {
"type": "Plane",
"parameters": {}
},
"surface": {
"color": [0.8, 0.8, 0.8, 1.0],
"vis_mode": "visual"
}
}
]
}

**Constraints:**

*   **Valid JSON:** The output MUST be valid JSON.
*   **Genesis Schema:** The JSON MUST conform to the Genesis configuration schema.
*   **Parameter Types:** Use the correct parameter types (e.g., lists for `pos`, floats for `scale`).
*   **Realistic Values:** Choose parameter values that are physically realistic and appropriate for the simulation.

**Generate ONLY the JSON configuration.** Do not include any additional text or explanations.
"""

# Example configurations for common scenarios
EXAMPLE_CONFIGS = {
    "bouncing_ball": {
        "scene_config": {
            "sim_options": {
                "dt": 0.01,
                "substeps": 4,
                "gravity": [0, 0, -9.81]
            },
            "viewer_options": {
                "camera_pos": [2, 2, 1.5],
                "camera_lookat": [0, 0, 0.5],
                "camera_fov": 40
            }
        },
        "entities": [
            {
                "type": "PBD.Elastic",
                "morph": {
                    "type": "Sphere",
                    "parameters": {
                        "radius": 0.1,
                        "pos": [0, 0, 1.0]
                    }
                },
                "surface": {
                    "color": [1.0, 0.5, 0.2, 1.0],
                    "vis_mode": "visual"
                }
            },
            {
                "type": "Rigid",
                "morph": {
                    "type": "Plane",
                    "parameters": {}
                },
                "surface": {
                    "color": [0.8, 0.8, 0.8, 1.0],
                    "vis_mode": "visual"
                }
            }
        ]
    }
}

class ConfigurationAdapter:
    """Adapts various physics configuration formats to Genesis schema"""

    @staticmethod
    def adapt_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert any supported config format to Genesis schema"""
        try:
            # Ensure basic structure
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a dictionary")

            # Create base structure if missing
            if "scene_config" not in config:
                config["scene_config"] = {
                    "sim_options": {
                        "dt": 0.01,
                        "substeps": 10,
                        "gravity": [0, 0, -9.81]
                    },
                    "viewer_options": {
                        "camera_pos": [2, 2, 1.5],
                        "camera_lookat": [0, 0, 0.5],
                        "camera_fov": 40,
                        "res": [640, 480]
                    }
                }

            if "entities" not in config:
                config["entities"] = []

            # Clean and validate entities
            valid_entities = []
            for entity in config["entities"]:
                try:
                    # Fix material type names
                    if entity.get("type") == "Rigid Body":
                        entity["type"] = "Rigid"
                    elif not any(typ in entity.get("type", "") for typ in ["MPM", "PBD", "SPH", "Rigid"]):
                        entity["type"] = "Rigid"  # Default to Rigid if unknown

                    # Fix vis_mode for liquid entities
                    if "Liquid" in entity.get("type", ""):
                        entity["surface"]["vis_mode"] = "particle"

                    # Ensure morph section exists
                    if "morph" not in entity:
                        entity["morph"] = {
                            "type": "Box",
                            "parameters": {
                                "pos": [0, 0, 1],
                                "size": [1, 1, 1]
                            }
                        }

                    # Fix parameter names
                    if "parameters" not in entity["morph"]:
                        entity["morph"]["parameters"] = {}

                    params = entity["morph"]["parameters"]
                    if "file_path" in params:
                        params["file"] = params.pop("file_path")
                    if "position" in params:
                        params["pos"] = params.pop("position")

                    # Handle mesh file paths (find actual location)
                    if "file" in params:
                        mesh_file = params["file"]
                        # Check different possible mesh locations
                        possible_paths = [
                            mesh_file,
                            os.path.join("assets", mesh_file),
                            os.path.join("meshes", os.path.basename(mesh_file)),
                            os.path.join("/usr/local/lib/python3.9/site-packages/genesis/assets", mesh_file)
                        ]
                        mesh_found = False
                        for path in possible_paths:
                            if os.path.exists(path):
                                params["file"] = path
                                mesh_found = True
                                break
                        if not mesh_found:
                            print(f"Warning: Mesh file not found in any location: {mesh_file}")

                    # Add surface properties if missing
                    if "surface" not in entity:
                        entity["surface"] = {
                            "color": [0.8, 0.8, 0.8, 1.0],
                            "vis_mode": "visual"
                        }

                    valid_entities.append(entity)
                except Exception as e:
                    st.warning(f"Skipping invalid entity: {e}")
                    continue

            config["entities"] = valid_entities

            # Ensure there's a ground plane
            has_plane = any(
                e.get("type") == "Rigid" and e["morph"]["type"] == "Plane"
                for e in config["entities"]
            )

            if not has_plane:
                config["entities"].append({
                    "type": "Rigid",
                    "morph": {
                        "type": "Plane",
                        "parameters": {}
                    },
                    "surface": {
                        "color": [0.8, 0.8, 0.8, 1.0],
                        "vis_mode": "visual"
                    }
                })

            return config

        except Exception as e:
            st.error(f"Configuration adaptation error: {e}")
            return {
                "scene_config": {
                    "sim_options": {
                        "dt": 0.01,
                        "substeps": 10,
                        "gravity": [0, 0, -9.81]
                    },
                    "viewer_options": {
                        "camera_pos": [2, 2, 1.5],
                        "camera_lookat": [0, 0, 0.5],
                        "camera_fov": 40,
                        "res": [640, 480]
                    }
                },
                "entities": [
                    {
                        "type": "Rigid",
                        "morph": {
                            "type": "Plane",
                            "parameters": {}
                        },
                        "surface": {
                            "color": [0.8, 0.8, 0.8, 1.0],
                            "vis_mode": "visual"
                        }
                    }
                ]
            }

    @staticmethod
    def _convert_simulation_format(sim_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert simulation format to Genesis schema"""
        genesis_config = {
            "scene_config": {
                "sim_options": {
                    "dt": sim_config.get("time_step", 0.01),
                    "substeps": 4,  # Default value
                    "gravity": [
                        sim_config["gravity"].get("x", 0),
                        sim_config["gravity"].get("y", -9.81),
                        sim_config["gravity"].get("z", 0)
                    ]
                },
                "viewer_options": {
                    "camera_pos": [
                        sim_config["camera"]["position"]["x"],
                        sim_config["camera"]["position"]["y"],
                        sim_config["camera"]["position"]["z"]
                    ],
                    "camera_lookat": [
                        sim_config["camera"]["look_at"]["x"],
                        sim_config["camera"]["look_at"]["y"],
                        sim_config["camera"]["look_at"]["z"]
                    ],
                    "camera_fov": 40,  # Default value
                    "res": [640, 480]  # Default resolution
                }
            },
            "entities": []
        }

        # Convert objects to entities
        for obj in sim_config.get("objects", []):
            material_type = ConfigurationAdapter._determine_material_type(obj)

            entity = {
                "type": material_type,
                "morph": {
                    "type": "Box" if obj["name"] == "car" else "Mesh",
                    "parameters": {
                        "pos": [obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]],
                        "size": [2.0, 1.0, 1.0] if obj["name"] == "car" else [0.5, 0.5, 0.5]
                    }
                },
                "surface": {
                    "color": [0.8, 0.8, 0.8, 1.0],
                    "vis_mode": "visual"
                }
            }

            genesis_config["entities"].append(entity)

        # Add ground plane if not present
        if not any(e.get("type") == "Rigid" and e["morph"]["type"] == "Plane"
                  for e in genesis_config["entities"]):
            genesis_config["entities"].append({
                "type": "Rigid",
                "morph": {
                    "type": "Plane",
                    "parameters": {}
                },
                "surface": {
                    "color": [0.8, 0.8, 0.8, 1.0],
                    "vis_mode": "visual"
                }
            })

        return genesis_config

    @staticmethod
    def _determine_material_type(obj: Dict[str, Any]) -> str:
        """Determine appropriate Genesis material type based on object properties"""
        if "material" in obj:
            material = obj["material"].lower()
            if "fluid" in material or "water" in material:
                return "SPH.Liquid"
            elif "cloth" in material or "fabric" in material:
                return "PBD.Cloth"
            elif "soft" in material or obj.get("mass", 1000) < 10:
                return "PBD.Elastic"
        return "Rigid"

class OllamaHandler:
    def __init__(self, base_url: str = "http://localhost:11434", model_name: Optional[str] = None):
        self.base_url = base_url
        self.logger = logging.getLogger('OllamaHandler')
        self.current_model = model_name # Set current_model from parameter
        self.mesh_base_path = os.path.join(os.path.dirname(__file__), "assets/meshes")

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models] if models else ["mistral"]
            else:
                self.logger.warning("Error getting Ollama models, falling back to default")
                return ["mistral"]
        except Exception as e:
            self.logger.error(f"Error getting Ollama models: {e}")
            return ["mistral"]

    def get_available_meshes(self) -> Dict[str, str]:
        """Get dictionary of available meshes and their paths"""
        mesh_dict = {}
        try:
            # Check both potential mesh locations
            mesh_paths = [
                self.mesh_base_path,
                os.path.join(os.path.dirname(__file__), "meshes"),
                "/usr/local/lib/python3.9/site-packages/genesis/assets/meshes"
            ]
            
            for base_path in mesh_paths:
                if os.path.exists(base_path):
                    for root, dirs, files in os.walk(base_path):
                        for file in files:
                            if file.endswith('.obj'):
                                name = os.path.splitext(file)[0]
                                full_path = os.path.join(root, file)
                                mesh_dict[name] = full_path
            
            return mesh_dict
        except Exception as e:
            self.logger.error(f"Error getting meshes: {e}")
            return {}

    def find_mesh_file(self, mesh_file: str) -> Optional[str]:
        """
        Find the actual path to a mesh file, handling different possible locations and searching recursively in subdirectories.
        """
        # Possible base paths to check:
        possible_base_paths = [
            self.mesh_base_path,
            os.path.join(os.path.dirname(__file__), "meshes"),
            "/usr/local/lib/python3.9/site-packages/genesis/assets/meshes"  # You might need to adjust this path.
        ]

        # Check if the provided path is already a valid path
        if os.path.exists(mesh_file):
            return mesh_file
        
        # Check if only the filename is provided
        mesh_filename = os.path.basename(mesh_file)

        # Search in base paths and their subdirectories:
        for base_path in possible_base_paths:
            if os.path.exists(base_path):
                for root, _, files in os.walk(base_path):
                    if mesh_filename in files:
                        return os.path.join(root, mesh_filename)

        return None

    def select_model(self) -> str:
        """Interactive model selection using pick"""
        try:
            models = self.get_available_models()
            if not models:
                self.logger.warning("No Ollama models found, using default")
                return "mistral"
            
            # Let user select model
            title = 'Choose an Ollama model (press SPACE to select, ENTER to confirm):'
            selected_model, _ = pick(models, title)
            self.logger.info(f"Selected model: {selected_model}")
            self.current_model = selected_model
            return selected_model
        except Exception as e:
            self.logger.error(f"Error selecting Ollama model: {e}")
            return "mistral"

    async def generate_physics_config(self, description: str) -> Optional[Dict[str, Any]]:
            """Generate physics simulation configuration from description"""
            if not self.current_model:
                # Only call select_model if not already set (e.g., by Streamlit)
                self.current_model = self.select_model()

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": self.current_model,
                            "messages": [
                                {"role": "system", "content": PHYSICS_SYSTEM_PROMPT},
                                {"role": "user", "content": description}
                            ],
                            "format": "json",
                            "stream": False
                        }
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"Ollama API error: {await response.text()}")
                        
                        result = await response.json()
                        if 'message' in result:
                            try:
                                raw_config = json.loads(result['message']['content'])
                                # Adapt configuration to Genesis format
                                adapted_config = ConfigurationAdapter.adapt_config(raw_config)
                                return adapted_config
                            except json.JSONDecodeError as e:
                                self.logger.error(f"Invalid JSON in response: {e}")
                                self.logger.debug(f"Raw response: {result['message']['content']}")
                        
                        raise Exception("Invalid response format from Ollama")

            except Exception as e:
                self.logger.error(f"Error generating physics configuration: {e}")
                return None

    async def stream_response(self, messages: List[Dict[str, str]], callback) -> None:
        """Stream response from Ollama with callback for each chunk"""
        if not self.current_model:
             # Only call select_model if not already set (e.g., by Streamlit)
            self.current_model = self.select_model()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.current_model,
                        "messages": messages,
                        "stream": True
                    }
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Ollama API error: {await response.text()}")
                    
                    async for line in response.content:
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            if 'message' in data and 'content' in data['message']:
                                await callback(data['message']['content'])
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            self.logger.error(f"Error processing response chunk: {e}")
                            continue

        except Exception as e:
            self.logger.error(f"Error in streaming response: {e}")
            await callback(f"Error: {str(e)}")

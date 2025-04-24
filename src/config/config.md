# Moriarty Configuration Module Documentation

This document provides detailed descriptions of the functions in the configuration module (`src/config`). The module contains utilities for generating pose configurations and setting up the RAG (Retrieval-Augmented Generation) system.

## 1. `generate_pose_configs.py`

This script generates pose configurations for React Three Fiber GLB models using AI models as teachers.

### `PoseConfigGenerator` Class

#### `__init__(self, base_model_path: str = "models/pose_model.pth")`
- **Description**: Initializes the PoseConfigGenerator with a base model path.
- **Parameters**:
  - `base_model_path` (str): Path to the base pose model (default: "models/pose_model.pth")
- **Behavior**: 
  - Sets up the device (CPU/CUDA)
  - Initializes OpenAI and Anthropic API clients
  - Loads the base pose model
  - Defines the structure of pose configurations

#### `generate_config_examples(self, num_examples: int = 100) -> List[Dict[str, Any]]`
- **Description**: Generates pose configuration examples using GPT-4 and Claude.
- **Parameters**:
  - `num_examples` (int): Number of examples to generate (default: 100)
- **Returns**: List of dictionaries containing query-configuration pairs
- **Behavior**:
  - Creates example queries about pose configurations
  - Gets responses from teacher models (GPT-4 and Claude)
  - Extracts pose data for random frames
  - Creates training examples with queries, frame numbers, teacher responses, and pose configurations

#### `_get_teacher_responses(self, query: str) -> Dict[str, str]`
- **Description**: Gets responses from both OpenAI and Anthropic models.
- **Parameters**:
  - `query` (str): The query about pose configuration
- **Returns**: Dictionary with responses from both models
- **Behavior**:
  - Queries GPT-4 with specific instructions for GLB configuration
  - Queries Claude-3-Opus with the same query
  - Returns both responses in a dictionary

#### `_extract_pose_data(self, frame: int) -> Dict[str, Any]`
- **Description**: Extracts pose data for a specific frame from the base model.
- **Parameters**:
  - `frame` (int): Frame number to extract data from
- **Returns**: Dictionary containing frame data
- **Behavior**: Retrieves the data for the specified frame from the base model

#### `_create_pose_config(self, pose_data: Dict[str, Any]) -> Dict[str, Any]`
- **Description**: Creates a pose configuration suitable for GLB models.
- **Parameters**:
  - `pose_data` (Dict[str, Any]): Raw pose data from the base model
- **Returns**: Structured pose configuration
- **Behavior**:
  - Converts raw pose data into the required configuration format
  - Processes joint positions and rotations
  - Calculates joint angles
  - Calculates center of mass

#### `_calculate_joint_angles(self, pose_data: Dict[str, Any]) -> Dict[str, float]`
- **Description**: Calculates joint angles from pose data.
- **Parameters**:
  - `pose_data` (Dict[str, Any]): Raw pose data
- **Returns**: Dictionary mapping joint names to angles
- **Behavior**: Calculates angles for major joints based on position and rotation data

#### `_calculate_center_of_mass(self, pose_data: Dict[str, Any]) -> List[float]`
- **Description**: Calculates the center of mass of the pose.
- **Parameters**:
  - `pose_data` (Dict[str, Any]): Raw pose data
- **Returns**: List representing the 3D coordinates of the center of mass
- **Behavior**: Averages the positions of all joints to find the center of mass

#### `_calculate_angle(self, position: np.ndarray, rotation: np.ndarray) -> float`
- **Description**: Calculates the angle from position and rotation data.
- **Parameters**:
  - `position` (np.ndarray): 3D position data
  - `rotation` (np.ndarray): Rotation data
- **Returns**: Angle in degrees
- **Behavior**: Placeholder implementation in the current code

### `main()`
- **Description**: Command-line interface for generating pose configurations.
- **Behavior**:
  - Parses command-line arguments for base model path, number of examples, and output directory
  - Creates a PoseConfigGenerator
  - Generates configuration examples
  - Saves examples to a JSON file

## 2. `setup_rag.py`

This script sets up and runs the Retrieval-Augmented Generation (RAG) system for sports video analysis.

### `check_dependencies()`
- **Description**: Checks if all required dependencies are installed.
- **Returns**: Boolean indicating if all dependencies are installed
- **Behavior**:
  - Attempts to import all required packages
  - Reports any missing dependencies
  - Suggests installing from requirements.txt if any are missing

### `check_api_key()`
- **Description**: Checks if the OpenAI API key is set in the environment.
- **Returns**: Boolean indicating if the API key is available
- **Behavior**:
  - Checks for OPENAI_API_KEY in environment variables
  - If not found, prompts the user to enter it
  - Saves the entered key to a .env file
  - Sets the API key in the environment

### `check_processed_videos()`
- **Description**: Checks if there are processed videos in the output directory.
- **Returns**: Boolean indicating if processed videos were found
- **Behavior**:
  - Verifies that the output directory exists
  - Looks for annotated video files in the output directory
  - Lists the found videos if any

### `extract_data()`
- **Description**: Extracts data from processed videos.
- **Returns**: Boolean indicating success or failure
- **Behavior**:
  - Creates a data store directory if it doesn't exist
  - Imports and uses the VideoDataExtractor
  - Extracts data from all videos with a specified sample rate
  - Reports the number of videos processed

### `setup_vector_db()`
- **Description**: Sets up and indexes the vector database.
- **Returns**: Boolean indicating success or failure
- **Behavior**:
  - Initializes the RAG system
  - Loads and indexes video data
  - Reports the number of documents indexed

### `start_api_server()`
- **Description**: Starts the API server.
- **Returns**: Subprocess object if successful, None otherwise
- **Behavior**:
  - Starts the API server as a subprocess
  - Waits for the server to initialize
  - Checks if the process is still running
  - Reports any errors

### `test_query(query="What sports are shown in the videos?")`
- **Description**: Tests querying the RAG system.
- **Parameters**:
  - `query` (str): The query to test (default: "What sports are shown in the videos?")
- **Returns**: Boolean indicating success or failure
- **Behavior**:
  - Imports the client and queries the system
  - Displays the response if successful
  - Reports any errors

### `main()`
- **Description**: Main function to set up and run the RAG system.
- **Behavior**:
  - Parses command-line arguments
  - Runs dependency and API key checks
  - Extracts data from videos
  - Sets up the vector database
  - Starts the API server
  - Tests the system with a query
  - Provides instructions for using the system
  - Keeps the server running until interrupted

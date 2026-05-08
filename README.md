
# UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents

----------

## Notes

We are currently collecting feedback to help improve this work. A major v2.0 release is planned in the next 2–3 months, and we would greatly appreciate your input in shaping its development and improvement.

Please share your suggestions and comments through the following form: [Feedback Form](https://forms.gle/wQ8Ypw2x2kUb18XVA)

## Content

- [Introduction](#introduction)
- [Demo](#demo)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [TODO](#todo)
- [Acknowledgment](#acknowledgment)

## Introduction

<p align="center">
  <img src="image/task_demo.png" width="90%">
</p>

Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on step-by-step linguistic instructions, limiting its scalability and autonomy. To bridge this gap, we propose **UAV-ON**, a benchmark designed to facilitate research on large-scale **Object Goal Navigation (ObjectNav)** by aerial agents operating in open-world environments. UAV-ON comprises **14 high-fidelity Unreal Engine environments** with diverse semantic regions and complex spatial layouts, covering **urban, natural, and mixed-use** settings. It defines **1270 annotated target objects**, each paired with a structured semantic prompt that encodes category, estimated physical footprint, and detailed visual descriptors, allowing grounded reasoning. These prompts serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. We also propose **Aerial ObjectNav Agent (AOA)**, a modular baseline policy that integrates prompt semantics with egocentric observations to perform long-horizon, goal-directed exploration. Empirical results demonstrate that standard baselines perform poorly in this setting, underscoring the compounded difficulty of aerial navigation and semantic goal grounding. **UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments**.

**For detailed supplementary information about this project, please refer to the appendix: [Click to view appendix](https://drive.google.com/file/d/11nc_SmsQ5fDNz_wON3vqqATLWKm251Ik/view?usp=drive_link)**

## Demo

Watch a full successful flight of our Aerial ObjectNav Agent in action:

<p align="center">
  <a href="https://youtu.be/Zx-Bhzc5Cv4">
    <img src="https://img.youtube.com/vi/Zx-Bhzc5Cv4/0.jpg" alt="UAV-ON Demo" width="70%"/>
  </a>
</p>

> Click the image above or [this link](https://youtu.be/Zx-Bhzc5Cv4) to view the demo video.



## Getting Started

- **Step1: Configure the Python environment**

    For the full AOA/CLIP baselines, install all dependencies:

    ```bash
    conda create -n uavon python==3.8
    conda activate uavon
    pip install -r requirements.txt
    ```

    For the 3D A* oracle baseline, a lighter AirSim-focused environment is enough. This avoids installing large model dependencies such as PyTorch and Transformers when only the simulator, collision checks, and voxel grid generation are needed:

    ```bash
    # Optional: remove a large unused environment first.
    conda env remove -n vllm -y

    conda create -n uavon python==3.8 -y
    conda activate uavon

    # AirSim 1.8.1 imports numpy and msgpackrpc during setup metadata generation,
    # so install these first instead of relying on requirements.txt order.
    pip install numpy==1.24.4 msgpack-python==0.5.6 msgpack-rpc-python==0.4.1

    # Minimal dependencies for AirSim server/client and the A* oracle.
    pip install airsim==1.8.1 opencv-contrib-python==4.8.0.76 tqdm==4.66.1 attrs==23.1.0
    pip install yacs==0.1.8 numba==0.57.1 llvmlite==0.40.1
    ```

    Verify the A* simulation stack:

    ```bash
    python -c "import airsim; print(airsim.__version__, hasattr(airsim.MultirotorClient, 'simCreateVoxelGrid'))"
    python src/eval_astar.py --help
    python airsim_plugin/AirVLNSimulatorServerTool.py --help
    pip check
    ```

    `simCreateVoxelGrid` must print `True`; the A* oracle uses this AirSim API to create binvox occupancy grids for planning.

- **Step2: Prepare the simulation environment**

    You can get UAV-ON train environments from [train envs](https://huggingface.co/datasets/Kyaren/UAV-ON-envs-train) (44.1G) and [test envs](https://huggingface.co/datasets/Kyaren/UAV-ON-envs-test) (26.8G)
    The environment directory should be structured as follows:

    ``` text
    TRAIN_ENVS/
    ├── Barnyard/
    ├── BrushifyRoad/
    ├── CabinLake/
    └── ... (other training environments)

    TEST_ENVS/
    ├── Barnyard/
    ├── BrushifyRoad/
    ├── CabinLake/
    └── ... (other testing environments)
    ```

- **Step3: Get dataset json files**

    You can download dataset from [here](https://huggingface.co/datasets/Kyaren/UAV-ON-dataset)，you can use [script](https://github.com/Kyaren/UAV_ON/tree/main/scripts) to merge split data files into a single JSON file

- **Project directory structure**
  
  Your workspace directory should be structured as follows:
  
  ```text
  workspace/
  ├── UAV_ON/        
  ├── DATASET/        
  ├── TRAIN_ENVS/
  └── TEST_ENVS/
  ```

## Usage
  
  1.First, you should launch the AirSim environment server.

  ```bash
  python airsim_plugin/AirVLNSimulatorServerTool.py --port=30000 --root_path= "your workspace path"
  ```

  2.Then, you can execute the bash script to run the simulator

  ```bash
  #AOA-F/V
  bash scripts/eval_fixed.sh
  bash scripts/eval_unfixed.sh
  #CLIP-H
  bash scripts/eval_cliph.sh
  #3D A* oracle
  bash scripts/eval_astar.sh

  bash scripts/metric.sh
  ```

  If you encounter the "**Ping returned false**" error and **no output in server console**, this is caused by the package version. You can run the following command:

  ```bash 
  pip uninstall msgpack-python msgpack-rpc-python
  pip install msgpack-rpc-python
  ```

### 3D A* Trajectory Videos

The repository includes helper scripts for debugging and visualizing the 3D A* oracle on an already-open AirSim scene:

```bash
# Launch the test AirSim server. Use a real NVIDIA adapter for image capture.
UAV_ON_DATA_ROOT=/path/to/uav-on-data \
UAV_ON_SIM_PORT=30000 \
UAV_ON_SERVER_GPUS=0 \
bash scripts/start_server_test.sh

# After the CityPark_test scene is open, run a full A* rollout.
# The first AirSim scene port is normally UAV_ON_SIM_PORT + 100.
python scripts/smoke_astar_existing_scene.py \
  --dataset astar_smoke_citypark_1.json \
  --output-dir astar_logs/full_citypark_episode0 \
  --port 30100 \
  --max-actions 100

# Render a four-camera 2x2 video from trajectory.jsonl.
python scripts/render_uavon_trajectory_video.py \
  --trajectory astar_logs/full_citypark_episode0/log/trajectory.jsonl \
  --output astar_logs/full_citypark_episode0/astar_citypark_4view_full.mp4 \
  --port 30100 \
  --offset 0,0,0 \
  --fps 10

# Render a top-down OpenCV trajectory video that does not require UE image capture.
python scripts/render_trajectory_topdown_video.py \
  --trajectory astar_logs/full_citypark_episode0/log/trajectory.jsonl \
  --dataset astar_smoke_citypark_1.json \
  --output astar_logs/full_citypark_episode0/astar_citypark_topdown_full.mp4
```

For real four-view video generation, Unreal must render with the NVIDIA GPU. `AirVLNSimulatorServerTool.py` starts packaged UE scenes with `-RenderOffscreen`, `-NoSound`, `-NoVSync`, and `-GraphicsAdapter=<gpu_id>`. In root-based containers, the tool launches only the Unreal process through the `uavonrunner` user because packaged UE4 Linux builds refuse to run as root. If `simGetImages` hangs, returns empty frames, or UE logs show `RenderThread` timeouts, check that the server was launched with a valid GPU id and that Vulkan/OpenGL did not fall back to `llvmpipe`.

`render_uavon_trajectory_video.py` replays trajectory poses while the simulator is paused, so the video reflects the saved trajectory instead of letting physics settle the drone into nearby geometry. Add `--simulate-settle` only when you explicitly want to unpause after each pose and inspect physics-side behavior.

`render_uavon_trajectory_video.py --offset` is only for trajectories saved in local coordinates. Use `--offset 0,0,0` for trajectories produced by `scripts/smoke_astar_existing_scene.py`, because the updated smoke script writes global AirSim poses. If replaying an older local-coordinate trajectory, pass the global start position as `--offset`, for example `--offset=-363.7956,-311.0911,-10.0`.

The earlier pure-white top-down video issue was caused by mixing local trajectory coordinates with global dataset start/target coordinates, which made the path collapse into a nearly invisible point on a large white canvas. The current renderer expects global trajectory positions and plots the start, target, and UAV path in the same coordinate frame.




### TODO

- Example of Reinforcement Learning using PPO
- Example of training a model using Imitation Learning

### **Acknowledgment**

- The simulation interaction module of this project is built upon the works of [AirVLN](https://github.com/AirVLN/AirVLN/) and [TravelUAV](https://github.com/prince687028/TravelUAV/). We sincerely thank them for their outstanding contributions.

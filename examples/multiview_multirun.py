# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import json
import glob
from pathlib import Path
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
from cosmos_transfer2.multiview2world import MultiviewInference
from cosmos_transfer2._src.imaginaire.utils import distributed
from cosmos_transfer2.config import get_multiview_params_from_json


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Video2World inference script."""
    parser = argparse.ArgumentParser(description="Multiview Video2World multi-run inference script")
    parser.add_argument("--params_file", type=str, required=True, help="Base config file (e.g., example_autoregressive_config.json)")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--experiment", type=str, required=False, help="Experiment name")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to checkpoint")
    parser.add_argument("--control_videos", type=str, 
                       default="/home/tianshic/lustre/i4_wan/imaginaire4/data/world_scenario/v3_10fps",
                       help="Directory containing control videos organized by camera")
    parser.add_argument("--prompts_folder", type=str,
                       default="/home/tianshic/lustre/i4_wan/imaginaire4/data/prompt_rewrite/v3",
                       help="Directory containing prompt JSON files")
    parser.add_argument("--input_dir", type=str, default=None,
                       help="Directory containing input videos (if None, uses control videos as input)")
    parser.add_argument("--output_base_dir", type=str, default="outputs/multirun",
                       help="Base directory for outputs")

    return parser.parse_args()


def get_camera_mapping():
    """Map camera directory names to config keys."""
    return {
        "ftheta_camera_front_wide_120fov": "front_wide",
        "ftheta_camera_cross_left_120fov": "cross_left", 
        "ftheta_camera_cross_right_120fov": "cross_right",
        "ftheta_camera_rear_left_70fov": "rear_left",
        "ftheta_camera_rear_right_70fov": "rear_right",
        "ftheta_camera_rear_tele_30fov": "rear",
        "ftheta_camera_front_tele_30fov": "front_tele"
    }

def get_camera_mapping_cosmos1():
    """Map camera directory names to config keys."""
    return {
        "ftheta_camera_front_wide_120fov": "0",
        "ftheta_camera_cross_left_120fov": "1", 
        "ftheta_camera_cross_right_120fov": "2",
        "ftheta_camera_rear_left_70fov": "4",
        "ftheta_camera_rear_right_70fov": "5",
        "ftheta_camera_rear_tele_30fov": "3",
        "ftheta_camera_front_tele_30fov": "6"
    }

def find_video_uids(control_videos_dir):
    """Find all unique video UIDs by scanning one camera directory."""
    camera_mapping = get_camera_mapping()
    first_camera_dir = list(camera_mapping.keys())[0]
    camera_path = os.path.join(control_videos_dir, first_camera_dir)
    
    if not os.path.exists(camera_path):
        raise ValueError(f"Camera directory not found: {camera_path}")
    
    video_files = glob.glob(os.path.join(camera_path, "*.mp4"))
    uids = []
    
    for video_file in video_files:
        filename = os.path.basename(video_file)
        uid = filename.replace(".mp4", "")
        uids.append(uid)
    
    return uids


def get_prompts_for_uid(uid, prompts_folder):
    """Get all prompt variants for a specific UID."""
    prompt_file = os.path.join(prompts_folder, f"{uid}.json")
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            prompt_data = json.load(f)
        return prompt_data
    else:
        logging.warning(f"Prompt file not found: {prompt_file}")
        return {}


def create_params_for_uid_and_prompt(base_params, uid, prompt_key, prompt_text, control_videos_dir, input_dir, output_base_dir):
    """Create parameter configuration for a specific UID and prompt variant."""
    camera_mapping = get_camera_mapping()
    
    # Create a copy of base params
    params_dict = base_params.to_kwargs().copy()
    # if distributed.get_rank() == 0:
    #     import ipdb
    #     ipdb.set_trace()
    # distributed.barrier()
    # Set output directory based on UID and prompt key
    output_name = f"{uid}_{prompt_key}"
    params_dict["output_dir"] = os.path.join(output_base_dir, output_name)
    
    # Set prompt text directly
    params_dict["prompt"] = prompt_text
    params_dict["prompt_path"] = ""  # Clear prompt_path since we're setting prompt directly
    
    # Set video paths for each camera
    for camera_dir, config_key in camera_mapping.items():
        control_path = os.path.join(control_videos_dir, camera_dir, f"{uid}.mp4")
        
        if not os.path.exists(control_path):
            raise ValueError(f"Control video not found: {control_path}")
        
        # Set input path
        if input_dir is not None:
            "/home/tianshic/lustre/i4_wan/imaginaire4/outputs/cosmos1_sv2mv_50steps_on_2pt5_15fps/c905d8cc-fe02-4bf9-9593-68fd91b9a08d_663832901621_Bus/0.mp4"
            camera_c1_name = get_camera_mapping_cosmos1()[camera_dir]
            input_path = os.path.join(input_dir, f"{uid}_{prompt_key}", f"{camera_c1_name}.mp4")
            if not os.path.exists(input_path):
                logging.warning(f"Input video not found, using control video: {input_path}")
                input_path = control_path
        else:
            input_path = control_path
        
        # Update the camera configuration
        if config_key not in params_dict:
            params_dict[config_key] = {}
        
        params_dict[config_key]["control_path"] = control_path
        params_dict[config_key]["input_path"] = input_path
    
    return params_dict


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    
    # Load base parameters
    if not os.path.exists(args.params_file):
        raise ValueError(f"Params file {args.params_file} does not exist")
    
    base_params = get_multiview_params_from_json(args.params_file)
    logging.info(f"Loaded base parameters from {args.params_file}")
    
    # Find all video UIDs
    try:
        uids = find_video_uids(args.control_videos)
        logging.info(f"Found {len(uids)} video UIDs: {uids}")
    except Exception as e:
        logging.error(f"Error finding video UIDs: {e}")
        return
    
    # Initialize inference pipeline once
    pipe = MultiviewInference(
        num_gpus=args.num_gpus,
        experiment=args.experiment,
        ckpt_path=args.checkpoint_path,
    )
    logging.info("Initialized MultiviewInference pipeline")
    
    # Create output base directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Count total number of inference runs
    total_runs = 0
    uid_prompt_pairs = []
    
    for uid in uids:
        prompts = get_prompts_for_uid(uid, args.prompts_folder)
        for prompt_key in prompts.keys():
            uid_prompt_pairs.append((uid, prompt_key, prompts[prompt_key]))
            total_runs += 1
    
    logging.info(f"Found {total_runs} total inference runs across {len(uids)} UIDs")
    
    # Process each UID and prompt combination
    for run_idx, (uid, prompt_key, prompt_text) in enumerate(uid_prompt_pairs):
        output_name = f"{uid}_{prompt_key}"
        logging.info(f"Processing run {run_idx+1}/{total_runs}: {output_name}")
        
        # Create parameters for this UID and prompt variant
        params_dict = create_params_for_uid_and_prompt(
            base_params, uid, prompt_key, prompt_text, 
            args.control_videos, args.input_dir, args.output_base_dir
        )
        
        # Convert back to MultiviewParams object
        from cosmos_transfer2.config import MultiviewParams
        params = MultiviewParams.create(params_dict)
        
        # Create output directory for this UID and prompt
        os.makedirs(params.output_dir, exist_ok=True)
        
        logging.info(f"Running inference for {output_name}")
        logging.info(f"Output directory: {params.output_dir}")
        logging.info(f"Prompt key: {prompt_key}")
        logging.info(f"Prompt: {params.prompt[:100]}..." if len(params.prompt) > 100 else f"Prompt: {params.prompt}")
        
        # Run inference
        pipe.infer(params)
        
        logging.info(f"Completed inference for {output_name}")
        
    
    logging.info(f"Completed processing all {total_runs} inference runs")


if __name__ == "__main__":
    main()

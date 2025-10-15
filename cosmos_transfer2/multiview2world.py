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
from typing import Union
import ipdb
import einops
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import numpy as np
from cosmos_transfer2._src.imaginaire.utils import distributed
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video, create_video_grid
from cosmos_transfer2._src.transfer2.inference.utils import get_prompt_from_path, color_message
from cosmos_transfer2._src.transfer2_multiview.inference.inference import ControlVideo2WorldInference
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.conditioner import ConditionLocation, ConditionLocationList
from cosmos_transfer2._src.transfer2_multiview.datasets.local_dataset import (
    LocalMultiviewAugmentorConfig,
    LocalMultiviewDatasetBuilder,
)
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.driving import (
    MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION,
)
from cosmos_transfer2.config import MODEL_CHECKPOINTS, ModelKey, MultiviewParams, VIEW_INDEX_DICT

_DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(variant="drive")]
NUM_DATALOADER_WORKERS = 8


class MultiviewInference:
    def __init__(
        self,
        num_gpus=8,
        experiment="",
        ckpt_path="",
    ):
        if not ckpt_path:
            ckpt_path = _DEFAULT_CHECKPOINT.path
        if not experiment:
            experiment = _DEFAULT_CHECKPOINT.experiment

        log.info(f"Using {experiment=} and {ckpt_path=}")

        # Enable deterministic inference
        os.environ["NVTE_FUSED_ATTN"] = "0"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.enable_grad(False)  # Disable gradient calculations for inference

        self.pipe = ControlVideo2WorldInference(
            experiment_name=experiment,
            ckpt_path=ckpt_path,
            context_parallel_size=num_gpus,
        )

        self.rank0 = True
        if num_gpus > 1:
            self.rank0 = torch.distributed.get_rank() == 0

    def infer(self, params: Union[MultiviewParams, dict]):
        if isinstance(params, dict):
            p = MultiviewParams.create(params)
        else:
            p = params
        log.info(f"params: {p}")
        
        # Choose appropriate dataloader config based on autoregressive mode
        if p.enable_autoregressive:
            config_key = f"{self.pipe.config.model.config.resolution}_autoregressive"
            if config_key not in MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION:
                config_key = "720_autoregressive"  # fallback to 720p autoregressive
            driving_dataloader_config = MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION[config_key]
        else:
            driving_dataloader_config = MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION[
                self.pipe.config.model.config.resolution
            ]
        driving_dataloader_config.n_views = p.n_views
        driving_dataloader_config.num_video_frames_per_view = p.num_video_frames_per_view
        driving_dataloader_config.minimum_start_index = p.minimum_start_index
        driving_dataloader_config.num_video_frames_loaded_per_view = p.num_video_frames_loaded_per_view
        prompt, _ = get_prompt_from_path(p.prompt_path, p.prompt)
        # if self.rank0:
        #     ipdb.set_trace()
        # distributed.barrier()
        self.pipe.config.model.config.condition_locations = ConditionLocationList([ConditionLocation.FIRST_FRAMES_EXCEPT_TEL if p.no_cond_rear_tele else ConditionLocation.FIRST_RANDOM_N])
        # setup the control and input videos dict
        input_video_file_dict = {}
        control_video_file_dict = {}
        for key, value in p.input_and_control_paths.items():
            if "_input" in key:
                input_video_file_dict[key.removesuffix("_input")] = value
            elif "_control" in key:
                control_video_file_dict[key.removesuffix("_control")] = value

        dataset = LocalMultiviewDatasetBuilder(
            input_video_file_dict=input_video_file_dict, control_video_file_dict=control_video_file_dict
        ).build_dataset(
            LocalMultiviewAugmentorConfig(
                resolution=self.pipe.config.model.config.resolution,
                driving_dataloader_config=driving_dataloader_config,
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_DATALOADER_WORKERS,
        )

        if len(dataloader) == 0:
            raise ValueError("No input data found")

        for i, batch in enumerate(dataloader):
            
            batch["ai_caption"] = [prompt]
            batch["control_weight"] = p.control_weight
            if len(p.num_conditional_frames_per_view) > 0:
                index_view_dict = {}
                for k,v in VIEW_INDEX_DICT.items():
                    index_view_dict[v] = k
                num_conditional_frames_per_view = []
                for k,v in index_view_dict.items():
                    if len(num_conditional_frames_per_view) < p.n_views:
                        num_conditional_frames_per_view.append(p.num_conditional_frames_per_view[v])
                print(f"num_conditional_frames_per_view: {num_conditional_frames_per_view}")
                batch["num_conditional_frames"] = num_conditional_frames_per_view
            else:
                batch["num_conditional_frames"] = p.num_conditional_frames
                
            if p.enable_autoregressive:
                log.info(f"------ Generating video with autoregressive mode ------")
                final_video, final_control = self._generate_autoregressive(batch, p)
            else:
                log.info(f"------ Generating video ------")
                video = self.pipe.generate_from_batch(batch, guidance=float(p.guidance), seed=int(p.seed), num_steps=p.num_steps)
                final_video = video[0]
                final_control = batch.get("control_input_hdmap_bbox", None)
                if final_control is not None:
                    final_control = final_control[0]  # Remove batch dimension
                
            if self.rank0:
                self._save_video(final_video, p, final_control)
    

    def _save_video(self, final_video, p, control_video=None):
        final_video = final_video.cpu().clamp(-1, 1) / 2 + 0.5
        video_segments = einops.rearrange(final_video, "c (v t) h w -> v t c h w", v=p.n_views)

        if p.n_views == 6:
            grid = create_video_grid(
                [
                    video_segments[5],
                    video_segments[0],  
                    video_segments[1],
                    video_segments[4],
                    video_segments[3],
                    video_segments[2],
                ],
                n_row=2,
            )

        elif p.n_views == 7:
            grid = create_video_grid(
                [
                    video_segments[5],
                    video_segments[0],
                    video_segments[6],
                    video_segments[1],
                    video_segments[4],
                    video_segments[3],
                    video_segments[2],
                ],
                n_row=2,
            )

        else:
            grid = create_video_grid([video_segments[i] for i in range(p.n_views)], n_row=2)
        log.info(f"Grid shape: {grid.shape}")
        # If control video is provided, create control grid and concatenate vertically
        if control_video is not None:
            control_video = control_video.cpu().clamp(0.0, 1.0)# / 2 + 0.5
            control_segments = einops.rearrange(control_video, "c (v t) h w -> v t c h w", v=p.n_views)
            
            if p.n_views == 6:
                control_grid = create_video_grid(
                    [
                        control_segments[5],
                        control_segments[0],  
                        control_segments[1],
                        control_segments[4],
                        control_segments[3],
                        control_segments[2],
                    ],
                    n_row=2,
                )

            elif p.n_views == 7:
                control_grid = create_video_grid(
                    [
                        control_segments[5],
                        control_segments[0],
                        control_segments[6],
                        control_segments[1],
                        control_segments[4],
                        control_segments[3],
                        control_segments[2],
                    ],
                    n_row=2,
                )

            else:
                control_grid = create_video_grid([control_segments[i] for i in range(p.n_views)], n_row=2)
            
            # Concatenate control grid (top) and generated grid (bottom) vertically
            combined_grid = torch.cat([control_grid, grid], dim=2)  # Concatenate along height dimension
            combined_grid = einops.rearrange(combined_grid, "t c h w -> c t h w")
            log.info(f"Combined grid shape: {combined_grid.shape}")
            save_img_or_video(combined_grid, f"{p.output_dir}/output_with_control_grid", fps=int(p.fps))
        
        grid = einops.rearrange(grid, "t c h w -> c t h w")
        save_img_or_video(grid, f"{p.output_dir}/output_grid", fps=int(p.fps))
        save_img_or_video(final_video, f"{p.output_dir}/output", fps=int(p.fps))
        log.info(color_message(f"Generated video saved to {p.output_dir}/output.mp4\n", "green"))

    def _generate_autoregressive(self, full_batch, params):
        """
        Generate video using autoregressive sliding window approach.
        
        Args:
            full_batch: Full batch containing all video frames loaded by the dataloader
            params: MultiviewParams containing generation parameters
            
        Returns:
            Generated video tensor
        """
        # Extract full video and control tensors
        
        full_control = full_batch["control_input_hdmap_bbox"]  # Shape: [1, 3, total_frames, H, W]
        # if self.rank0:
        #     ipdb.set_trace()
        # distributed.barrier()
        batch_size, channels, total_frames, height, width = full_batch["video"].shape # Shape: [1, 3, total_frames, H, W]
        n_views = params.n_views
        chunk_size = 29  # frames per view per chunk (model's native capacity)
        overlap = params.chunk_overlap
        
        # Calculate frames per view from the loaded video
        frames_per_view = total_frames // n_views
        log.info(f"Total frames loaded: {total_frames}, Frames per view: {frames_per_view}, Views: {n_views}")
        
        # Initialize output video list
        generated_chunks = []
        
        # Calculate number of chunks needed
        effective_chunk_size = chunk_size - overlap
        num_chunks = max(1, (frames_per_view - overlap + effective_chunk_size - 1) // effective_chunk_size)
        
        log.info(f"Generating {num_chunks} chunks with overlap {overlap}")
        
        # Generate first chunk using original input videos
        current_input_video = full_batch["video"].clone()
        
        for chunk_idx in range(num_chunks):
            # Calculate frame range for this chunk
            start_frame = chunk_idx * effective_chunk_size
            end_frame = min(start_frame + chunk_size, frames_per_view)
            
            if start_frame >= frames_per_view:
                break
                
            log.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}, frames {start_frame}-{end_frame}")
            
            # Create chunk batch (extract 29-frame window from full video)
            chunk_batch = self._create_chunk_batch(
                full_batch, current_input_video, full_control, 
                start_frame, end_frame, n_views
            )
            if chunk_idx == 0:
                chunk_batch["num_conditional_frames"] = full_batch["num_conditional_frames"]
            else:
                chunk_batch["num_conditional_frames"] = params.chunk_overlap

            # Generate chunk (model processes 29 frames per view -> outputs 29 frames per view)
            chunk_video = self.pipe.generate_from_batch(
                chunk_batch, guidance=float(params.guidance), seed=int(params.seed) + chunk_idx, num_steps=params.num_steps,
            )[0]    # C_T_H_W
            chunk_video = einops.rearrange(chunk_video, "C (V T) H W -> V C T H W", V=n_views)
            # Store generated chunk (remove overlap from previous chunks)
            if chunk_idx == 0:
                generated_chunks.append(chunk_video)
            else:
                # Remove overlap frames from the beginning of this chunk
                generated_chunks.append(chunk_video[:, :, overlap:])
            
            # Update input video for next iteration using generated frames
            if chunk_idx < num_chunks - 1:  # Not the last chunk
                current_input_video = self._update_input_video_with_generated(
                    current_input_video, chunk_video, start_frame, end_frame, n_views, params.num_conditional_frames if chunk_idx == 0 else params.chunk_overlap
                )
        
        # Concatenate all chunks along time dimension
        final_video = torch.cat(generated_chunks, dim=2)
        # Return the corresponding control video for the same time range
        final_control = einops.rearrange(full_control[0].float() / 255.0, "C (V T) H W -> V C T H W", V=n_views)[:, :, :final_video.shape[2]]

        final_video = einops.rearrange(final_video, "V C T H W -> C (V T) H W")
        final_control = einops.rearrange(final_control, "V C T H W -> C (V T) H W")
        log.info(f"Final video shape: {final_video.shape}")
        log.info(f"Final control shape: {final_control.shape}")
        return final_video, final_control
    
    def _create_chunk_batch(self, original_batch, input_video, control_video, start_frame, end_frame, n_views):
        """
        Create a batch for a specific chunk by extracting a 29-frame window from each view.
        
        Args:
            original_batch: Original batch with metadata
            input_video: Full input video tensor [1, 3, total_frames, H, W]
            control_video: Full control video tensor [1, 3, total_frames, H, W] 
            start_frame: Start frame index within each view
            end_frame: End frame index within each view
            n_views: Number of views (7)
            
        Returns:
            Batch dictionary with 29*7=203 frame tensors
        """
        chunk_batch = {}
        
        # Copy non-video fields from original batch
        for key, value in original_batch.items():
            if key not in ["video", "control_input_hdmap_bbox"]:
                chunk_batch[key] = value
        
        # Calculate frames per view in the full video
        input_video_NVCTHW = einops.rearrange(input_video, "N C (V T) H W -> N V C T H W", V=n_views)
        input_video_chunk = input_video_NVCTHW[:, :, :, start_frame:end_frame, :, :]
        control_video_NVCTHW = einops.rearrange(control_video, "N C (V T) H W -> N V C T H W", V=n_views)
        control_video_chunk = control_video_NVCTHW[:, :, :, start_frame:end_frame, :, :]
        view_indices = einops.rearrange(original_batch["view_indices"], "N (V T) -> N V T", V=n_views)
        view_indices_chunk = view_indices[:, :, start_frame:end_frame]
        
        input_video_chunk = einops.rearrange(input_video_chunk, "N V C T H W -> N C (V T) H W")
        control_video_chunk = einops.rearrange(control_video_chunk, "N V C T H W -> N C (V T) H W") 
        view_indices_chunk = einops.rearrange(view_indices_chunk, "N V T -> N (V T)")
        chunk_batch["video"] = input_video_chunk.clone()
        chunk_batch["control_input_hdmap_bbox"] = control_video_chunk.clone()
        chunk_batch["num_video_frames_per_view"] = torch.tensor([end_frame - start_frame,]).to(original_batch["num_video_frames_per_view"])
        chunk_batch["view_indices"] = view_indices_chunk.clone()
        chunk_batch["fps"] = torch.tensor([30.0]).to(original_batch["fps"])
        return chunk_batch
    
    def _update_input_video_with_generated(self, current_input, generated_chunk, start_frame, end_frame, n_views, overlap):
        """
        Update input video with generated frames for next iteration.
        
        Args:
            current_input: Full input video [1, 3, total_frames, H, W]
            generated_chunk: Generated chunk [ 7, 3, 29, H, W] 
            start_frame: Start frame index within each view
            end_frame: End frame index within each view
            n_views: Number of views
            overlap: Number of overlapping frames
            
        Returns:
            Updated input video with generated frames replacing future frames
        """
        chunk_frames_per_view = generated_chunk.shape[2]  # Should be 29

        update_start = start_frame + overlap  # Skip overlap frames
        update_end = end_frame
        gen_view_start = overlap
        gen_view_end = chunk_frames_per_view
        actual_update_end = min(update_end, current_input.shape[2])
        actual_gen_end = gen_view_end
        
        frames_to_copy = min(
            actual_update_end - update_start,
            actual_gen_end - gen_view_start
        )
        log.info(f"Frames to copy: {frames_to_copy}")
        log.info(f"Update start: {update_start}, Update end: {update_end}")
        log.info(f"Gen view start: {gen_view_start}, Gen view end: {gen_view_end}")
        current_input_NVCTHW = einops.rearrange(current_input.clone(), "N C (V T) H W -> N V C T H W", V=n_views)
        generated_chunk_NVCTHW = generated_chunk.unsqueeze(0)
        generated_chunk_NVCTHW = ((generated_chunk_NVCTHW / 2.0 + 0.5).clamp(-1.0, 1.0) * 255.0).to(current_input.dtype)
        
        current_input_NVCTHW[:, :, :, update_start:update_start + frames_to_copy] = \
                        generated_chunk_NVCTHW[:, :, :, gen_view_start:gen_view_start + frames_to_copy]
        updated_input = einops.rearrange(current_input_NVCTHW, "N V C T H W -> N C (V T) H W", V=n_views)
        
        return updated_input

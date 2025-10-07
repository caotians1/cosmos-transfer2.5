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

# Configs for resuming from stage3 training

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict
from cosmos_transfer2._src.predict2.datasets.cached_replay_dataloader import get_cached_replay_dataloader
from cosmos_transfer2._src.predict2.datasets.dataset_provider import get_image_dataset, get_video_dataset
from cosmos_transfer2._src.predict2.datasets.joint_dataloader import IterativeJointDataLoader
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy

_TRAINER_DEBUG_CONFIG = dict(
    max_iter=1000,
    logging_iter=50,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=1000000000000,
        ),
        every_n_sample_ema=dict(
            every_n=1000000000000,
        ),
    ),
)
_CKPT_DEBUG_CONFIG = dict(
    save_iter=10,
    load_path="",
    load_training_state=False,
    strict_resume=False,
)


def build_debug_runs(job):
    wo_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_WO_RESUME" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
    )

    mock_wo_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_MOCK_WO_RESUME" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
    )

    return [wo_resume, mock_wo_resume]


T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_IMPROVED_PRETRAINING_DATA_RECTIFIED_FLOW_GCP = LazyDict(
    dict(
        defaults=[
            {"override /data_train": None},
            {"override /model": "fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_14B"},
            {"override /conditioner": "video_prediction_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            {"override /checkpoint": "gcp"},
            {"override /tokenizer": "wan2pt1_tokenizer_gcp"},
            "_self_",
        ],
        job=dict(
            group="official_runs_text2world",
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_improved_pretraining_data_rectified_flow_gcp",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.2,
            betas=[0.9, 0.999],
        ),
        scheduler=dict(
            f_max=[0.3],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                fsdp_shard_size=32,
                resolution="720",
                state_t=24,
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="reweighting",
                train_time_distribution="logitnormal",
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    timestep_scale=0.001,
                    sac_config=dict(
                        mode="predict2_14b_720_aggressive",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    use_wan_fp32_strategy=True,
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.1,
                        use_empty_string=False,
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    s3_credential_path="credentials/gcp_checkpoint.secret",
                ),
            )
        ),
        checkpoint=dict(
            save_iter=250,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_improved_pretraining_data_gcp/checkpoints/iter_000055750",
            load_training_state=False,
            strict_resume=True,
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=False,
                max_diff=1.5,
            ),
            callbacks=dict(
                grad_clip=dict(
                    clip_norm=0.1,
                ),
                manual_gc=dict(
                    every_n=200,
                ),
                every_n_sample_reg=dict(
                    every_n=1000000000000,
                ),
                every_n_sample_ema=dict(
                    every_n=1000000000000,
                ),
            ),
        ),
        dataloader_train=L(IterativeJointDataLoader)(
            dataloaders={
                "image_data": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_pretrain_and_synthetic_photoreal_20250805_image_whole",
                            object_store="gcp",
                            resolution="${model.config.resolution}",
                            is_train=True,
                            caption_type="qwen2p5_7b_v4",
                            dataset_resolution_type="gt720p",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=3,
                        num_workers=8,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                    ),
                    ratio=1,
                ),
                "image_data_prompt_captions": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_image_dataset)(
                            dataset_name="cosmos_synthetic_filtered_combined_20250805_image_whole",
                            object_store="gcp",
                            resolution="${model.config.resolution}",
                            is_train=True,
                            caption_type="prompts",
                            dataset_resolution_type="gt720p",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                        batch_size=3,
                        num_workers=8,
                        prefetch_factor=4,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="image_dataloader",
                    ),
                    ratio=1,
                ),
                "video_data": dict(
                    dataloader=L(get_cached_replay_dataloader)(
                        dataset=L(get_video_dataset)(
                            dataset_name="cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_video_whole",
                            object_store="gcp",
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="noframedrop_nocameramove_video_augmentor_v1",
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            dataset_resolution_type="gt720p",
                            use_native_fps=True,
                            embedding_type=None,
                            is_train=True,
                            chunk_size=256,
                        ),
                        batch_size=1,
                        num_workers=4,
                        prefetch_factor=2,
                        sampler=None,
                        persistent_workers=False,
                        pin_memory=True,
                        cache_replay_name="video_dataloader",
                        use_cache=False,
                    ),
                    ratio=2,
                ),
            }
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)


# Rectified flow run with original training data with >= 720p resolution
T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5 = LazyDict(
    dict(
        defaults=[
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_s3"
            },
            {"override /model": "fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_14B"},
            {"override /conditioner": "video_prediction_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="official_runs_text2world",
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.001,
            betas=[0.9, 0.999],
        ),
        scheduler=dict(
            f_max=[0.3],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[200_000],
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                fsdp_shard_size=32,
                resolution="720",
                state_t=24,
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="reweighting",
                train_time_distribution="logitnormal",
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    timestep_scale=0.001,
                    sac_config=dict(
                        mode="predict2_14b_720_aggressive",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    use_wan_fp32_strategy=True,
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                        use_empty_string=False,
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    s3_credential_path="credentials/s3_checkpoint.secret",
                ),
            )
        ),
        checkpoint=dict(
            save_iter=250,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40_qwen_concat_v2/checkpoints/iter_000031000",
            load_training_state=False,
            strict_resume=True,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=100,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
            callbacks=dict(
                grad_clip=dict(
                    clip_norm=0.1,
                ),
                manual_gc=dict(
                    every_n=200,
                ),
                every_n_sample_reg=dict(
                    every_n=5000,
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
                ),
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=3,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            dataset_resolution_type="gt720p",
                            caption_type="qwen2p5_7b_v4",
                            embedding_type=None,
                            augmentor_name="image_basic_augmentor_without_embeddings",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        dataset=dict(
                            resolution="${model.config.resolution}",
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            embedding_type=None,
                            max_fps_thres=60,
                            min_fps_thres=10,
                            caption_type="t2w_qwen2p5_7b",
                            num_video_frames=93,
                            dataset_resolution_type="gt720p",
                            use_native_fps=True,
                        ),
                    ),
                    ratio=3,
                ),
            ),
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

# Config for shift 2
T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT2 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5[
                "job"
            ]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift2",
        ),
        model=dict(
            config=dict(
                shift=2,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# Config for shift 7
T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT7 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5[
                "job"
            ]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift7",
        ),
        model=dict(
            config=dict(
                shift=7,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# Config for shift 5 with high sigma strategy
T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5_HIGH_SIGMA = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5[
                "job"
            ]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma",
        ),
        model=dict(
            config=dict(
                use_high_sigma_strategy=True,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

# 4K cooldown config - S3
T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_4K_DATA_S3 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_4K_20250812_s3"
            },
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5[
                "job"
            ]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_4K_data_s3",
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40_qwen_concat_v2/checkpoints/iter_000031000",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)


# 4K cooldown config - GCP
T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_4K_DATA_GCP = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_posttraining_hq_v7_4K_20250812_gcp"
            },
            {"override /tokenizer": "wan2pt1_tokenizer_gcp"},
            {"override /checkpoint": "s3"},
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5[
                "job"
            ]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_4K_data_gcp",
        ),
        model=dict(
            config=dict(
                net=dict(
                    sac_config=dict(
                        mode="predict2_14b_720_aggressive",
                    ),
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    s3_credential_path="credentials/gcp_checkpoint.secret",
                ),
            ),
        ),
        trainer=dict(
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        batch_size=3,
                        dataset=dict(
                            object_store="gcp",
                        ),
                    ),
                    ratio=1,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        dataset=dict(
                            object_store="gcp",
                        ),
                    ),
                    ratio=3,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=250,
            load_path="cosmos_diffusion_v2/official_runs_text2world/Stage-c_pt_4-reason_embeddings-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40_qwen_concat_v2/checkpoints/iter_000031000",
            load_training_state=False,
            strict_resume=True,
        ),
    ),
    flags={"allow_objects": True},
)

# Config for shift 7 - GCP
T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT7_GCP = LazyDict(
    dict(
        defaults=[
            f"/experiment/{T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_4K_DATA_GCP['job']['name']}",
            {
                "override /data_train": "image_cosmos_pretrain_and_synthetic_20250520_video_cosmos_pretrainvideo_20250806_dedup_accumulated_and_high_quality_v3_202505_gcp"
            },
            "_self_",
        ],
        job=dict(
            group=T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_4K_DATA_GCP[
                "job"
            ]["group"],
            name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift7_gcp",
        ),
        model=dict(
            config=dict(
                shift=7,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

for _item, _item_wo_resume, _item_mock_wo_resume in [
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_IMPROVED_PRETRAINING_DATA_RECTIFIED_FLOW_GCP,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_IMPROVED_PRETRAINING_DATA_RECTIFIED_FLOW_GCP
        ),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5
        ),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT2,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT2
        ),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT7,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT7
        ),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_4K_DATA_GCP,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_4K_DATA_GCP
        ),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_4K_DATA_S3,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_4K_DATA_S3
        ),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT7_GCP,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT7_GCP
        ),
    ],
    [
        T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5_HIGH_SIGMA,
        *build_debug_runs(
            T2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16_RESUME_FROM_REASON1P1_RECTIFIED_FLOW_SHIFT5_HIGH_SIGMA
        ),
    ],
]:
    cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}", node=_item)
    if _item_wo_resume is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_wo_resume",
            node=_item_wo_resume,
        )
    if _item_mock_wo_resume is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_mock_wo_resume",
            node=_item_mock_wo_resume,
        )

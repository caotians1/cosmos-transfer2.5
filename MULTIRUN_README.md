# Multiview Multi-Run Inference

This document describes the `multiview_multirun.py` script for running batch inference on multiple video sequences.

## Overview

The `multiview_multirun.py` script extends the single-inference capability to process multiple video sequences automatically. It reads control videos and prompts from organized directory structures and generates outputs for each unique video ID (UID) and each prompt variant, creating comprehensive results across all combinations.

## Directory Structure

### Control Videos Directory
```
/path/to/control_videos/
├── ftheta_camera_front_wide_120fov/
│   ├── uid1.mp4
│   ├── uid2.mp4
│   └── ...
├── ftheta_camera_cross_left_120fov/
│   ├── uid1.mp4
│   ├── uid2.mp4
│   └── ...
├── ftheta_camera_cross_right_120fov/
├── ftheta_camera_rear_left_70fov/
├── ftheta_camera_rear_right_70fov/
├── ftheta_camera_rear_tele_30fov/
└── ftheta_camera_front_tele_30fov/
```

### Prompts Directory
```
/path/to/prompts/
├── uid1.json
├── uid2.json
└── ...
```

Each prompt JSON file contains multiple prompt variations:
```json
{
    "Original": "Original prompt text...",
    "Morning": "Morning variant...",
    "Night": "Night variant...",
    "Rainy": "Rainy variant...",
    "Snowy": "Snowy variant...",
    "Foggy": "Foggy variant...",
    "Construction Vehicle": "Construction variant...",
    "Bus": "Bus variant...",
    ...
}
```

**All prompt keys** in each JSON file will be processed, generating separate outputs for each variant.

### Input Videos Directory (Optional)
If provided, follows the same structure as control videos:
```
/path/to/input_videos/
├── ftheta_camera_front_wide_120fov/
├── ftheta_camera_cross_left_120fov/
└── ...
```

## Usage

### Basic Usage
```bash
python examples/multiview_multirun.py \
    --params_file example_autoregressive_config.json \
    --num_gpus 8
```

### Full Configuration
```bash
python examples/multiview_multirun.py \
    --params_file example_autoregressive_config.json \
    --num_gpus 8 \
    --control_videos /path/to/control_videos \
    --prompts_folder /path/to/prompts \
    --input_dir /path/to/input_videos \
    --output_base_dir outputs/batch_results \
    --experiment my_experiment \
    --checkpoint_path /path/to/checkpoint.pth
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--params_file` | Required | Base configuration file (e.g., `example_autoregressive_config.json`) |
| `--num_gpus` | 8 | Number of GPUs to use for inference |
| `--control_videos` | `/home/tianshic/lustre/i4_wan/imaginaire4/data/world_scenario/v3_30fps_concat` | Directory containing control videos |
| `--prompts_folder` | `/home/tianshic/lustre/i4_wan/imaginaire4/data/prompt_rewrite/v3` | Directory containing prompt JSON files |
| `--input_dir` | None | Directory containing input videos (if None, uses control videos) |
| `--output_base_dir` | `outputs/multirun` | Base directory for all outputs |
| `--experiment` | None | Experiment name override |
| `--checkpoint_path` | None | Checkpoint path override |

## Camera Mapping

The script automatically maps camera directory names to configuration keys:

| Directory Name | Config Key |
|----------------|------------|
| `ftheta_camera_front_wide_120fov` | `front_wide` |
| `ftheta_camera_cross_left_120fov` | `cross_left` |
| `ftheta_camera_cross_right_120fov` | `cross_right` |
| `ftheta_camera_rear_left_70fov` | `rear_left` |
| `ftheta_camera_rear_right_70fov` | `rear_right` |
| `ftheta_camera_rear_tele_30fov` | `rear` |
| `ftheta_camera_front_tele_30fov` | `front_tele` |

## Output Structure

For each UID and prompt combination, the script creates a separate output directory:
```
outputs/multirun/
├── uid1_Original/
│   ├── output.mp4
│   ├── output_grid.mp4
│   └── output_with_control_grid.mp4
├── uid1_Morning/
│   ├── output.mp4
│   ├── output_grid.mp4
│   └── output_with_control_grid.mp4
├── uid1_Night/
│   ├── output.mp4
│   ├── output_grid.mp4
│   └── output_with_control_grid.mp4
├── uid2_Original/
├── uid2_Morning/
├── uid2_Night/
└── ...
```

## Features

- **Automatic UID Discovery**: Scans camera directories to find all available video UIDs
- **Multi-Prompt Processing**: Automatically processes all prompt variants from JSON files
- **Comprehensive Coverage**: Generates results for every UID × prompt combination
- **Flexible Input Sources**: Can use separate input videos or control videos as input
- **Error Handling**: Continues processing other combinations if one fails
- **Progress Logging**: Detailed logging of processing status with run counts
- **Resource Efficiency**: Initializes the inference pipeline once and reuses it

## Example Workflow

1. **Prepare Data**: Organize control videos, input videos (optional), and prompts in the expected directory structure
2. **Create Base Config**: Prepare a base configuration file (e.g., `example_autoregressive_config.json`)
3. **Run Multi-Inference**: Execute the script with appropriate arguments
4. **Review Results**: Check the output directories for generated videos

## Error Handling

The script includes robust error handling:
- Missing video files are reported with warnings
- Failed inference for one UID doesn't stop processing of others
- Detailed error messages help identify issues
- Graceful fallbacks (e.g., using control videos when input videos are missing)

## Performance Considerations

- **Memory Usage**: Each UID is processed sequentially to manage memory usage
- **GPU Utilization**: The inference pipeline is initialized once and reused
- **Disk Space**: Ensure sufficient disk space for all output videos
- **Processing Time**: Total time scales linearly with the number of UIDs

## Troubleshooting

### Common Issues

1. **Missing Video Files**: Ensure all camera directories contain videos for each UID
2. **Prompt File Errors**: Verify JSON format and file naming consistency
3. **Memory Issues**: Reduce `num_gpus` or process fewer UIDs at once
4. **Permission Errors**: Check read/write permissions for input and output directories

### Debug Mode

For detailed debugging, modify the logging level in the script:
```python
logging.basicConfig(level=logging.DEBUG)
```

This provides extensive information about file discovery, parameter creation, and inference progress.

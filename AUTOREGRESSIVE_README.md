# Autoregressive Multiview Generation

This document describes the new autoregressive generation functionality added to the MultiviewInference class.

## Overview

The autoregressive generation mode enables the model to generate longer videos by processing them in overlapping chunks and using the output of each chunk as input for the next chunk. This allows for generating videos longer than the model's native 29-frame capacity.

## Key Features

- **Sliding Window Generation**: Processes videos in 29-frame chunks with configurable overlap
- **Extended Video Loading**: Loads longer input and control videos for autoregressive processing
- **Autoregressive Feedback**: Uses generated frames as input for subsequent chunks
- **Configurable Overlap**: Supports 1-5 frame overlap between chunks for smooth transitions
- **Control Video Visualization**: Saves combined grid showing both control inputs and generated outputs

## Configuration

### New Parameters

Add these parameters to your JSON configuration file:

```json
{
    "enable_autoregressive": true,
    "chunk_overlap": 1
}
```

- `enable_autoregressive`: Boolean flag to enable autoregressive mode (default: false)
- `chunk_overlap`: Number of overlapping frames between chunks (default: 1, range: 1-28)

### Example Configuration

```json
{
    "output_dir": "outputs/autoregressive_test",
    "prompt": "A car driving through a scenic mountain road during sunset",
    "guidance": 3,
    "seed": 2025,
    "enable_autoregressive": true,
    "chunk_overlap": 1,
    "front_wide": {
        "input_path": "path/to/input_video.mp4",
        "control_path": "path/to/control_video.mp4"
    }
    // ... other view configurations
}
```

## How It Works

1. **Extended Video Loading**: The system loads longer video sequences using the `720_autoregressive` configuration
2. **Chunk Processing**: Videos are processed in 29-frame chunks with specified overlap
3. **Sliding Window**: Each chunk starts `(chunk_size - overlap)` frames after the previous chunk
4. **Frame Replacement**: Generated frames replace input frames for subsequent chunks
5. **Final Assembly**: All chunks are concatenated to form the final long video

## Technical Details

### New Configuration

A new dataloader configuration `MADS_DRIVING_DATALOADER_CONFIG_res720_AUTOREGRESSIVE` has been added:
- `num_video_frames_loaded_per_view`: 561 (loads ~561 frames per view)
- `num_video_frames_per_view`: 561 (outputs all loaded frames for chunking)

**Important Constraint**: The video parser requires `(num_video_frames_loaded_per_view - 1) % (num_video_frames_per_view - 1) == 0`. 
For 29-frame chunks: `(loaded - 1) % 28 == 0`, so `loaded = 28*k + 1`. We use `561 = 28*20 + 1` to support ~20 chunks of generation.

### Implementation

The implementation consists of three main methods:

1. `_generate_autoregressive()`: Main orchestration method
2. `_create_chunk_batch()`: Creates batch data for each chunk
3. `_update_input_video_with_generated()`: Updates input videos with generated frames

## Usage

### Command Line

```bash
python examples/multiview.py --params_file config_autoregressive.json --num_gpus 1
```

### Python API

```python
from cosmos_transfer2.multiview2world import MultiviewInference
from cosmos_transfer2.config import get_multiview_params_from_json

# Load configuration
params = get_multiview_params_from_json("config_autoregressive.json")

# Initialize inference
pipe = MultiviewInference(num_gpus=1)

# Run autoregressive generation
pipe.infer(params)
```

## Output Files

The system generates several output files:

- `output.mp4`: Raw generated video (all views concatenated)
- `output_grid.mp4`: Generated video arranged in a grid layout
- `output_with_control_grid.mp4`: Combined visualization with control inputs (top) and generated outputs (bottom)

## Performance Notes

- Generation time scales linearly with the number of chunks
- Use with either 1 or 5 overlap (corresponds to 1 or 2 latent).

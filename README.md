# grounding-sam-rerun
<video controls autoplay src="https://user-images.githubusercontent.com/25287427/233673259-8a0743e7-8e80-4929-91b9-8dfbd9dbbba9.mp4
" controls="controls" style="max-width: 730px;"></video>

This repository uses [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to generate bounding boxes using natural language which are then fed into MetaAI's [Segment Anything Model](https://github.com/facebookresearch/segment-anything) and visualized using [rerun](https://www.rerun.io/)

This can work on either an image, a set of images, or a video.
## Install
first install the main repos requirements using
```
pip install -r requirements.txt
```

then install GroundingDINO by running

```
git submodule update --init --recursive
cd GroundingDINO
pip install -e .
```

## Running Demo
Make sure to be in the main directory
```
python main.py
```

Use `--help` to understand argparse inputs
To use video input ([here is an example video to download](https://www.pexels.com/video/nature-woman-dog-path-4143804/)) use the following command

```
python main.py --video-path <PATH TO YOUR VIDEO FILE> --prompt "<YOUR CHOSEN PROMPT>"
```


# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json
from flask import Flask, request, jsonify

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

from detectron2.projects import point_rend

# constants
WINDOW_NAME = "COCO detections"

app = Flask(__name__)

def setup_cfg(config_file, opts, confidence_threshold):
    # load config from file and command-line arguments
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg

@app.route('/process_images', methods=['POST'])
def process_images():
    data = request.json
    config_file = data.get('config_file')
    input_images = data.get('input_images')
    confidence_threshold = data.get('confidence_threshold', 0.5)
    opts = data.get('opts', [])
    
    if not config_file or not input_images:
        return jsonify({"error": "config_file and input_images are required"}), 400
    
    cfg = setup_cfg(config_file, opts, confidence_threshold)
    demo = VisualizationDemo(cfg)
    
    results = {}
    
    if len(input_images) == 1:
        input_images = glob.glob(os.path.expanduser(input_images[0]))
        assert input_images, "The input path(s) was not found"
    if not os.path.exists("jsonFiles"):
        os.makedirs("jsonFiles")
    for path in tqdm.tqdm(input_images):
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        
        image_results = {}
        
        if "panoptic_seg" in predictions: 
            panoptic_masks = predictions["panoptic_seg"][0].tolist()
            panoptic_classes = [seg_info["category_id"] for seg_info in predictions["panoptic_seg"][1]]
            image_results["panoptic_masks"] = panoptic_masks
            image_results["panoptic_classes"] = panoptic_classes
        else:
            contours = [cv2.findContours(pred_mask.numpy().astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0][0].tolist() for pred_mask in predictions['instances'].pred_masks]
            classes = predictions['instances'].pred_classes.tolist()
            image_results["contours"] = contours
            image_results["classes"] = classes
        
        results[path] = image_results
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
    return jsonify(results)

@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    config_file = data.get('config_file')
    video_input = data.get('video_input')
    output = data.get('output')
    confidence_threshold = data.get('confidence_threshold', 0.5)
    opts = data.get('opts', [])
    
    if not config_file or not video_input:
        return jsonify({"error": "config_file and video_input are required"}), 400
    
    cfg = setup_cfg(config_file, opts, confidence_threshold)
    demo = VisualizationDemo(cfg)
    
    video = cv2.VideoCapture(video_input)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(video_input)
    codec, file_ext = (
        ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    )
    if codec == ".mp4v":
        warnings.warn("x264 codec not available, switching to mp4v")
    if output:
        if os.path.isdir(output):
            output_fname = os.path.join(output, basename)
            output_fname = os.path.splitext(output_fname)[0] + file_ext
        else:
            output_fname = output
        assert not os.path.isfile(output_fname), output_fname
        output_file = cv2.VideoWriter(
            filename=output_fname,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )
    assert os.path.isfile(video_input)
    for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
        if output:
            output_file.write(vis_frame)
    video.release()
    if output:
        output_file.release()
    
    return jsonify({"status": "processing completed", "output_file": output_fname if output else None})

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    app.run(host='0.0.0.0', port=5000)

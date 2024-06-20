import shutup; shutup.please()

import warnings
warnings.filterwarnings("ignore")

from inference import InferencePipeline
# import VideoFrame for type hinting
from inference.core.interfaces.camera.entities import VideoFrame
# import opencv to display our annotated images
import cv2

ROBOFLOW_API_KEY = "paste api key here"
ROBOFLOW_MODEL = "servia-learning-platform/2" # eg xx-xxxx--#
ROBOFLOW_SIZE = 416

# define sink function
def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # print the frame ID of the video_frame object
    print(f"Predictions: {predictions['predictions'][0]}")
    
    cv2.imshow("Predictions", video_frame.image.copy())
    cv2.waitKey(1)

# initialize a pipeline object
pipeline = InferencePipeline.init(
    model_id = ROBOFLOW_MODEL, # Roboflow model to use
    api_key = ROBOFLOW_API_KEY,
    video_reference="/dev/video2", # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction=my_custom_sink, # Function to run after each prediction
)
pipeline.start()
pipeline.join()

WIUT HACKATHON 2026 - COMPUTER VISION TRACK
===============================================

Team: TriVision

Project: Road Traffic Event Detection and Accident Anticipation


1. PROJECT OVERVIEW
-------------------

This project is a computer vision system for a fixed road CCTV camera.

The system has two parts:

Part A - Traffic Event Detection
Given an .mp4 video, the system detects traffic events and returns each event as:

[start_sec, end_sec, label]

Part B - Accident Anticipation (Optional/Bonus)
The system processes video frames in chronological order and returns a risk
score between 0 and 1 indicating how likely an accident is to start within
the next 5 seconds.


2. EVENT CLASSES
----------------

The system uses the official event class IDs:

- accident
- near_miss
- red_light
- wrong_way
- illegal_u_turn
- stopped_vehicle
- jaywalking
- failure_to_yield
- illegal_turn
- solid_line_crossing
- stop_line
- congestion
- road_obstacle
- fire_smoke


3. PROJECT STRUCTURE
--------------------

your-repo/
|
|-- solution.py
|-- run_submission.py
|-- evaluate.py
|-- requirements.txt
|
|-- weights/
|   |-- [MODEL WEIGHTS]
|
|-- src/
|   |-- [SOURCE CODE]
|
|-- notebooks/
|   |-- [OPTIONAL EXPERIMENTS]
|
|-- predictions_samples.json
|-- README.txt
|
|-- [download.sh - OPTIONAL]


4. REQUIREMENTS
---------------

Python 3.10 or newer.

Install the required dependencies with:

    pip install -r requirements.txt

If a Dockerfile is used instead, build the container with:

    docker build -t [TriVision] .


5. MODEL WEIGHTS
----------------

Model weights used by the project:

[WRITE THE MODEL NAME(S) HERE]

Weights are stored in:

    weights/

Total shipped model weights must remain within the hackathon limit of 5 GB.

If weights are downloaded before evaluation, describe the procedure here:

    [DESCRIBE HOW weights/download.sh IS USED, IF APPLICABLE]


6. APPROACH
-----------

The system processes the fixed-camera road video and detects traffic events.

The main pipeline is:

    Input .mp4
        |
        v
    Frame sampling / preprocessing
        |
        v
    Object detection
        |
        v
    Object tracking
        |
        v
    Traffic behaviour analysis
        |
        v
    Event classification / rules
        |
        v
    Temporal event segmentation
        |
        v
    predictions.json


Learned components:
- [DESCRIBE THE ML / DEEP LEARNING MODELS USED]
- [DESCRIBE WHAT THEY ARE TRAINED OR USED FOR]


Rule-based components:
- [DESCRIBE THE RULES USED FOR TRAFFIC EVENTS]
- [DESCRIBE ANY THRESHOLDS OR TEMPORAL POST-PROCESSING]


7. PART A - EVENT DETECTION
---------------------------

The required interface is:

    def detect_events(video_path: str) -> list[list]:

The output format is:

    [
        [start_sec, end_sec, label],
        ...
    ]

Example:

    [
        [12.4, 18.9, "accident"],
        [40.0, 43.5, "red_light"]
    ]

Event times are measured in seconds from the first frame of the video.

Each event is represented as one contiguous segment with one class.

Different classes may overlap, but segments of the same class must not overlap.


8. PART B - ACCIDENT ANTICIPATION
---------------------------------

The optional RiskEstimator interface is:

    class RiskEstimator:
        def reset(self, meta):
            ...

        def step(self, frame, t_sec):
            ...

The risk score is a float in the range [0, 1].

The score represents the probability that an accident will start within
the next 5 seconds.

The estimator uses only frames that have already been received and does not
read future frames or open the video file itself.

Risk signals used by our system:

- [TIME-TO-COLLISION]
- [SUDDEN BRAKING]
- [WRONG-WAY TRAJECTORIES]
- [RED-LIGHT TRAJECTORIES]
- [PEDESTRIANS ENTERING THE ROADWAY]
- [OTHER SIGNALS]


9. DATASETS
-----------

Public datasets used for training:

1. https://universe.roboflow.com/laasya-wdzur/road-obstacle-detection-6zwj5
License:
CC BY 4.0
2. https://universe.roboflow.com/fire-project/fire-and-smoke-lkzok/dataset/8
License:
CC BY 4.0

Own annotations:
We annotate the provided sample videos to create a development set for
testing and improving the system.

No additional footage from the same camera is collected or scraped.


10. SAMPLE PREDICTIONS
----------------------

Predictions produced on the sample videos are stored in:

    predictions_samples.json

The file follows the required predictions format.


11. RUNNING THE SOLUTION
------------------------

To run the submission on a folder of videos:

    python run_submission.py --videos /data/test --out predictions.json

The generated file is:

    predictions.json


12. VALIDATING THE OUTPUT
-------------------------

Before submission, validate the predictions file with:

    python evaluate.py --pred predictions.json --validate-only

To evaluate against our own development labels:

    python evaluate.py --pred predictions.json --gt my_labels.json


13. REPRODUCIBILITY
-------------------

Random seeds are fixed.

Seed used:

    [SEED NUMBER]

The project is designed so that repeated runs on the same machine produce
the same predictions apart from floating-point noise.

Other non-deterministic components:

    [NONE / DESCRIBE THEM HERE]


14. PERFORMANCE
---------------

The system is designed to stay within the hackathon time limit.

Target evaluation environment:

- 1 NVIDIA GPU
- 16 GB VRAM
- 8 CPU cores
- 32 GB RAM

The total processing time for Part A and Part B should remain within
3 times the duration of the input video.


15. CODE ORGANIZATION
---------------------

solution.py
    Main interface required by the hackathon.

run_submission.py
    Official starter-kit harness used to run the solution.

evaluate.py
    Format checker and evaluation script.

src/
    Project source code, models, tracking and traffic-event rules.

weights/
    Model weights required for offline inference.

predictions_samples.json
    Predictions generated on the sample videos.


16. TEAM
--------

Team name:
TriVision

Team members:

1. Munisa Rustamova - Team Leader
   Contribution: Website + Final Integration

2. Gulrukh Abdurakhimova - AI/Computer Vision Lead
   Contribution: Model + Event Detection

3. Ferangiz Bafoyeva - AI/Research + Accident Anticipation Lead
   Contribution: Risk Prediction + EDA + Evaluation


17. LINKS
---------

GitHub repository:
https://github.com/rustamovamsh/WIUT_Hackathon_TriVision

Website:
[WEBSITE LINK]

Weights:
[WEIGHTS LINK, IF APPLICABLE]

predictions_samples.json:
[LINK, IF APPLICABLE]


18. LIMITATIONS / FAILURE CASES
-------------------------------

Current limitations:

- [LIMITATION 1]
- [LIMITATION 2]
- [LIMITATION 3]

Known failure cases:

- [FAILURE CASE 1]
- [FAILURE CASE 2]


19. FUTURE IMPROVEMENTS
-----------------------

Possible future improvements:

- Improve temporal event boundaries.
- Improve object tracking.
- Improve detection of rare traffic events.
- Improve accident anticipation.
- Improve risk-score calibration.
- Improve robustness to lighting and traffic-density changes.


20. RULES AND COMPLIANCE
------------------------

The project follows the hackathon requirements:

- Open-weight models are used for inference.
- No paid hosted AI APIs are used during inference.
- All required inference weights are available for offline evaluation.
- Public datasets are listed with their licences.
- The RiskEstimator uses only frames received so far.
- Random seeds are fixed.
- Reused open-source code is attributed in this repository.


21. CONTACT
-----------

Team contact:
[email: munisarustamova1006@gmail.com / tg: @rustamovamsh / num: +998973041006]


Last updated:
27.09.2026

# Introduction/purpose of project:
The Servia Learning Platform is a humanoid robot that aims to be a form of support for first-year students. Many first year students face challenges navigating from high school to college, and this stems from the fact that the college environment is very different
from what they are accustomed to in high school. They are expected to be more independent, and do not always receive the support they need. The Servia Learning platform solves this problem by being in an accessible location in the DISC (Room 2306 in SCDI - see
Figure 2) that receives a lot of foot traffic. It will use various machine learning models, from image processing to ChatGPT, as well as the Google Calendar API to accomplish its various tasks, from answering student questions to helping students stay on track to complete their assignments. To clarify, students will have the option to give Servia their calendar to help keep them on track, however, this is not required to work with Servia. It will also refer to human sources of support if needed. 

# What it consists of:
There are four folders in this repository:

**servia_talking**: A ChatGPT powered Python script that allows the robot to hold a conversation and serves as a medium of
interaction between the student and the AI robot system, allowing the robot to offer help and guidance where necessary

**servia_opencv**: A computer vision AI model that determines whether or not students are being productive or distracted.
It is based on ResNet 18, though it has been retrained with the Kaggle Yawn Eye Dataset

**servia_calendar**: Contains all files pertaining to the Google Calendar API. The python file is used to fetch deadlines

**servia_decision**: Where it all comes together. Decision making algorithm will take in input from other three folders to, well make decisions
# Types of Machine Learning

01. **Supervised learning** - uses labeled inputs (meaning the input has a corresponding output label) to train models and learn outputs.

    *Example*: Imagine you have a dataset of images of cars, planes, and motorcycles. Each image is labeled with its corresponding category (car, plane, motorcycle). In supervised learning, you would feed these labeled images into a machine learning algorithm. The algorithm learns from these labeled examples to recognize the features of each category. After training, the model can classify new, unseen images into the correct categories (car, plane, motorcycle) based on what it has learned.

02. **Unsupervised learning** - uses unlabeled data to learn about patterns in data.

    *Example*: Consider the same dataset of images of cars, planes, and motorcycles, but this time without any labels. In unsupervised learning, you would feed these unlabeled images into a machine learning algorithm. The algorithm would look for patterns and similarities in the data to group the images into clusters. For example, it might group all the images of cars together, all the images of planes together, and all the images of motorcycles together, based on the visual features it identifies. This process is called clustering, and it helps in discovering hidden structures in the data.

03. **Reinforcement learning** - agent learning in an interactive environment based on rewards and penalties.

    *Example*: Imagine a robot navigating a maze. The robot (agent) starts at the entrance of the maze and must find its way to the exit. In reinforcement learning, the robot receives feedback in the form of rewards and penalties. For example, it might receive a positive reward for moving closer to the exit and a negative reward for hitting a wall. The robot learns through trial and error, adjusting its actions based on the rewards and penalties it receives. Over time, it develops a strategy (policy) to navigate the maze efficiently and reach the exit. This type of learning is particularly useful for tasks that involve decision-making in dynamic environments.
